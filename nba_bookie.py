
"""
NBA Bookie Agent
================
Daily value report generator for NBA betting.

Fetches today's matchups and market odds, then runs them through the 
538-based edge model in nba_edge.py to find value bets.

Market odds sources (in order of preference):
1. CBS Sports web scrape (no API key needed)
2. Fallback: Simulated lines with noise (if scrape fails)

Usage:
    python nba_bookie.py              # Generate daily report
    python nba_bookie.py --mock       # Force mock odds for testing
    python nba_bookie.py --date 2025-01-15  # Analyze specific date

Output:
    - Console summary ranked by EV
    - JSON: nba_analysis_output/bookie_report_YYYYMMDD.json
"""

import os
import sys
import json
import time
import argparse
import warnings
from datetime import datetime, date

# Load shared .env (BALLDONTLIE_API_KEY, ODDS_API_KEY, etc.)
_env_path = "/root/.openclaw/workspace/538data/.env"
if os.path.exists(_env_path):
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())
from typing import Optional, List, Dict, Any
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from scipy import stats

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "nba_analysis_output")
os.makedirs(OUT, exist_ok=True)

# ── Import edge logic from nba_edge.py ──────────────────────────────────────
# We import the functions we need to avoid duplicating the model
sys.path.insert(0, BASE)
try:
    from nba_edge import (
        load_team_ratings,
        predict,
        evaluate_line,
        prob_to_american,
        margin_to_win_prob,
        HCA_PTS,
        SPREAD_SIGMA,
    )
except ImportError:
    print("ERROR: Could not find nba_edge.py in the current directory.")
    sys.exit(1)

# ── Team name mapping ───────────────────────────────────────────────────────
# Map full names to standard abbreviations
TEAM_NAME_TO_ABBR = {
    # Atlantic
    "Boston Celtics": "BOS",
    "Brooklyn Nets": "BKN",
    "New York Knicks": "NYK",
    "Philadelphia 76ers": "PHI",
    "Toronto Raptors": "TOR",
    # Central
    "Chicago Bulls": "CHI",
    "Cleveland Cavaliers": "CLE",
    "Detroit Pistons": "DET",
    "Indiana Pacers": "IND",
    "Milwaukee Bucks": "MIL",
    # Southeast
    "Atlanta Hawks": "ATL",
    "Charlotte Hornets": "CHA",
    "Miami Heat": "MIA",
    "Orlando Magic": "ORL",
    "Washington Wizards": "WAS",
    # Northwest
    "Denver Nuggets": "DEN",
    "Minnesota Timberwolves": "MIN",
    "Oklahoma City Thunder": "OKC",
    "Portland Trail Blazers": "POR",
    "Utah Jazz": "UTA",
    # Pacific
    "Golden State Warriors": "GSW",
    "LA Clippers": "LAC",
    "Los Angeles Clippers": "LAC",
    "Los Angeles Lakers": "LAL",
    "Phoenix Suns": "PHO",
    "Sacramento Kings": "SAC",
    # Southwest
    "Dallas Mavericks": "DAL",
    "Houston Rockets": "HOU",
    "Memphis Grizzlies": "MEM",
    "New Orleans Pelicans": "NOP",
    "San Antonio Spurs": "SAS",
    # URL short codes and common variations
    "ORL": "ORL", "PHO": "PHO", "PHI": "PHI", "NO": "NOP",
    "DET": "DET", "CHI": "CHI", "MEM": "MEM", "MIA": "MIA",
    "SAC": "SAC", "SA": "SAS", "HOU": "HOU", "NY": "NYK",
    "BOS": "BOS", "BKN": "BKN", "TOR": "TOR", "CLE": "CLE",
    "IND": "IND", "MIL": "MIL", "ATL": "ATL", "CHA": "CHA",
    "WAS": "WAS", "DEN": "DEN", "MIN": "MIN", "OKC": "OKC",
    "POR": "POR", "UTA": "UTA", "GS": "GSW", "LAC": "LAC",
    "LAL": "LAL", "PHX": "PHO", "DAL": "DAL", "UTAH": "UTA",
    "GSW": "GSW", "NYK": "NYK", "SAS": "SAS", "NOP": "NOP",
    "NOR": "NOP", "WSH": "WAS", "PHL": "PHI", "BRO": "BKN",
    "BRK": "BKN", "KLAC": "LAC", "PHOENIX": "PHO", "LAKERS": "LAL",
}


# ── Fetch market odds ───────────────────────────────────────────────────────

def fetch_cbssports_odds(target_date: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Scrape odds from CBS Sports NBA odds page.
    Returns list of games with spreads and moneylines.
    """
    url = "https://www.cbssports.com/nba/odds/"
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
    except Exception as e:
        print(f"  [CBS Sports] Failed to fetch: {e}")
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    games = []
    
    # Each game has two rows: away team and home team
    game_containers = soup.find_all("div", class_="table-body")
    
    for container in game_containers:
        rows = container.find_all("tr", class_=["row-away-team", "row-home-team"])
        if len(rows) != 2:
            continue
        
        # Parse away and home teams
        game_teams = []
        for i, row in enumerate(rows):
            cells = row.find_all("td")
            team_info = {"team": None, "spread": None, "ml": None}
            
            for cell in cells:
                # Team abbreviation in link
                team_link = cell.find("a", href=lambda x: x and "/nba/teams/" in x)
                if team_link:
                    href = team_link.get("href", "")
                    parts = href.split("/")
                    if len(parts) >= 3:
                        abbr = parts[-2].upper()
                        team_info["team"] = TEAM_NAME_TO_ABBR.get(abbr, abbr)
                
                # Spread and ML
                text = cell.get_text(strip=True)
                if text.startswith(("-", "+")):
                    try:
                        if "." in text and len(text) < 6:
                            val = float(text)
                            if abs(val) < 50: team_info["spread"] = val
                        elif text[1:].isdigit() and len(text) > 2:
                            val = int(text)
                            if abs(val) > 90: team_info["ml"] = val
                    except:
                        pass
            game_teams.append(team_info)
        
        if game_teams[0]["team"] and game_teams[1]["team"]:
            games.append({
                "away": game_teams[0]["team"],
                "home": game_teams[1]["team"],
                "away_spread": game_teams[0]["spread"],
                "home_spread": game_teams[1]["spread"],
                "away_ml": game_teams[0]["ml"],
                "home_ml": game_teams[1]["ml"],
            })
    
    return games


def generate_mock_odds(games: List[Dict[str, str]], ratings: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Generate mock lines based on model predictions with realistic noise.
    Used when real odds APIs are unavailable.
    """
    mock_games = []
    for game in games:
        away = TEAM_NAME_TO_ABBR.get(game["away"], game["away"])
        home = TEAM_NAME_TO_ABBR.get(game["home"], game["home"])
        
        try:
            pred = predict(away, home, home, ratings)
            pred_margin = pred["predicted_margin_ats"]  # Negative means home favored
            
            noise = np.random.normal(0, 1.2)
            market_spread = -(pred_margin + noise)
            market_spread = round(market_spread * 2) / 2
            
            if market_spread >= 0:
                home_ml = int(-110 - market_spread * 40)
                away_ml = int(110 + market_spread * 35)
            else:
                home_ml = int(110 - market_spread * 35)
                away_ml = int(-110 + market_spread * 40)
            
            mock_games.append({
                "home": home, "away": away,
                "home_spread": market_spread, "away_spread": -market_spread,
                "home_ml": home_ml, "away_ml": away_ml,
                "mock": True,
            })
        except Exception as e:
            print(f"  [Mock] Skipping {away}@{home}: {e}")
    return mock_games


# ── Fetch today's games ─────────────────────────────────────────────────────

def fetch_todays_games(target_date: Optional[str] = None) -> List[Dict[str, str]]:
    """Fetch today's NBA games using BallDontLie (BDL) as priority, fallback to nba_api."""
    bdl_api_key = os.environ.get("BALLDONTLIE_API_KEY")
    if bdl_api_key:
        try:
            d = target_date or datetime.now().strftime("%Y-%m-%d")
            url = f"https://api.balldontlie.io/nba/v1/games?dates[]={d}"
            headers = {"Authorization": bdl_api_key}
            resp = requests.get(url, headers=headers, timeout=15)
            if resp.status_code == 200:
                data = resp.json().get("data", [])
                if data:
                    return [{"home": g["home_team"]["abbreviation"], "away": g["visitor_team"]["abbreviation"]} for g in data]
        except Exception as e:
            print(f"  [BDL] Matchup fetch failed: {e}")

    try:
        from nba_api.stats.endpoints import scoreboardv2
        raw_d = target_date or datetime.now().strftime("%Y-%m-%d")
        d = datetime.strptime(raw_d, "%Y-%m-%d").strftime("%m/%d/%Y")
        board = scoreboardv2.ScoreboardV2(game_date=d, timeout=15)
        df = board.get_data_frames()[0]
        return [{"home": r["HOME_TEAM_ABBREVIATION"], "away": r["VISITOR_TEAM_ABBREVIATION"]} for _, r in df.iterrows()]
    except Exception as e:
        print(f"  [nba_api] Matchup fetch failed: {e}")
    return []


# ── Core analysis ───────────────────────────────────────────────────────────

def analyze_game(home, away, home_spread, home_ml, away_ml, ratings) -> Dict[str, Any]:
    pred_home = predict(home, away, home, ratings)
    pred_away = predict(away, home, home, ratings)

    res = {"game": f"{away} @ {home}", "home_team": home, "away_team": away, "home": {}, "away": {}}

    if home_ml:
        eval_h = evaluate_line(pred_home, home_spread, float(home_ml))
        res["home"] = {
            "moneyline": home_ml, "spread": home_spread, "fair_ml": pred_home["fair_american"],
            "edge_pct": eval_h.get("edge_pct", 0), "ev_per_100": eval_h.get("ev_per_100", 0),
            "verdict": eval_h.get("verdict", "PASS"), "win_prob": pred_home["win_prob_ats"]
        }
    if away_ml:
        eval_a = evaluate_line(pred_away, -home_spread if home_spread else None, float(away_ml))
        res["away"] = {
            "moneyline": away_ml, "spread": -home_spread if home_spread else None, "fair_ml": pred_away["fair_american"],
            "edge_pct": eval_a.get("edge_pct", 0), "ev_per_100": eval_a.get("ev_per_100", 0),
            "verdict": eval_a.get("verdict", "PASS"), "win_prob": pred_away["win_prob_ats"]
        }

    # Spread bet evaluation — assumes standard -110 juice when no spread odds scraped
    if home_spread is not None:
        eval_h_spd = evaluate_line(pred_home, home_spread, -110)
        res["home"].update({
            "spread_cover_prob": eval_h_spd.get("cover_prob"),
            "spread_edge_pct": eval_h_spd.get("edge_pct", 0),
            "spread_ev_per_100": eval_h_spd.get("ev_per_100", 0),
            "spread_verdict": eval_h_spd.get("verdict", "PASS"),
        })
        eval_a_spd = evaluate_line(pred_away, -home_spread, -110)
        res["away"].update({
            "spread_cover_prob": eval_a_spd.get("cover_prob"),
            "spread_edge_pct": eval_a_spd.get("edge_pct", 0),
            "spread_ev_per_100": eval_a_spd.get("ev_per_100", 0),
            "spread_verdict": eval_a_spd.get("verdict", "PASS"),
        })

    return res


def print_report_console(report: Dict[str, Any]):
    sep = "=" * 76
    print(f"\n{sep}")
    print(f"  NBA BOOKIE DAILY VALUE REPORT - {report['report_date']}")
    print(sep)
    
    for category, label in [("strong_edges", "🔥 STRONG EDGES (>=5% edge)"), 
                            ("edges", "📈 EDGES (3-5% edge)"), 
                            ("marginals", "⚠️  MARGINAL (2-3% edge)")]:
        items = report.get(category, [])
        if items:
            print(f"\n  {label}:")
            print(f"  {'Team':<8} {'Type':<6} {'Game':<22} {'Line':<10} {'Fair':<10} {'Edge':<8} {'EV/$100':<10}")
            print(f"  {'-'*74}")
            for b in items[:5]:
                line = f"{b['line']:+.1f}" if b['type'] == 'spread' else f"{b['line']:.0f}"
                fair = f"{b.get('fair_ml', 'n/a')}"
                ev = f"${b.get('ev_per_100', 0):>+.2f}"
                print(f"  {b['team']:<8} {b['type']:<6} {b['game']:<22} {line:<10} {fair:<10} {b['edge_pct']:>+.1%}  {ev}")

    if not any(report.get(k) for k in ["strong_edges", "edges", "marginals"]):
        print(f"\n  No significant edges found today based on model confidence.")

    print(f"\n{sep}")
    print(f"  Logs: nba_analysis_output/bookie_report_{report['report_date']}.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, default=None)
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()
    
    target_date = args.date or datetime.now().strftime("%Y-%m-%d")
    ratings = load_team_ratings()
    games = fetch_todays_games(target_date)
    
    if not games:
        # Fallback to common matchups if API is down, or just exit
        print("  [nba_api] No live games found. Check your connection or the schedule.")
        return

    market = fetch_cbssports_odds() if not args.mock else []
    if not market or args.mock:
        market = generate_mock_odds(games, ratings)
    
    results = []
    for g in market:
        try:
            results.append(analyze_game(g["home"], g["away"], g.get("home_spread"), g.get("home_ml"), g.get("away_ml"), ratings))
        except: continue
        
    all_bets = []
    for g in results:
        for side in ["home", "away"]:
            d = g[side]
            if not d:
                continue
            # Moneyline bet
            if d.get("moneyline"):
                all_bets.append({
                    "game": g["game"], "team": g[f"{side}_team"], "type": "moneyline",
                    "line": d["moneyline"], "fair_ml": d["fair_ml"],
                    "edge_pct": d["edge_pct"], "ev_per_100": d["ev_per_100"],
                    "verdict": d["verdict"],
                })
            # Spread bet
            if d.get("spread") is not None and d.get("spread_verdict"):
                cover = d.get("spread_cover_prob")
                all_bets.append({
                    "game": g["game"], "team": g[f"{side}_team"], "type": "spread",
                    "line": d["spread"],
                    "fair_ml": f"{cover:.1%}" if cover is not None else "n/a",
                    "edge_pct": d.get("spread_edge_pct", 0),
                    "ev_per_100": d.get("spread_ev_per_100", 0),
                    "verdict": d.get("spread_verdict", "PASS"),
                })
    
    all_bets.sort(key=lambda x: x["ev_per_100"], reverse=True)
    report = {
        "report_date": target_date,
        "strong_edges": [b for b in all_bets if b["verdict"] == "STRONG EDGE"],
        "edges": [b for b in all_bets if b["verdict"] == "EDGE"],
        "marginals": [b for b in all_bets if b["verdict"] == "MARGINAL"],
        "all_bets": all_bets
    }
    
    with open(os.path.join(OUT, f"bookie_report_{target_date}.json"), "w") as f:
        json.dump(report, f, indent=2)
    
    print_report_console(report)

if __name__ == "__main__":
    main()
