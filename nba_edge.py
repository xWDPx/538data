"""
NBA Edge Finder
===============
Uses 538 historical data + current season team ratings to generate:
  - Fair point spread  (predicted margin ± confidence interval)
  - Fair moneyline     (win probability → American odds)
  - Cover probability  (if you supply a posted spread)
  - Expected value     (if you supply a posted moneyline)
  - Team ATS tendency  (how much each team historically beats/misses the model)

Model calibration (from 538 ELO data, 2010–2015, ~9 000 games):
  predicted_margin = 0.035 * elo_diff + 1.807 * is_home + 1.042
  residual σ ≈ 12 points  →  win_prob = Φ(predicted_margin / σ)

Current team strength sourced from nba_live.py (nba_api or RAPTOR fallback).

Usage:
  python nba_edge.py --team1 BOS --team2 GSW
  python nba_edge.py --team1 BOS --team2 GSW --home BOS
  python nba_edge.py --team1 BOS --team2 GSW --spread -4.5
  python nba_edge.py --team1 BOS --team2 GSW --spread -4.5 --moneyline -190
  python nba_edge.py --team1 BOS --team2 GSW --neutral      (neutral site)
"""

import os, argparse, warnings
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

warnings.filterwarnings("ignore")
sns.set_theme(style="darkgrid", palette="muted")

BASE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(BASE, "nba_analysis_output")
os.makedirs(OUT, exist_ok=True)

TITLE_KW = dict(fontsize=13, fontweight="bold", pad=10)

# ── Model constants (calibrated on 538 ELO data 2010-2015) ───────────────────
SPREAD_SIGMA = 11.95   # residual std dev in points
HCA_PTS      = 3.0     # home court advantage in points (rounded from regression)
# NET_RATING is pts per 100 poss; scale to a per-game prediction
# (league avg ~100 poss/game → NET_RATING ≈ expected pt diff per game)
# ELO calibration: 1 NET_RATING pt ≈ 28.5 ELO pts (from 0.035 pts/ELO)
NET_RATING_TO_ELO = 28.5

# ── ATS history (pre-computed from 538 data, 2010–2015) ──────────────────────
# mean outperformance vs ELO-predicted spread (positive = beats the spread more)
ATS_HISTORY = {
    "SAS": +2.00, "LAC": +1.82, "GSW": +1.80, "OKC": +1.62, "IND": +1.43,
    "DEN": +1.20, "POR": +0.86, "UTA": +0.77, "HOU": +0.60, "MEM": +0.52,
    "ATL": +0.47, "DAL": +0.44, "MIA": +0.38, "MIN": +0.24, "NYK": +0.16,
    "SAC": -0.06, "DET": -0.21, "WAS": -0.33, "BOS": -0.38, "ORL": -0.44,
    "TOR": -0.55, "LAL": -0.64, "CHI": -0.72, "NOP": -0.80, "PHO": -0.90,
    "MIL": -1.46, "PHI": -1.59, "CLE": -1.25, "BKN": -0.90, "CHA": -1.20,
}


# ── Probability / odds helpers ────────────────────────────────────────────────

def margin_to_win_prob(predicted_margin: float) -> float:
    """P(team wins) using normal spread distribution (σ≈12 pts)."""
    return 1 - stats.norm.cdf(0, loc=predicted_margin, scale=SPREAD_SIGMA)

def cover_prob(predicted_margin: float, posted_spread: float) -> float:
    """
    P(team covers posted_spread).
    posted_spread < 0  → team is favourite (e.g. -4.5 means must win by 5+)
    posted_spread > 0  → team is dog     (e.g. +4.5 means can lose by up to 4)
    Bettor wins if actual_margin > -posted_spread
    """
    threshold = -posted_spread
    return 1 - stats.norm.cdf(threshold, loc=predicted_margin, scale=SPREAD_SIGMA)

def prob_to_american(p: float) -> str:
    """Convert win probability to fair American odds string."""
    if p <= 0 or p >= 1:
        return "n/a"
    if p >= 0.5:
        american = -(p / (1 - p)) * 100
        return f"{american:+.0f}"
    else:
        american = ((1 - p) / p) * 100
        return f"+{american:.0f}"

def american_to_implied_prob(american: float) -> float:
    if american > 0:
        return 100 / (american + 100)
    else:
        return abs(american) / (abs(american) + 100)

def prob_to_decimal(p: float) -> float:
    return 1 / p if p > 0 else float("inf")

def expected_value(our_prob: float, posted_american: float, stake: float = 100) -> float:
    """
    EV of a $stake bet given our win probability vs posted American odds.
    Positive EV = attractive bet.
    """
    if posted_american > 0:
        profit_if_win = stake * posted_american / 100
    else:
        profit_if_win = stake * 100 / abs(posted_american)
    return our_prob * profit_if_win - (1 - our_prob) * stake


# ── Team strength (current season) ───────────────────────────────────────────

CACHE_DIR  = os.path.join(OUT, "cache")
SEASON     = "2024-25"

def _nba_api_team_stats(force_refresh=False):
    """Try nba_api leaguedashteamstats; cache result; return (df, 'live'/'cache')."""
    import json, time
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, "team_stats_current.json")

    if not force_refresh and os.path.exists(cache_path):
        with open(cache_path) as f:
            data = json.load(f)
        return pd.DataFrame(data["rows"], columns=data["columns"]), "cache"

    try:
        from nba_api.stats.endpoints import leaguedashteamstats
        obj = leaguedashteamstats.LeagueDashTeamStats(
            season=SEASON,
            season_type_all_star="Regular Season",
            per_mode_simple="PerGame",
            timeout=15,
        )
        time.sleep(0.6)
        df = obj.get_data_frames()[0]
        with open(cache_path, "w") as f:
            json.dump({"columns": list(df.columns), "rows": df.values.tolist()}, f)
        return df, "live"
    except Exception as exc:
        print(f"  [nba_api] unavailable ({type(exc).__name__}), using synthetic data")
        return None, "synthetic"

def _synthetic_team_stats():
    """Derive current-season team ratings from 538 RAPTOR data (2022 season)."""
    rt  = pd.read_csv(os.path.join(BASE, "nba-raptor", "modern_RAPTOR_by_team.csv"))
    rs  = rt[(rt["season_type"] == "RS") & (rt["season"] == 2022)].copy()
    agg = rs.groupby("team", as_index=False).agg(
        raptor_off=("raptor_offense", "mean"),
        raptor_def=("raptor_defense", "mean"),
        raptor_tot=("raptor_total",   "mean"),
        war=       ("war_reg_season", "sum"),
    )
    agg["TEAM_ABBREVIATION"] = agg["team"]
    agg["OFF_RATING"] = 112 + agg["raptor_off"] * 1.8
    agg["DEF_RATING"] = 112 - agg["raptor_def"] * 1.8
    agg["NET_RATING"] = agg["OFF_RATING"] - agg["DEF_RATING"]
    return agg

def load_team_ratings(force_refresh=False) -> pd.DataFrame:
    """
    Load current-season team NET_RATING / OFF_RATING / DEF_RATING.
    Priority: nba_api live → cached response → RAPTOR-derived synthetic.
    No dependency on nba_live.py.
    """
    df, src = _nba_api_team_stats(force_refresh)
    if df is None:
        df  = _synthetic_team_stats()
        src = "synthetic"

    abbr_col = "TEAM_ABBREVIATION" if "TEAM_ABBREVIATION" in df.columns else "team"
    net_col  = "NET_RATING"        if "NET_RATING"        in df.columns else "raptor_tot"
    off_col  = "OFF_RATING"        if "OFF_RATING"        in df.columns else "raptor_off"
    def_col  = "DEF_RATING"        if "DEF_RATING"        in df.columns else "raptor_def"
    war_col  = "war"               if "war"               in df.columns else "war_reg_season"

    out = df[[abbr_col, net_col, off_col, def_col]].copy()
    out.columns = ["TEAM", "NET_RATING", "OFF_RATING", "DEF_RATING"]
    out["WAR"]      = df[war_col].values if war_col in df.columns else 0.0
    out["ATS_BIAS"] = out["TEAM"].map(ATS_HISTORY).fillna(0.0)

    print(f"  Team ratings loaded [{src}]")
    return out.set_index("TEAM")


# ── Core prediction ───────────────────────────────────────────────────────────

def predict(team1: str, team2: str, home: str | None,
            team_ratings: pd.DataFrame) -> dict:
    """
    Generate full prediction for team1 vs team2.
    home = team1 | team2 | None (neutral)
    """
    def get(abbr):
        # Exact match first
        if abbr in team_ratings.index:
            return team_ratings.loc[abbr]
        # Fuzzy: first 3 chars
        matches = [i for i in team_ratings.index if i.startswith(abbr[:3])]
        if matches:
            return team_ratings.loc[matches[0]]
        raise ValueError(f"Team '{abbr}' not found in ratings. "
                         f"Available: {sorted(team_ratings.index.tolist())}")

    r1, r2 = get(team1), get(team2)

    is_home_1 = 1 if home == team1 else (-1 if home == team2 else 0)
    hca_margin = HCA_PTS * is_home_1   # positive if team1 is home

    net_diff = float(r1["NET_RATING"]) - float(r2["NET_RATING"])
    predicted_margin = net_diff + hca_margin

    # Adjust for historical ATS bias
    ats1 = float(r1.get("ATS_BIAS", 0))
    ats2 = float(r2.get("ATS_BIAS", 0))
    ats_adj = ats1 - ats2
    predicted_margin_ats = predicted_margin + ats_adj

    win_prob_raw = margin_to_win_prob(predicted_margin)
    win_prob_ats = margin_to_win_prob(predicted_margin_ats)

    # 90% CI on the spread
    ci_lo = predicted_margin_ats + stats.norm.ppf(0.05) * SPREAD_SIGMA
    ci_hi = predicted_margin_ats + stats.norm.ppf(0.95) * SPREAD_SIGMA

    return {
        "team1": team1, "team2": team2, "home": home,
        "net_diff": net_diff,
        "hca_margin": hca_margin,
        "ats_adj": ats_adj,
        "predicted_margin": predicted_margin,
        "predicted_margin_ats": predicted_margin_ats,
        "win_prob_raw": win_prob_raw,
        "win_prob_ats": win_prob_ats,
        "ci_lo": ci_lo, "ci_hi": ci_hi,
        "fair_spread": -predicted_margin_ats,   # from team1's perspective as favourite
        "fair_american": prob_to_american(win_prob_ats),
        "r1": r1, "r2": r2,
    }


# ── Edge evaluation (vs a posted line) ───────────────────────────────────────

def evaluate_line(pred: dict, posted_spread: float | None,
                  posted_american: float | None) -> dict:
    result = {}

    if posted_spread is not None:
        cp = cover_prob(pred["predicted_margin_ats"], posted_spread)
        market_implied = american_to_implied_prob(
            posted_american) if posted_american else 0.5
        result["cover_prob"] = cp
        result["posted_spread"] = posted_spread
        result["spread_edge"] = pred["predicted_margin_ats"] - (-posted_spread)

    if posted_american is not None:
        market_prob = american_to_implied_prob(posted_american)
        ev = expected_value(pred["win_prob_ats"], posted_american)
        edge_pct = pred["win_prob_ats"] - market_prob
        result["posted_american"] = posted_american
        result["market_implied_prob"] = market_prob
        result["edge_pct"] = edge_pct
        result["ev_per_100"] = ev
        # Verdict tiers
        if edge_pct >= 0.05:
            result["verdict"] = "STRONG EDGE"
        elif edge_pct >= 0.03:
            result["verdict"] = "EDGE"
        elif edge_pct >= 0.02:
            result["verdict"] = "MARGINAL"
        else:
            result["verdict"] = "PASS"
        result["attractive"] = edge_pct >= 0.02

    return result


# ── ATS overview for all teams ────────────────────────────────────────────────

def ats_overview(team_ratings: pd.DataFrame) -> pd.DataFrame:
    """Build a ranked table of all teams by ATS tendency + current ratings."""
    df = team_ratings.reset_index().copy()
    df = df.sort_values("ATS_BIAS", ascending=False)
    df["Fair ML (home)"] = df.apply(
        lambda r: prob_to_american(
            margin_to_win_prob(r["NET_RATING"] + HCA_PTS)), axis=1)
    return df[["TEAM","NET_RATING","OFF_RATING","DEF_RATING","ATS_BIAS","Fair ML (home)"]]


# ── Visualisations ────────────────────────────────────────────────────────────

def plot_calibration(ax=None):
    """
    Plot ELO differential bins vs actual win rate and our normal-model curve.
    Proves the spread model is well-calibrated on historical data.
    """
    elo = pd.read_csv(os.path.join(BASE, "nba-elo", "nbaallelo.csv"))
    g = elo[(elo["_iscopy"] == 0) & (elo["year_id"] >= 2010)].copy()
    g["elo_diff"] = g["elo_i"] - g["opp_elo_i"]
    g["won"]      = (g["game_result"] == "W").astype(int)
    g["margin"]   = g["pts"] - g["opp_pts"]

    bins = pd.cut(g["elo_diff"], bins=15)
    tbl  = g.groupby(bins, observed=True).agg(
        win_rate=("won", "mean"),
        avg_margin=("margin", "mean"),
        n=("won", "count"),
    ).reset_index()
    tbl["elo_mid"] = tbl["elo_diff"].apply(lambda x: x.mid)
    tbl = tbl[tbl["n"] >= 20]

    # Model curve: win_prob = Φ(predicted_margin / σ)
    xs   = np.linspace(-400, 400, 200)
    pred = 0.03515 * xs + 1.042          # neutral site
    yhat = 1 - stats.norm.cdf(0, loc=pred, scale=SPREAD_SIGMA)

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(9, 5))

    ax.scatter(tbl["elo_mid"], tbl["win_rate"], s=tbl["n"] / 8,
               alpha=0.8, color="#457b9d", label="Actual win rate (bins)")
    ax.plot(xs, yhat, color="#e63946", lw=2, label="Model: Φ(predicted margin / σ)")
    ax.axhline(0.5, color="gray", ls="--", lw=1)
    ax.axvline(0,   color="gray", ls="--", lw=1)
    ax.set_xlabel("ELO differential (team − opponent)")
    ax.set_ylabel("Win probability")
    ax.set_title("Model Calibration — ELO diff vs actual win rate", **TITLE_KW)
    ax.legend()

    if standalone:
        fig.tight_layout()
        path = os.path.join(OUT, "model_calibration.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Saved: model_calibration.png")


def plot_spread_distribution(pred: dict, posted_spread: float | None):
    fig, ax = plt.subplots(figsize=(10, 5))
    t1, t2  = pred["team1"], pred["team2"]
    mu      = pred["predicted_margin_ats"]

    xs = np.linspace(mu - 4 * SPREAD_SIGMA, mu + 4 * SPREAD_SIGMA, 500)
    ys = stats.norm.pdf(xs, mu, SPREAD_SIGMA)

    # Shade win region (margin > 0)
    ax.fill_between(xs, ys, where=(xs > 0), alpha=0.35, color="#2a9d8f",
                    label=f"{t1} wins ({pred['win_prob_ats']:.1%})")
    ax.fill_between(xs, ys, where=(xs <= 0), alpha=0.35, color="#e63946",
                    label=f"{t2} wins ({1-pred['win_prob_ats']:.1%})")
    ax.plot(xs, ys, color="white", lw=1.5)

    ax.axvline(mu, color="gold", lw=2, ls="-",  label=f"Predicted margin: {mu:+.1f}")
    ax.axvline(0,  color="gray",  lw=1, ls="--", label="Break-even (push)")

    if posted_spread is not None:
        cover_line = -posted_spread
        cp = cover_prob(mu, posted_spread)
        ax.axvline(cover_line, color="white", lw=1.5, ls=":",
                   label=f"Cover line (spread {posted_spread:+.1f}): {cp:.1%} cover")

    ax.set_xlabel(f"Margin of victory for {t1} (pts)")
    ax.set_ylabel("Probability density")
    home_str = f"(home)" if pred["home"] == t1 else f"(away)" if pred["home"] == t2 else "(neutral)"
    ax.set_title(f"Predicted margin distribution — {t1} {home_str} vs {t2}", **TITLE_KW)
    ax.legend(fontsize=9)
    fig.tight_layout()
    path = os.path.join(OUT, "spread_distribution.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: spread_distribution.png")


def plot_team_ats(team_ratings: pd.DataFrame):
    df = ats_overview(team_ratings).copy()
    df = df.sort_values("ATS_BIAS")
    colors = ["#e63946" if v < 0 else "#2a9d8f" for v in df["ATS_BIAS"]]

    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(df["TEAM"], df["ATS_BIAS"], color=colors, edgecolor="none")
    ax.axvline(0, color="white", lw=1)
    ax.set_xlabel("Avg pts above/below ELO-predicted margin")
    ax.set_title("Team ATS Tendency vs ELO Model (2010–2015 historical)", **TITLE_KW)
    fig.tight_layout()
    path = os.path.join(OUT, "team_ats.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: team_ats.png")


def plot_edge_summary(pred: dict, line_eval: dict):
    t1, t2 = pred["team1"], pred["team2"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # ── Panel 1: Win probability ───────────────────────────────────────────
    ax = axes[0]
    p1, p2 = pred["win_prob_ats"], 1 - pred["win_prob_ats"]
    ax.bar([t1, t2], [p1, p2],
           color=["#2a9d8f" if p1 > p2 else "#adb5bd",
                  "#2a9d8f" if p2 > p1 else "#adb5bd"])
    for x, v in enumerate([p1, p2]):
        ax.text(x, v + 0.01, f"{v:.1%}", ha="center", fontweight="bold", fontsize=12)
    ax.set_ylim(0, 1.15)
    ax.set_title("Win Probability", **TITLE_KW)
    ax.axhline(0.5, color="gray", ls="--", lw=1)

    # ── Panel 2: Spread breakdown ──────────────────────────────────────────
    ax = axes[1]
    labels = ["Net Rating\ndiff", "Home Court\nadv", "ATS bias\nadj", "Fair Spread"]
    vals   = [pred["net_diff"], pred["hca_margin"], pred["ats_adj"],
              pred["predicted_margin_ats"]]
    colors2 = ["#457b9d" if v >= 0 else "#e63946" for v in vals]
    ax.bar(labels, vals, color=colors2, edgecolor="none")
    ax.axhline(0, color="gray", ls="--", lw=1)
    ax.set_ylabel(f"Points (+ = favours {t1})")
    ax.set_title("Spread Components", **TITLE_KW)
    for i, v in enumerate(vals):
        ax.text(i, v + (0.15 if v >= 0 else -0.4), f"{v:+.1f}",
                ha="center", fontsize=10, fontweight="bold")

    # ── Panel 3: EV / edge (if line provided) ─────────────────────────────
    ax = axes[2]
    if line_eval:
        items, item_vals, item_colors = [], [], []
        if "edge_pct" in line_eval:
            items.append("Edge\n(our prob − market)")
            item_vals.append(line_eval["edge_pct"] * 100)
            item_colors.append("#2a9d8f" if line_eval["edge_pct"] > 0 else "#e63946")
        if "ev_per_100" in line_eval:
            items.append("EV\n(per $100 bet)")
            item_vals.append(line_eval["ev_per_100"])
            item_colors.append("#2a9d8f" if line_eval["ev_per_100"] > 0 else "#e63946")
        if "cover_prob" in line_eval:
            items.append("Cover\nProbability")
            item_vals.append(line_eval["cover_prob"] * 100)
            item_colors.append("#2a9d8f" if line_eval["cover_prob"] > 0.5 else "#e63946")

        ax.bar(items, item_vals, color=item_colors, edgecolor="none")
        ax.axhline(0, color="gray", ls="--", lw=1)
        for i, v in enumerate(item_vals):
            ax.text(i, v + (0.3 if v >= 0 else -1.5), f"{v:+.1f}",
                    ha="center", fontsize=10, fontweight="bold")
        verdict = line_eval.get("verdict", "PASS")
        color   = "#2a9d8f" if line_eval.get("attractive") else "#e63946"
        ax.set_title(f"Line Evaluation — {verdict}", color=color, **TITLE_KW)
    else:
        # No line provided — show fair lines
        ax.text(0.5, 0.6, f"Fair spread: {pred['fair_spread']:+.1f}",
                ha="center", transform=ax.transAxes, fontsize=14)
        ax.text(0.5, 0.4, f"Fair ML: {pred['fair_american']}",
                ha="center", transform=ax.transAxes, fontsize=14)
        ax.text(0.5, 0.2, "Compare these to\nyour sportsbook's line",
                ha="center", transform=ax.transAxes, fontsize=10, color="#adb5bd")
        ax.set_title("Fair Lines", **TITLE_KW)
        ax.axis("off")

    fig.suptitle(f"{t1} vs {t2} — Edge Analysis", fontsize=15, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(OUT, "edge_summary.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: edge_summary.png")


# ── Console report ────────────────────────────────────────────────────────────

def print_report(pred: dict, line_eval: dict):
    t1, t2 = pred["team1"], pred["team2"]
    home_label = f"  ({pred['home']} is home)" if pred["home"] else "  (neutral site)"
    sep = "=" * 56

    print(f"\n{sep}")
    print(f"  NBA Edge Report: {t1} vs {t2}{home_label}")
    print(sep)

    print(f"\n  MODEL PREDICTION")
    print(f"  {'Net rating diff':30s} {pred['net_diff']:+.2f} pts")
    print(f"  {'Home court adjustment':30s} {pred['hca_margin']:+.2f} pts")
    print(f"  {'ATS historical bias':30s} {pred['ats_adj']:+.2f} pts")
    print(f"  {'─'*40}")
    m = pred['predicted_margin_ats']
    ci_lo, ci_hi = pred['ci_lo'], pred['ci_hi']
    print(f"  {'Predicted margin (for ' + t1 + ')':30s} {m:+.1f} pts")
    print(f"  {'90% CI':30s} [{ci_lo:+.1f},  {ci_hi:+.1f}]")
    print(f"\n  FAIR LINES (no vig)")
    print(f"  {'Spread':30s} {pred['fair_spread']:+.1f}  (for {t1})")
    print(f"  {'Win probability':30s} {t1}: {pred['win_prob_ats']:.1%}  |  "
          f"{t2}: {1-pred['win_prob_ats']:.1%}")
    print(f"  {'Moneyline':30s} {t1}: {pred['fair_american']}  |  "
          f"{t2}: {prob_to_american(1-pred['win_prob_ats'])}")

    if line_eval:
        print(f"\n  LINE EVALUATION")
        if "posted_spread" in line_eval:
            ps = line_eval['posted_spread']
            cp = line_eval['cover_prob']
            se = line_eval['spread_edge']
            print(f"  {'Posted spread (for ' + t1 + ')':30s} {ps:+.1f}")
            print(f"  {'Cover probability':30s} {cp:.1%}")
            print(f"  {'Spread edge':30s} {se:+.1f} pts vs posted line")
        if "posted_american" in line_eval:
            pa   = line_eval['posted_american']
            mkt  = line_eval['market_implied_prob']
            edge = line_eval['edge_pct']
            ev   = line_eval['ev_per_100']
            print(f"  {'Posted moneyline':30s} {pa:+.0f}")
            print(f"  {'Market implied prob':30s} {mkt:.1%}")
            print(f"  {'Our win probability':30s} {pred['win_prob_ats']:.1%}")
            print(f"  {'Edge':30s} {edge:+.1%}")
            print(f"  {'EV per $100 bet':30s} ${ev:+.2f}")
            v = line_eval.get("verdict", "PASS")
            icon = "✓" if line_eval.get("attractive") else "✗"
            print(f"\n  {'Verdict':30s} {icon} {v}")
    else:
        print(f"\n  Tip: add --spread -4.5 --moneyline -190 to evaluate a specific line.")

    print(f"\n{sep}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="NBA edge finder")
    parser.add_argument("--team1",     required=True,  help="Team 1 abbreviation (the bet subject)")
    parser.add_argument("--team2",     required=True,  help="Team 2 abbreviation (the opponent)")
    parser.add_argument("--home",      default=None,   help="Which team is home (default: team1)")
    parser.add_argument("--neutral",   action="store_true", help="Neutral site (no HCA)")
    parser.add_argument("--spread",    type=float, default=None,
                        help="Posted spread for team1 (e.g. -4.5 or +2.0)")
    parser.add_argument("--moneyline", type=float, default=None,
                        help="Posted American moneyline for team1 (e.g. -190 or +155)")
    parser.add_argument("--all-teams", action="store_true",
                        help="Show ATS overview for all 30 teams and exit")
    args = parser.parse_args()

    print("\nLoading team ratings...")
    ratings = load_team_ratings()

    if args.all_teams:
        tbl = ats_overview(ratings)
        print("\n" + tbl.to_string(index=False))
        plot_team_ats(ratings)
        plot_calibration()
        print("\nDone.")
        return

    t1   = args.team1.upper()
    t2   = args.team2.upper()
    home = None if args.neutral else (args.home.upper() if args.home else t1)

    print("Running prediction...")
    pred = predict(t1, t2, home, ratings)

    line_eval = {}
    if args.spread is not None or args.moneyline is not None:
        line_eval = evaluate_line(pred, args.spread, args.moneyline)

    print("Generating charts...")
    plot_calibration()
    plot_spread_distribution(pred, args.spread)
    plot_team_ats(ratings)
    plot_edge_summary(pred, line_eval)

    print_report(pred, line_eval)
    print("All outputs in nba_analysis_output/")


if __name__ == "__main__":
    main()
