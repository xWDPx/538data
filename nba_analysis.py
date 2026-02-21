"""
NBA Analysis & Prediction Script
=================================
Datasets used:
  - nba-raptor/modern_RAPTOR_by_player.csv  (2014–2022, player-level)
  - nba-raptor/modern_RAPTOR_by_team.csv    (2014–2022, team-level, RS + PO)
  - nba-elo/nbaallelo.csv                   (filtered to 2014+)

Outputs (saved to nba_analysis_output/):
  1. elo_win_probability.png   – ELO differential vs actual win rate + logistic model
  2. top_players_raptor.png    – Top 20 players by career WAR (reg season)
  3. team_raptor_trend.png     – Team RAPTOR total over seasons (reg season)
  4. player_comparison.png     – Scatter: raptor_offense vs raptor_defense, sized by WAR
  5. game_prediction_model.txt – Logistic regression accuracy & coefficients
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# ── Output directory ──────────────────────────────────────────────────────────
OUT = os.path.join(os.path.dirname(__file__), "nba_analysis_output")
os.makedirs(OUT, exist_ok=True)

sns.set_theme(style="darkgrid", palette="muted")
TITLE_KW = dict(fontsize=14, fontweight="bold", pad=12)

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
BASE = os.path.dirname(__file__)

print("Loading data...")
rp = pd.read_csv(os.path.join(BASE, "nba-raptor", "modern_RAPTOR_by_player.csv"))
rt = pd.read_csv(os.path.join(BASE, "nba-raptor", "modern_RAPTOR_by_team.csv"))
elo_raw = pd.read_csv(os.path.join(BASE, "nba-elo", "nbaallelo.csv"))

# Filter ELO to modern era & regular season, one row per game (drop _iscopy duplicate)
elo = elo_raw[(elo_raw["year_id"] >= 2014) & (elo_raw["_iscopy"] == 0)].copy()
elo["won"] = (elo["game_result"] == "W").astype(int)
elo["elo_diff"] = elo["elo_i"] - elo["opp_elo_i"]

rs_rt = rt[rt["season_type"] == "RS"].copy()

print(f"  ELO games (modern, deduped): {len(elo):,}")
print(f"  RAPTOR players: {len(rp):,}  |  RAPTOR team-seasons: {len(rs_rt):,}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. ELO: WIN PROBABILITY MODEL
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1/4] ELO win probability model...")

# Bin ELO differential and compute empirical win rate
elo["elo_bin"] = pd.cut(elo["elo_diff"], bins=np.arange(-400, 401, 25))
binned = (
    elo.groupby("elo_bin", observed=True)["won"]
    .agg(win_rate="mean", count="count")
    .reset_index()
)
binned["bin_mid"] = binned["elo_bin"].apply(lambda x: x.mid).astype(float)
binned = binned[binned["count"] >= 5]  # drop sparse bins

# Logistic regression
X = elo[["elo_diff"]].values
y = elo["won"].values
scaler = StandardScaler()
X_sc = scaler.fit_transform(X)
X_tr, X_te, y_tr, y_te = train_test_split(X_sc, y, test_size=0.2, random_state=42)

lr = LogisticRegression()
lr.fit(X_tr, y_tr)
y_pred = lr.predict(X_te)
acc = accuracy_score(y_te, y_pred)
report = classification_report(y_te, y_pred, target_names=["Loss", "Win"])

# Predict smooth curve
smooth_x = np.linspace(-400, 400, 300).reshape(-1, 1)
smooth_x_sc = scaler.transform(smooth_x)
smooth_prob = lr.predict_proba(smooth_x_sc)[:, 1]

fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(
    binned["bin_mid"], binned["win_rate"],
    s=binned["count"] / binned["count"].max() * 200,
    alpha=0.7, label="Empirical win rate (bubble = sample size)", zorder=3
)
ax.plot(smooth_x, smooth_prob, color="tomato", lw=2.5,
        label=f"Logistic model  (test acc = {acc:.1%})")
ax.axhline(0.5, color="gray", ls="--", lw=1)
ax.axvline(0, color="gray", ls="--", lw=1)
ax.set_xlabel("ELO Differential (team − opponent)")
ax.set_ylabel("Win Probability")
ax.set_title("ELO Differential → Win Probability (2014–2022)", **TITLE_KW)
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUT, "elo_win_probability.png"), dpi=150)
plt.close(fig)
print(f"  Model accuracy: {acc:.1%}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. TOP PLAYERS BY CAREER WAR (RAPTOR)
# ─────────────────────────────────────────────────────────────────────────────
print("[2/4] Top players by career WAR...")

career = (
    rp.groupby("player_name", as_index=False)
    .agg(
        career_war=("war_reg_season", "sum"),
        seasons=("season", "nunique"),
        avg_raptor=("raptor_total", "mean"),
        avg_offense=("raptor_offense", "mean"),
        avg_defense=("raptor_defense", "mean"),
    )
    .sort_values("career_war", ascending=False)
)
top20 = career.head(20).iloc[::-1]  # flip for horizontal bar

fig, ax = plt.subplots(figsize=(10, 7))
bars = ax.barh(
    top20["player_name"], top20["career_war"],
    color=sns.color_palette("viridis", len(top20))
)
for bar, val in zip(bars, top20["career_war"]):
    ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}", va="center", fontsize=8)
ax.set_xlabel("Career WAR (Wins Above Replacement, regular season)")
ax.set_title("Top 20 NBA Players by Career WAR — 2014–2022", **TITLE_KW)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "top_players_raptor.png"), dpi=150)
plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# 4. TEAM RAPTOR TREND OVER SEASONS
# ─────────────────────────────────────────────────────────────────────────────
print("[3/4] Team RAPTOR trends...")

# Aggregate to one row per team per season (some players switch teams mid-season)
team_season = (
    rs_rt.groupby(["team", "season"], as_index=False)
    .agg(raptor_total=("raptor_total", "sum"), war=("war_reg_season", "sum"))
)

# Pick top-10 franchises by total WAR for clarity
top_teams = (
    team_season.groupby("team")["war"].sum()
    .nlargest(10).index.tolist()
)
ts_top = team_season[team_season["team"].isin(top_teams)]

fig, ax = plt.subplots(figsize=(12, 6))
for team, grp in ts_top.groupby("team"):
    grp = grp.sort_values("season")
    ax.plot(grp["season"], grp["raptor_total"], marker="o", label=team, lw=2)

ax.set_xlabel("Season")
ax.set_ylabel("Team RAPTOR Total (sum of player RAPTOR)")
ax.set_title("Top-10 Teams: RAPTOR Trajectory 2014–2022", **TITLE_KW)
ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
ax.legend(ncol=2, fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "team_raptor_trend.png"), dpi=150)
plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# 5. PLAYER COMPARISON SCATTER: OFFENSE vs DEFENSE
# ─────────────────────────────────────────────────────────────────────────────
print("[4/4] Player comparison scatter...")

# Use career averages, filter to players with 2+ seasons for reliability
reliable = career[career["seasons"] >= 2].copy()

fig, ax = plt.subplots(figsize=(11, 8))
sc = ax.scatter(
    reliable["avg_offense"], reliable["avg_defense"],
    c=reliable["career_war"], cmap="RdYlGn",
    s=reliable["career_war"].clip(lower=0) * 12 + 20,
    alpha=0.7, edgecolors="gray", linewidths=0.3
)
cbar = fig.colorbar(sc, ax=ax, label="Career WAR")

# Label top-15 by career WAR
for _, row in career.head(15).iterrows():
    ax.annotate(
        row["player_name"].split()[-1],  # last name
        xy=(row["avg_offense"], row["avg_defense"]),
        fontsize=7.5, ha="left",
        xytext=(4, 2), textcoords="offset points"
    )

ax.axhline(0, color="gray", ls="--", lw=1)
ax.axvline(0, color="gray", ls="--", lw=1)
ax.set_xlabel("Avg RAPTOR Offense (pts per 100 poss added)")
ax.set_ylabel("Avg RAPTOR Defense (pts per 100 poss saved)")
ax.set_title("Player Offensive vs Defensive RAPTOR — 2014–2022\n"
             "(size & color = career WAR)", **TITLE_KW)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "player_comparison.png"), dpi=150)
plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# 6. SAVE MODEL REPORT
# ─────────────────────────────────────────────────────────────────────────────
coef = lr.coef_[0][0]
intercept = lr.intercept_[0]

with open(os.path.join(OUT, "game_prediction_model.txt"), "w") as f:
    f.write("NBA Game Outcome Prediction — Logistic Regression (ELO differential)\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"Training samples : {len(X_tr):,}\n")
    f.write(f"Test samples     : {len(X_te):,}\n")
    f.write(f"Test accuracy    : {acc:.2%}\n\n")
    f.write("Classification Report:\n")
    f.write(report + "\n\n")
    f.write("Model coefficients (standardised ELO diff):\n")
    f.write(f"  coef      = {coef:.4f}\n")
    f.write(f"  intercept = {intercept:.4f}\n\n")
    f.write("Quick prediction function:\n")
    f.write("  from sklearn.linear_model import LogisticRegression\n")
    f.write("  # win_prob = logistic(coef * (elo_diff - mean) / std + intercept)\n\n")
    f.write("Top 10 players by career WAR (2014–2022):\n")
    for i, row in career.head(10).iterrows():
        f.write(f"  {row['player_name']:<25}  WAR={row['career_war']:.1f}"
                f"  OFF={row['avg_offense']:+.2f}  DEF={row['avg_defense']:+.2f}\n")

# ─────────────────────────────────────────────────────────────────────────────
# 7. PRINT PREDICTION HELPER
# ─────────────────────────────────────────────────────────────────────────────
print("\n─── Quick game predictor ──────────────────────────────────────────────")
examples = [(1600, 1500), (1650, 1400), (1480, 1520)]
for elo_team, elo_opp in examples:
    diff = elo_team - elo_opp
    diff_sc = scaler.transform([[diff]])
    prob = lr.predict_proba(diff_sc)[0][1]
    print(f"  Team ELO {elo_team} vs Opp ELO {elo_opp}  →  win prob = {prob:.1%}")

print("\n─── Outputs written to: nba_analysis_output/ ──────────────────────────")
print("  elo_win_probability.png")
print("  top_players_raptor.png")
print("  team_raptor_trend.png")
print("  player_comparison.png")
print("  game_prediction_model.txt")
