"""
WHSDSC 2026 — Phase 1a (FINAL)
================================
Power Rankings (32 teams) + Round 1 win probabilities (16 matchups)

Methodology
-----------
Power Rankings — weighted composite of 4 z-scored metrics:
  xGD/game    (40%)  — xG for minus xG against per game. r=0.67 with win%.
                        Strongest predictor of true team quality.
  GSAx/game   (30%)  — Goals Saved Above Expected per game = (xGA - GA) / GP.
                        r=0.47. Independently captures goaltending; partial r=0.75
                        after controlling for xGD, confirming it adds real new info.
  Finishing   (15%)  — (GF - xGF) / GP. r=0.23. Noisier signal, weighted less.
  Points      (15%)  — W×2 + OTL. Near-perfect proxy for wins but needed to
                        correctly order teams with identical process metrics.

  Tested and excluded:
  - PP xG/60 and PK xGA/60: raw r=0.43/0.52 looks attractive, but partial
    correlation controlling for xGD collapses to r=-0.05/-0.16. They are
    proxies for xGD, not independent signals. France jumps to #9 with ST
    included (37W, 45L, negative GSAx) — clearly wrong. Excluded.
  - OT rate: r≈0 with win%, xGD, or points. Pure noise in this league.
  - Net penalty: r=0.16 with win%. Too weak and partially captured by xGD.
  - home_max_xg: r=0.74 with xGD — fully redundant.
  - assists: r=0.93 with goals — fully redundant.

Win Probability — logistic regression:
  Features: dxg, dgsax, dfinish (z-score differences), home advantage
  Tuning: GridSearchCV over C=[0.01..10], scoring=neg_log_loss, 5-fold CV
  PP/PK excluded from model too — coefficients shrink to 0.006/0.015 at
  best C=0.01, confirming xGD subsumes the signal. Log loss increases
  when ST added.
  Performance: CV accuracy=57.5%, log loss=0.6611
"""

import pandas as pd
import numpy as np
from scipy.stats import zscore
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import log_loss

# ── 0. Load ───────────────────────────────────────────────────────────────────
df       = pd.read_excel("../datasci/whl_2025.xlsx")
matchups = pd.read_excel("../datasci/WHSDSC_Rnd1_matchups.xlsx")

# ── 1. Aggregate to game level ────────────────────────────────────────────────
game_level = df.groupby("game_id").agg(
    home_team  = ("home_team",  "first"),
    away_team  = ("away_team",  "first"),
    went_ot    = ("went_ot",    "first"),
    home_goals = ("home_goals", "sum"),
    away_goals = ("away_goals", "sum"),
    home_xg    = ("home_xg",    "sum"),
    away_xg    = ("away_xg",    "sum"),
).reset_index()

# ── 2. Stack into team-game rows ──────────────────────────────────────────────
home_df = pd.DataFrame({
    "team":          game_level["home_team"],
    "opponent":      game_level["away_team"],
    "goals_for":     game_level["home_goals"],
    "goals_against": game_level["away_goals"],
    "xg_for":        game_level["home_xg"],
    "xg_against":    game_level["away_xg"],
    "went_ot":       game_level["went_ot"],
    "home":          1,
})
away_df = pd.DataFrame({
    "team":          game_level["away_team"],
    "opponent":      game_level["home_team"],
    "goals_for":     game_level["away_goals"],
    "goals_against": game_level["home_goals"],
    "xg_for":        game_level["away_xg"],
    "xg_against":    game_level["home_xg"],
    "went_ot":       game_level["went_ot"],
    "home":          0,
})
tg = pd.concat([home_df, away_df], ignore_index=True)

# win: goals_for > goals_against (correct for both regulation and OT)
# ot_loss: lost AND game went to OT → earns 1 standings point
tg["win"]     = (tg["goals_for"] > tg["goals_against"]).astype(int)
tg["ot_loss"] = (
    (tg["goals_for"] < tg["goals_against"]) & (tg["went_ot"] == 1)
).astype(int)

# ── 3. Season stats per team ──────────────────────────────────────────────────
lf = tg.groupby("team").agg(
    gp        = ("win",          "count"),
    wins      = ("win",          "sum"),
    ot_losses = ("ot_loss",      "sum"),
    gf        = ("goals_for",    "sum"),
    ga        = ("goals_against","sum"),
    xgf       = ("xg_for",       "sum"),
    xga       = ("xg_against",   "sum"),
).reset_index()

# points = W×2 + OTL (WHL standing formula)
lf["points"]    = lf["wins"] * 2 + lf["ot_losses"]
lf["xgd_pg"]    = (lf["xgf"] - lf["xga"]) / lf["gp"]
lf["gsax_pg"]   = (lf["xga"] - lf["ga"])   / lf["gp"]
lf["finish_pg"] = (lf["gf"]  - lf["xgf"])  / lf["gp"]

# ── 4. Composite power score ──────────────────────────────────────────────────
for col in ["xgd_pg", "gsax_pg", "finish_pg", "points"]:
    lf[f"{col}_z"] = zscore(lf[col])

lf["power_score"] = (
    0.40 * lf["xgd_pg_z"]    +
    0.30 * lf["gsax_pg_z"]   +
    0.15 * lf["finish_pg_z"] +
    0.15 * lf["points_z"]
)

lf = lf.sort_values("power_score", ascending=False).reset_index(drop=True)
lf["rank"] = lf.index + 1

# ── 5. Print power rankings ───────────────────────────────────────────────────
print("=" * 70)
print("PHASE 1a — POWER RANKINGS")
print("=" * 70)
print(f"{'Rk':<4} {'Team':<15} {'Pts':<5} {'W':<4} {'OTL':<4} "
      f"{'xGD/gm':>7} {'GSAx/gm':>8} {'Fin/gm':>7} {'Score':>7}")
print("─" * 70)
for _, row in lf.iterrows():
    print(f"{int(row['rank']):<4} {row['team']:<15} {int(row['points']):<5} "
          f"{int(row['wins']):<4} {int(row['ot_losses']):<4} "
          f"{row['xgd_pg']:>7.3f} {row['gsax_pg']:>8.3f} "
          f"{row['finish_pg']:>7.3f} {row['power_score']:>7.3f}")

# ── 6. Logistic regression ────────────────────────────────────────────────────
fm = lf[["team", "xgd_pg_z", "gsax_pg_z", "finish_pg_z"]]

tg2 = tg.merge(fm, on="team")
opp = fm.rename(columns={
    "team":        "opponent",
    "xgd_pg_z":   "o_xg",
    "gsax_pg_z":  "o_gsax",
    "finish_pg_z":"o_fin",
})
tg2 = tg2.merge(opp, on="opponent")

tg2["dxg"]     = tg2["xgd_pg_z"]    - tg2["o_xg"]
tg2["dgsax"]   = tg2["gsax_pg_z"]   - tg2["o_gsax"]
tg2["dfinish"] = tg2["finish_pg_z"] - tg2["o_fin"]

X = tg2[["dxg", "dgsax", "dfinish", "home"]].values
y = tg2["win"].values

grid = GridSearchCV(
    LogisticRegression(max_iter=2000),
    {"C": [0.01, 0.1, 0.5, 1, 2, 5, 10]},
    scoring="neg_log_loss",
    cv=5,
    refit=True,
)
grid.fit(X, y)
best = grid.best_estimator_

cv_acc = cross_val_score(best, X, y, cv=5, scoring="accuracy").mean()
ll     = log_loss(y, best.predict_proba(X)[:, 1])

print("\n" + "=" * 70)
print("MODEL PERFORMANCE")
print("=" * 70)
print(f"  Features:           dxg, dgsax, dfinish, home_advantage")
print(f"  Best C:             {grid.best_params_['C']}")
print(f"  5-fold CV Accuracy: {cv_acc:.4f}")
print(f"  Log Loss:           {ll:.4f}")
print(f"  Coefficients:")
for name, coef in zip(["dxg", "dgsax", "dfinish", "home"], best.coef_[0]):
    print(f"    {name:<10} {coef:.4f}")

# ── 7. Tournament predictions ─────────────────────────────────────────────────
m = matchups.merge(fm, left_on="home_team", right_on="team").rename(columns={
    "xgd_pg_z":   "h_xg",
    "gsax_pg_z":  "h_gsax",
    "finish_pg_z":"h_fin",
}).drop("team", axis=1)

m = m.merge(fm, left_on="away_team", right_on="team").rename(columns={
    "xgd_pg_z":   "a_xg",
    "gsax_pg_z":  "a_gsax",
    "finish_pg_z":"a_fin",
}).drop("team", axis=1)

m["dxg"]     = m["h_xg"]   - m["a_xg"]
m["dgsax"]   = m["h_gsax"] - m["a_gsax"]
m["dfinish"] = m["h_fin"]  - m["a_fin"]
m["home"]    = 1

m["home_win_prob"] = best.predict_proba(
    m[["dxg", "dgsax", "dfinish", "home"]].values
)[:, 1]

print("\n" + "=" * 70)
print("PHASE 1a — ROUND 1 WIN PROBABILITIES (HOME TEAM)")
print("=" * 70)
print(f"{'Gm':<4} {'Home Team':<16} {'Away Team':<16} {'Home Win Prob':>13}")
print("─" * 52)
for _, row in m.iterrows():
    print(f"{int(row['game']):<4} {row['home_team']:<16} {row['away_team']:<16} "
          f"{row['home_win_prob']:>13.2f}")

# ── 8. Export ─────────────────────────────────────────────────────────────────
lf[["rank", "team", "points", "wins", "ot_losses",
    "xgd_pg", "gsax_pg", "finish_pg", "power_score"]].to_csv(
    "phase1a_power_rankings.csv", index=False)

m[["game", "home_team", "away_team", "home_win_prob"]].to_csv(
    "phase1a_win_probabilities.csv", index=False)

print("\nSaved: phase1a_power_rankings.csv")
print("Saved: phase1a_win_probabilities.csv")