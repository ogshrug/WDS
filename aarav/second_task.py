import pandas as pd
import numpy as np

team_game = pd.read_csv("team_game.csv")

league_features = (
    team_game.groupby("team")
        .agg(
            games_played=("win", "count"),
            wins=("win", "sum"),
            goals_for=("goals_for", "sum"),
            goals_against=("goals_against", "sum"),
            xg_for=("xg_for", "sum"),
            xg_against=("xg_against", "sum"),
            shots_for=("shots_for", "sum"),
            shots_against=("shots_against", "sum"),
            penalty_minutes=("penalty_minutes", "sum")
        )
        .reset_index()
)

# ===============================
# 2. CORE METRICS
# ===============================

# Win %
league_features["win_pct"] = league_features["wins"] / league_features["games_played"]

# xG differential per game
league_features["xg_diff_per_game"] = (
    (league_features["xg_for"] - league_features["xg_against"]) /
    league_features["games_played"]
)

# Finishing skill (Goals - xG) per game
league_features["finishing_skill_per_game"] = (
    (league_features["goals_for"] - league_features["xg_for"]) /
    league_features["games_played"]
)

# Goals Saved Above Expected (GSAx) per game
league_features["gsax_per_game"] = (
    (league_features["xg_against"] - league_features["goals_against"]) /
    league_features["games_played"]
)

# Suppression ratio (defensive shot quality allowed)
league_features["suppression_ratio"] = (
    league_features["xg_against"] /
    league_features["shots_against"]
)

# Penalty minutes per game
league_features["pim_per_game"] = (
    league_features["penalty_minutes"] /
    league_features["games_played"]
)

# ===============================
# 3. CORRELATION ANALYSIS
# ===============================

feature_cols = [
    "xg_diff_per_game",
    "finishing_skill_per_game",
    "gsax_per_game",
    "suppression_ratio",
    "pim_per_game"
]

print("\nCorrelation with Win Percentage:\n")
for col in feature_cols:
    corr = league_features[col].corr(league_features["win_pct"])
    print(f"{col}: {corr:.4f}")

print("\nStandard Deviations:\n")
print(league_features[feature_cols].std())

print("\nFull Correlation Matrix:\n")
print(league_features[feature_cols + ["win_pct"]].corr())