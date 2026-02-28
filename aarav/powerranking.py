import pandas as pd
import numpy as np
from scipy.stats import zscore
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss

# =========================================================
# 1. LOAD DATA
# =========================================================

df = pd.read_excel("../datasci/whl_2025.xlsx")

# =========================================================
# 2. AGGREGATE LINE ROWS -> GAME LEVEL
# =========================================================

game_level = (
    df.groupby("game_id")
      .agg({
          "home_team": "first",
          "away_team": "first",
          "home_goals": "sum",
          "away_goals": "sum",
          "home_xg": "sum",
          "away_xg": "sum",
          "home_shots": "sum",
          "away_shots": "sum",
          "home_penalty_minutes": "sum",
          "away_penalty_minutes": "sum"
      })
      .reset_index()
)

# =========================================================
# 3. CREATE TEAM-GAME DATA
# =========================================================

home_df = pd.DataFrame({
    "team": game_level["home_team"],
    "opponent": game_level["away_team"],
    "goals_for": game_level["home_goals"],
    "goals_against": game_level["away_goals"],
    "xg_for": game_level["home_xg"],
    "xg_against": game_level["away_xg"],
    "shots_for": game_level["home_shots"],
    "shots_against": game_level["away_shots"],
    "home": 1
})

away_df = pd.DataFrame({
    "team": game_level["away_team"],
    "opponent": game_level["home_team"],
    "goals_for": game_level["away_goals"],
    "goals_against": game_level["home_goals"],
    "xg_for": game_level["away_xg"],
    "xg_against": game_level["home_xg"],
    "shots_for": game_level["away_shots"],
    "shots_against": game_level["home_shots"],
    "home": 0
})

team_game = pd.concat([home_df, away_df], ignore_index=True)
team_game["win"] = (team_game["goals_for"] > team_game["goals_against"]).astype(int)

# =========================================================
# 4. BUILD SEASON FEATURES
# =========================================================

league_features = (
    team_game.groupby("team")
        .agg(
            games_played=("win", "count"),
            wins=("win", "sum"),
            goals_for=("goals_for", "sum"),
            goals_against=("goals_against", "sum"),
            xg_for=("xg_for", "sum"),
            xg_against=("xg_against", "sum"),
            shots_against=("shots_against", "sum")
        )
        .reset_index()
)

league_features["xg_diff_per_game"] = (
    (league_features["xg_for"] - league_features["xg_against"]) /
    league_features["games_played"]
)

league_features["finishing_skill_per_game"] = (
    (league_features["goals_for"] - league_features["xg_for"]) /
    league_features["games_played"]
)

league_features["gsax_per_game"] = (
    (league_features["xg_against"] - league_features["goals_against"]) /
    league_features["games_played"]
)

league_features["suppression_ratio"] = (
    league_features["xg_against"] /
    league_features["shots_against"]
)

# Standardize season features
league_features["xg_z"] = zscore(league_features["xg_diff_per_game"])
league_features["gsax_z"] = zscore(league_features["gsax_per_game"])
league_features["supp_z"] = zscore(-league_features["suppression_ratio"])
league_features["finish_z"] = zscore(league_features["finishing_skill_per_game"])

# =========================================================
# 5. MERGE FEATURES BACK TO GAME LEVEL
# =========================================================

feature_map = league_features[[
    "team", "xg_z", "gsax_z", "supp_z", "finish_z"
]]

team_game = team_game.merge(feature_map, on="team")

opponent_features = feature_map.rename(columns={
    "team": "opponent",
    "xg_z": "opp_xg_z",
    "gsax_z": "opp_gsax_z",
    "supp_z": "opp_supp_z",
    "finish_z": "opp_finish_z"
})

team_game = team_game.merge(opponent_features, on="opponent")

# =========================================================
# 6. CREATE DIFFERENCE FEATURES
# =========================================================

team_game["dxg"] = team_game["xg_z"] - team_game["opp_xg_z"]
team_game["dgsax"] = team_game["gsax_z"] - team_game["opp_gsax_z"]
team_game["dsupp"] = team_game["supp_z"] - team_game["opp_supp_z"]
team_game["dfinish"] = team_game["finish_z"] - team_game["opp_finish_z"]

# =========================================================
# 7. FIT MULTIVARIABLE LOGISTIC MODEL
# =========================================================

X = team_game[["dxg", "dgsax", "dsupp", "dfinish", "home"]]
y = team_game["win"]

model = LogisticRegression(max_iter=1000)
model.fit(X, y)

probs = model.predict_proba(X)[:,1]
preds = model.predict(X)

print("\nMultivariable Logistic Results:")
print("Accuracy:", round(accuracy_score(y, preds),4))
print("Log Loss:", round(log_loss(y, probs),4))

print("\nModel Coefficients:")
for name, coef in zip(X.columns, model.coef_[0]):
    print(f"{name}: {round(coef,4)}")