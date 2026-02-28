import pandas as pd
import numpy as np
from scipy.stats import zscore
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, log_loss

# =========================================================
# 1. LOAD DATA
# =========================================================

df = pd.read_excel("../datasci/whl_2025.xlsx")

# =========================================================
# 2. AGGREGATE TO GAME LEVEL
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
    "penalty_minutes": game_level["home_penalty_minutes"],
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
    "penalty_minutes": game_level["away_penalty_minutes"],
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
            shots_against=("shots_against", "sum"),
            penalties_taken=("penalty_minutes", "sum")
        )
        .reset_index()
)

league_features["win_pct"] = league_features["wins"] / league_features["games_played"]

# Core strength metrics
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

# =========================================================
# 5. SPECIAL TEAMS PROXY (NET PENALTY PER GAME)
# =========================================================

# Penalties drawn = opponent penalties
opponent_penalties = (
    team_game.groupby("opponent")
        .agg(penalties_drawn=("penalty_minutes", "sum"))
        .reset_index()
        .rename(columns={"opponent": "team"})
)

league_features = league_features.merge(opponent_penalties, on="team")

league_features["net_penalty_per_game"] = (
    (league_features["penalties_drawn"] -
     league_features["penalties_taken"]) /
    league_features["games_played"]
)

# =========================================================
# 6. STANDARDIZE FEATURES
# =========================================================

league_features["xg_z"] = zscore(league_features["xg_diff_per_game"])
league_features["gsax_z"] = zscore(league_features["gsax_per_game"])
league_features["finish_z"] = zscore(league_features["finishing_skill_per_game"])
league_features["st_z"] = zscore(league_features["net_penalty_per_game"])

# =========================================================
# 7. MERGE FEATURES BACK TO GAME LEVEL
# =========================================================

feature_map = league_features[[
    "team", "xg_z", "gsax_z", "finish_z", "st_z"
]]

team_game = team_game.merge(feature_map, on="team")

opponent_features = feature_map.rename(columns={
    "team": "opponent",
    "xg_z": "opp_xg_z",
    "gsax_z": "opp_gsax_z",
    "finish_z": "opp_finish_z",
    "st_z": "opp_st_z"
})

team_game = team_game.merge(opponent_features, on="opponent")

# =========================================================
# 8. CREATE DIFFERENCE FEATURES
# =========================================================

team_game["dxg"] = team_game["xg_z"] - team_game["opp_xg_z"]
team_game["dgsax"] = team_game["gsax_z"] - team_game["opp_gsax_z"]
team_game["dfinish"] = team_game["finish_z"] - team_game["opp_finish_z"]
team_game["dst"] = team_game["st_z"] - team_game["opp_st_z"]

# =========================================================
# 9. LOGISTIC REGRESSION WITH CV TUNING
# =========================================================

X = team_game[["dxg", "dgsax", "dfinish", "dst", "home"]]
y = team_game["win"]

logreg = LogisticRegression(max_iter=2000)

param_grid = {"C": [0.01, 0.1, 0.5, 1, 2, 5, 10]}

grid = GridSearchCV(
    logreg,
    param_grid,
    scoring="neg_log_loss",
    cv=5
)

grid.fit(X, y)

best_model = grid.best_estimator_

probs = best_model.predict_proba(X)[:,1]
preds = best_model.predict(X)

print("\n=== Logistic with Special Teams Feature ===")
print("Best C:", grid.best_params_)
print("Accuracy:", round(accuracy_score(y, preds),4))
print("Log Loss:", round(log_loss(y, probs),4))

print("\nCoefficients:")
for name, coef in zip(X.columns, best_model.coef_[0]):
    print(f"{name}: {round(coef,4)}")
# FINAL BEST MODEL (SIMPLIFIED, NO DST)

X = team_game[["dxg", "dgsax", "dfinish", "home"]]
y = team_game["win"]

logreg = LogisticRegression(max_iter=2000)

param_grid = {"C": [0.01, 0.1, 0.5, 1, 2, 5, 10]}

grid = GridSearchCV(
    logreg,
    param_grid,
    scoring="neg_log_loss",
    cv=5
)

grid.fit(X, y)

best_model = grid.best_estimator_

print("Final Model Ready")
# =========================================================
# LOAD TOURNAMENT MATCHUPS
# =========================================================

matchups = pd.read_excel("../datasci/WHSDSC_Rnd1_matchups.xlsx")

# Map season z-scores
feature_map = league_features[[
    "team", "xg_z", "gsax_z", "finish_z"
]]

# Merge for home team
matchups = matchups.merge(
    feature_map,
    left_on="home_team",
    right_on="team",
    how="left"
).rename(columns={
    "xg_z": "home_xg",
    "gsax_z": "home_gsax",
    "finish_z": "home_finish"
}).drop(columns=["team"])

# Merge for away team
matchups = matchups.merge(
    feature_map,
    left_on="away_team",
    right_on="team",
    how="left"
).rename(columns={
    "xg_z": "away_xg",
    "gsax_z": "away_gsax",
    "finish_z": "away_finish"
}).drop(columns=["team"])

# =========================================================
# BUILD DIFFERENCE FEATURES
# =========================================================

matchups["dxg"] = matchups["home_xg"] - matchups["away_xg"]
matchups["dgsax"] = matchups["home_gsax"] - matchups["away_gsax"]
matchups["dfinish"] = matchups["home_finish"] - matchups["away_finish"]

# Since file explicitly says home_team, give home advantage
matchups["home"] = 1

X_tournament = matchups[["dxg", "dgsax", "dfinish", "home"]]

# =========================================================
# PREDICT PROBABILITIES
# =========================================================

matchups["home_win_probability"] = best_model.predict_proba(X_tournament)[:,1]

print("\n=== Tournament Predictions ===")
print(matchups[["home_team", "away_team", "home_win_probability"]])