import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, log_loss

# =========================================================
# 1. LOAD DATA
# =========================================================

df = pd.read_excel("../datasci/whl_2025.xlsx")

# Ensure chronological order
df = df.sort_values("game_id")

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
      })
      .reset_index()
)

# =========================================================
# 3. BUILD TIME-WEIGHTED TEAM STRENGTH
# =========================================================

alpha = 0.97

teams = pd.concat([game_level["home_team"],
                   game_level["away_team"]]).unique()

ratings = {
    team: {
        "xg": 0,
        "gsax": 0,
        "finish": 0,
        "weight_sum": 0
    }
    for team in teams
}

# Track game count for decay
games_played = {team: 0 for team in teams}

records = []

for _, row in game_level.iterrows():

    home = row["home_team"]
    away = row["away_team"]

    # Current weighted strengths BEFORE this game
    def get_strength(team):
        if ratings[team]["weight_sum"] == 0:
            return (0,0,0)
        return (
            ratings[team]["xg"] / ratings[team]["weight_sum"],
            ratings[team]["gsax"] / ratings[team]["weight_sum"],
            ratings[team]["finish"] / ratings[team]["weight_sum"]
        )

    home_xg_s, home_gsax_s, home_fin_s = get_strength(home)
    away_xg_s, away_gsax_s, away_fin_s = get_strength(away)

    # Store for modeling
    records.append({
        "home": 1,
        "dxg": home_xg_s - away_xg_s,
        "dgsax": home_gsax_s - away_gsax_s,
        "dfinish": home_fin_s - away_fin_s,
        "win": int(row["home_goals"] > row["away_goals"])
    })

    # Now update strengths AFTER game
    for team, xg_for, xg_against, goals_for, goals_against in [
        (home, row["home_xg"], row["away_xg"],
         row["home_goals"], row["away_goals"]),
        (away, row["away_xg"], row["home_xg"],
         row["away_goals"], row["home_goals"])
    ]:

        weight = alpha ** games_played[team]

        ratings[team]["xg"] += weight * (xg_for - xg_against)
        ratings[team]["gsax"] += weight * (xg_against - goals_against)
        ratings[team]["finish"] += weight * (goals_for - xg_for)
        ratings[team]["weight_sum"] += weight

        games_played[team] += 1

# =========================================================
# 4. BUILD MODEL DATAFRAME
# =========================================================

model_df = pd.DataFrame(records)

X = model_df[["dxg", "dgsax", "dfinish", "home"]]
y = model_df["win"]

# =========================================================
# 5. REGULARIZED LOGISTIC WITH CV
# =========================================================

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

print("\n=== Time-Weighted Logistic Model ===")
print("Best C:", grid.best_params_)
print("Accuracy:", round(accuracy_score(y, preds),4))
print("Log Loss:", round(log_loss(y, probs),4))

print("\nCoefficients:")
for name, coef in zip(X.columns, best_model.coef_[0]):
    print(f"{name}: {round(coef,4)}")