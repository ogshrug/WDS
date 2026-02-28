import pandas as pd

# ===============================
# 1. LOAD DATA
# ===============================

df = pd.read_excel("../datasci/whl_2025.xlsx")

# ===============================
# 2. AGGREGATE LINE ROWS -> GAME LEVEL
# ===============================

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

# ===============================
# 3. CREATE HOME TEAM TABLE
# ===============================

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

# ===============================
# 4. CREATE AWAY TEAM TABLE
# ===============================

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

# ===============================
# 5. COMBINE INTO TEAM-GAME DATA
# ===============================

team_game = pd.concat([home_df, away_df], ignore_index=True)

# ===============================
# 6. ADD GAME OUTCOME VARIABLES
# ===============================

team_game["win"] = (team_game["goals_for"] > team_game["goals_against"]).astype(int)
team_game["loss"] = (team_game["goals_for"] < team_game["goals_against"]).astype(int)
team_game["goal_diff"] = team_game["goals_for"] - team_game["goals_against"]
team_game["xg_diff"] = team_game["xg_for"] - team_game["xg_against"]

# ===============================
# 7. BUILD LEAGUE TABLE
# ===============================

league_table = (
    team_game.groupby("team")
        .agg(
            games_played=("win", "count"),
            wins=("win", "sum"),
            losses=("loss", "sum"),
            goals_for=("goals_for", "sum"),
            goals_against=("goals_against", "sum"),
            goal_diff=("goal_diff", "sum"),
            xg_for=("xg_for", "sum"),
            xg_against=("xg_against", "sum"),
            xg_diff=("xg_diff", "sum")
        )
        .reset_index()
)

# ===============================
# 8. ADD PER-GAME METRICS
# ===============================

league_table["win_pct"] = league_table["wins"] / league_table["games_played"]
league_table["xg_for_per_game"] = league_table["xg_for"] / league_table["games_played"]
league_table["xg_against_per_game"] = league_table["xg_against"] / league_table["games_played"]
league_table["xg_diff_per_game"] = league_table["xg_diff"] / league_table["games_played"]
league_table["goal_diff_per_game"] = league_table["goal_diff"] / league_table["games_played"]

# Base strength metric
league_table["team_strength"] = league_table["xg_diff_per_game"]

# ===============================
# 9. SORT STANDINGS
# ===============================

league_table = league_table.sort_values(
    by=["wins", "goal_diff"],
    ascending=False
).reset_index(drop=True)

# ===============================
# 10. SANITY CHECK
# ===============================

print("Unique games played per team:")
print(league_table["games_played"].unique())

print("\nTop 10 Teams:")
print(league_table.head(10))
team_game.to_csv("team_game.csv", index=False)