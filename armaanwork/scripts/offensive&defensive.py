import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.utils import resample

# ==========================================================
# 1. LOAD DATA
# ==========================================================

file_path = "../whl_2025.xlsx"
df = pd.read_excel(file_path)
df.columns = [c.strip().lower() for c in df.columns]

# ==========================================================
# 2. BUILD LONG MATCHUP DATASET
# ==========================================================

home = df[[
    "home_team",
    "home_off_line",
    "away_def_pairing",
    "away_goalie",
    "home_xg",
    "toi"
]].copy()

home.columns = ["team", "off_line", "def_pairing", "goalie", "xg", "toi"]

away = df[[
    "away_team",
    "away_off_line",
    "home_def_pairing",
    "home_goalie",
    "away_xg",
    "toi"
]].copy()

away.columns = ["team", "off_line", "def_pairing", "goalie", "xg", "toi"]

data = pd.concat([home, away], ignore_index=True)

# Remove special teams
data = data[~data["off_line"].str.contains("PP", na=False)]

data["xg"] = pd.to_numeric(data["xg"], errors="coerce")
data["toi"] = pd.to_numeric(data["toi"], errors="coerce")

data = data.dropna(subset=["xg", "toi"])

# Compute rate
data["xg_per60"] = (data["xg"] / data["toi"]) * 3600

# ==========================================================
# 3. SHRINKAGE (EMPIRICAL BAYES)
# ==========================================================

league_mean = data["xg_per60"].mean()
k = 500  # shrinkage strength

line_toi = data.groupby(["team", "off_line"])["toi"].sum().reset_index()
line_toi.columns = ["team", "off_line", "total_toi"]

data = data.merge(line_toi, on=["team", "off_line"])

data["shrunk_xg_per60"] = (
    (data["xg_per60"] * data["total_toi"] + league_mean * k) /
    (data["total_toi"] + k)
)

# ==========================================================
# 4. REGRESSION MODEL (CONTEXT CONTROL)
# ==========================================================

model = smf.ols(
    formula="shrunk_xg_per60 ~ C(off_line) + C(def_pairing) + C(goalie)",
    data=data
).fit()

print(model.summary())

# Extract line coefficients
params = model.params

# ==========================================================
# 5. ESTIMATE TEAM-SPECIFIC LINE EFFECTS
# ==========================================================

effects = []

for team in data["team"].unique():

    team_data = data[data["team"] == team]

    # Predict mean adjusted production for each line
    preds = model.predict(team_data)

    team_data = team_data.copy()
    team_data["pred"] = preds

    grouped = team_data.groupby("off_line")["pred"].mean()

    if "first_off" in grouped and "second_off" in grouped:
        disparity = grouped["first_off"] - grouped["second_off"]

        effects.append({
            "team": team,
            "Line1_Effect": grouped["first_off"],
            "Line2_Effect": grouped["second_off"],
            "Disparity_Diff": disparity
        })

effects_df = pd.DataFrame(effects)
effects_df = effects_df.sort_values("Disparity_Diff", ascending=False)

print("\nMatchup + Goalie Adjusted Disparity Ranking:\n")
print(effects_df.head(10))

# ==========================================================
# 6. BOOTSTRAP CONFIDENCE INTERVALS
# ==========================================================

boot_results = []

for i in range(100):
    sample = resample(data)

    m = smf.ols(
        formula="shrunk_xg_per60 ~ C(off_line) + C(def_pairing) + C(goalie)",
        data=sample
    ).fit()

    for team in sample["team"].unique():
        t_data = sample[sample["team"] == team]
        preds = m.predict(t_data)
        t_data = t_data.copy()
        t_data["pred"] = preds

        grouped = t_data.groupby("off_line")["pred"].mean()

        if "first_off" in grouped and "second_off" in grouped:
            boot_results.append({
                "team": team,
                "disparity": grouped["first_off"] - grouped["second_off"]
            })

boot_df = pd.DataFrame(boot_results)

ci = (
    boot_df.groupby("team")["disparity"]
    .agg([
        lambda x: np.percentile(x, 2.5),
        lambda x: np.percentile(x, 97.5)
    ])
    .reset_index()
)

ci.columns = ["team", "CI_lower", "CI_upper"]

final = effects_df.merge(ci, on="team", how="left")

print("\nTop 10 with 95% CI:\n")
print(final.head(10))

final.to_csv("moon_level_disparity_model.csv", index=False)
print("\nSaved as moon_level_disparity_model.csv")