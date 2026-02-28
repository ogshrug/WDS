import pandas as pd
import numpy as np

df = pd.read_excel('./whl_2025.xlsx')

# ── Home aggregations ────────────────────────────────────────────────────────
home = df.groupby('home_team').agg(
    home_games=('game_id', 'count'),
    home_wins=('home_goals', lambda x: ((x > df.loc[x.index, 'away_goals']) & ~df.loc[x.index, 'went_ot']).sum()),
    home_ot_wins=('home_goals', lambda x: ((x > df.loc[x.index, 'away_goals']) & df.loc[x.index, 'went_ot']).sum()),
    home_losses=('home_goals', lambda x: (x < df.loc[x.index, 'away_goals']).sum()),
    home_goals=('home_goals', 'sum'),
    home_goals_against=('away_goals', 'sum'),
    home_shots=('home_shots', 'sum'),
    home_shots_against=('away_shots', 'sum'),
    home_xg=('home_xg', 'sum'),
    home_xg_against=('away_xg', 'sum'),
    home_penalties=('home_penalties_committed', 'sum'),
    home_penalties_against=('away_penalties_committed', 'sum'),
).rename_axis('team')

# ── Away aggregations ────────────────────────────────────────────────────────
away = df.groupby('away_team').agg(
    away_games=('game_id', 'count'),
    away_wins=('away_goals', lambda x: ((x > df.loc[x.index, 'home_goals']) & ~df.loc[x.index, 'went_ot']).sum()),
    away_ot_wins=('away_goals', lambda x: ((x > df.loc[x.index, 'home_goals']) & df.loc[x.index, 'went_ot']).sum()),
    away_losses=('away_goals', lambda x: (x < df.loc[x.index, 'home_goals']).sum()),
    away_goals=('away_goals', 'sum'),
    away_goals_against=('home_goals', 'sum'),
    away_shots=('away_shots', 'sum'),
    away_shots_against=('home_shots', 'sum'),
    away_xg=('away_xg', 'sum'),
    away_xg_against=('home_xg', 'sum'),
    away_penalties=('away_penalties_committed', 'sum'),
    away_penalties_against=('home_penalties_committed', 'sum'),
).rename_axis('team')

t = home.join(away, how='outer').fillna(0)

# ── Totals ───────────────────────────────────────────────────────────────────
t['total_games']             = t['home_games'] + t['away_games']
t['total_wins']              = t['home_wins'] + t['away_wins']
t['total_ot_wins']           = t['home_ot_wins'] + t['away_ot_wins']
t['total_losses']            = t['home_losses'] + t['away_losses']
t['total_goals']             = t['home_goals'] + t['away_goals']
t['total_goals_against']     = t['home_goals_against'] + t['away_goals_against']
t['total_shots']             = t['home_shots'] + t['away_shots']
t['total_shots_against']     = t['home_shots_against'] + t['away_shots_against']
t['total_xg']                = t['home_xg'] + t['away_xg']
t['total_xg_against']        = t['home_xg_against'] + t['away_xg_against']
t['total_penalties']         = t['home_penalties'] + t['away_penalties']
t['total_penalties_against'] = t['home_penalties_against'] + t['away_penalties_against']

# ── Disparity: raw ───────────────────────────────────────────────────────────
t['goal_disparity']      = t['total_goals']       - t['total_goals_against']
t['shot_disparity']      = t['total_shots']        - t['total_shots_against']
t['xg_disparity']        = t['total_xg']           - t['total_xg_against']
t['finishing_disparity'] = t['total_goals']        - t['total_xg']
t['penalty_disparity']   = t['total_penalties']    - t['total_penalties_against']

# ── Disparity: per game (normalised) ────────────────────────────────────────
g = t['total_games']
t['goal_disparity_pg']      = t['goal_disparity']      / g
t['shot_disparity_pg']      = t['shot_disparity']       / g
t['xg_disparity_pg']        = t['xg_disparity']         / g
t['finishing_disparity_pg'] = t['finishing_disparity']  / g
t['penalty_disparity_pg']   = t['penalty_disparity']    / g

# ── Shot quality disparity (xG per shot) ────────────────────────────────────
t['shot_quality_for']       = t['total_xg']         / t['total_shots'].replace(0, np.nan)
t['shot_quality_against']   = t['total_xg_against'] / t['total_shots_against'].replace(0, np.nan)
t['shot_quality_disparity'] = t['shot_quality_for'] - t['shot_quality_against']

# ── Conversion disparity (goals per shot) ───────────────────────────────────
t['conversion_for']       = t['total_goals']         / t['total_shots'].replace(0, np.nan)
t['conversion_against']   = t['total_goals_against'] / t['total_shots_against'].replace(0, np.nan)
t['conversion_disparity'] = t['conversion_for'] - t['conversion_against']

# ── Home vs Away split disparity ─────────────────────────────────────────────
t['home_goal_disparity'] = t['home_goals'] - t['home_goals_against']
t['away_goal_disparity'] = t['away_goals'] - t['away_goals_against']
t['home_away_split']     = t['home_goal_disparity'] - t['away_goal_disparity']

# ── Win/loss context ─────────────────────────────────────────────────────────
t['win_pct']   = (t['total_wins'] + 0.5 * t['total_ot_wins']) / t['total_games']
t['point_pct'] = (t['total_wins'] * 2 + t['total_ot_wins'])   / (t['total_games'] * 2)

# ── Composite disparity score (equal-weighted z-scores) ─────────────────────
disparity_cols = [
    'goal_disparity_pg', 'shot_disparity_pg', 'xg_disparity_pg',
    'shot_quality_disparity', 'conversion_disparity'
]
z = t[disparity_cols]
t['composite_disparity_score'] = ((z - z.mean()) / z.std()).mean(axis=1)

# ── Tier assignment ──────────────────────────────────────────────────────────
def assign_tier(score):
    if score >= 0.75:    return '1 - Elite'
    elif score >= 0.25:  return '2 - Above Average'
    elif score >= -0.25: return '3 - Average'
    elif score >= -0.75: return '4 - Below Average'
    else:                return '5 - Poor'

t['tier']           = t['composite_disparity_score'].apply(assign_tier)
t['disparity_rank'] = t['composite_disparity_score'].rank(ascending=False).astype(int)

# ── Round all floats ─────────────────────────────────────────────────────────
t = t.round(3)

# ── Select final columns ─────────────────────────────────────────────────────
final_cols = [
    'total_games', 'total_wins', 'total_ot_wins', 'total_losses', 'win_pct', 'point_pct',
    'total_goals', 'total_goals_against', 'goal_disparity', 'goal_disparity_pg',
    'total_shots', 'total_shots_against', 'shot_disparity', 'shot_disparity_pg',
    'total_xg', 'total_xg_against', 'xg_disparity', 'xg_disparity_pg',
    'shot_quality_for', 'shot_quality_against', 'shot_quality_disparity',
    'conversion_for', 'conversion_against', 'conversion_disparity',
    'finishing_disparity', 'finishing_disparity_pg',
    'total_penalties', 'total_penalties_against', 'penalty_disparity', 'penalty_disparity_pg',
    'home_goal_disparity', 'away_goal_disparity', 'home_away_split',
    'composite_disparity_score', 'disparity_rank', 'tier',
]

summary_cols = [
    'total_games', 'win_pct', 'point_pct',
    'goal_disparity_pg', 'xg_disparity_pg', 'shot_quality_disparity',
    'conversion_disparity', 'finishing_disparity_pg', 'penalty_disparity_pg',
    'home_away_split', 'composite_disparity_score', 'disparity_rank', 'tier',
]

detailed = t[final_cols].sort_values('disparity_rank')
detailed.index.name = 'team'
summary = detailed[summary_cols].copy()

# ── League average row ───────────────────────────────────────────────────────
def add_league_avg(frame):
    avg = frame.select_dtypes(include='number').mean().round(3)
    avg['tier'] = 'LEAGUE AVG'
    avg.name = 'LEAGUE_AVERAGE'
    return pd.concat([frame, avg.to_frame().T])

detailed = add_league_avg(detailed)
summary  = add_league_avg(summary)

# ── Export ───────────────────────────────────────────────────────────────────
detailed.to_csv('../data/whl_2025_disparity_detailed.csv')
summary.to_csv('../data/whl_2025_disparity_summary.csv')

print(f"Exported whl_2025_disparity_detailed.csv  ({len(detailed)-1} teams, {len(final_cols)} columns)")
print(f"Exported whl_2025_disparity_summary.csv   ({len(summary)-1} teams, {len(summary_cols)} columns)")
print()
print(summary.drop('LEAGUE_AVERAGE').sort_values('disparity_rank')[
    ['disparity_rank', 'tier', 'win_pct', 'composite_disparity_score']
].to_string())