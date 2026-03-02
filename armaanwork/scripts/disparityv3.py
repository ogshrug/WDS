"""
WHL 2025 — Combined Line Disparity + Team Strength
====================================================
Outputs a single CSV (whl_2025_line_disparityv3.csv) containing:
  - Offensive line quality disparity metrics (first vs second line, adj. xG/60)
  - Overall team strength composite score (z-score across 5 team-level metrics)
  - Tier assignment
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

df = pd.read_excel('../whl_2025.xlsx')

# ═══════════════════════════════════════════════════════════════════════════════
# PART 1 — OVERALL TEAM STRENGTH (composite_disparity_score)
# ═══════════════════════════════════════════════════════════════════════════════

home_t = df.groupby('home_team').agg(
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

away_t = df.groupby('away_team').agg(
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

t = home_t.join(away_t, how='outer').fillna(0)

g = t['home_games'] + t['away_games']
total_goals         = t['home_goals']         + t['away_goals']
total_goals_against = t['home_goals_against'] + t['away_goals_against']
total_shots         = t['home_shots']         + t['away_shots']
total_shots_against = t['home_shots_against'] + t['away_shots_against']
total_xg            = t['home_xg']            + t['away_xg']
total_xg_against    = t['home_xg_against']    + t['away_xg_against']

goal_disparity_pg      = (total_goals - total_goals_against) / g
shot_disparity_pg      = (total_shots - total_shots_against) / g
xg_disparity_pg        = (total_xg   - total_xg_against)    / g
shot_quality_disparity = (total_xg / total_shots.replace(0, np.nan)) - \
                         (total_xg_against / total_shots_against.replace(0, np.nan))
conversion_disparity   = (total_goals / total_shots.replace(0, np.nan)) - \
                         (total_goals_against / total_shots_against.replace(0, np.nan))

z = pd.DataFrame({
    'goal_disparity_pg':      goal_disparity_pg,
    'shot_disparity_pg':      shot_disparity_pg,
    'xg_disparity_pg':        xg_disparity_pg,
    'shot_quality_disparity': shot_quality_disparity,
    'conversion_disparity':   conversion_disparity,
})
composite = ((z - z.mean()) / z.std()).mean(axis=1)
composite.name = 'composite_disparity_score'

def assign_tier(score):
    if score >= 0.75:    return '1 - Elite'
    elif score >= 0.25:  return '2 - Above Average'
    elif score >= -0.25: return '3 - Average'
    elif score >= -0.75: return '4 - Below Average'
    else:                return '5 - Poor'

tier = composite.apply(assign_tier)
tier.name = 'tier'

# ═══════════════════════════════════════════════════════════════════════════════
# PART 2 — OFFENSIVE LINE QUALITY DISPARITY (disparity_ratio)
# ═══════════════════════════════════════════════════════════════════════════════

ES_LINES = {'first_off', 'second_off'}
ES_DEF   = {'first_def', 'second_def'}

home_s = df[df['home_off_line'].isin(ES_LINES) & df['away_def_pairing'].isin(ES_DEF)].copy()
home_s = home_s.rename(columns={
    'home_team': 'team', 'home_off_line': 'off_line',
    'away_def_pairing': 'opp_def_pairing', 'home_xg': 'xg_for',
})

away_s = df[df['away_off_line'].isin(ES_LINES) & df['home_def_pairing'].isin(ES_DEF)].copy()
away_s = away_s.rename(columns={
    'away_team': 'team', 'away_off_line': 'off_line',
    'home_def_pairing': 'opp_def_pairing', 'away_xg': 'xg_for',
})

shifts = pd.concat([
    home_s[['team', 'off_line', 'opp_def_pairing', 'toi', 'xg_for']],
    away_s[['team', 'off_line', 'opp_def_pairing', 'toi', 'xg_for']],
], ignore_index=True)

line_raw = shifts.groupby(['team', 'off_line']).agg(
    total_xg=('xg_for', 'sum'),
    total_toi=('toi', 'sum'),
).reset_index()
line_raw['xg_per60_raw'] = (line_raw['total_xg'] / line_raw['total_toi']) * 3600

def_quality = shifts.groupby('opp_def_pairing').agg(
    def_xg_conceded=('xg_for', 'sum'),
    def_toi=('toi', 'sum'),
).reset_index()
def_quality['def_xg_per60'] = (def_quality['def_xg_conceded'] / def_quality['def_toi']) * 3600
league_avg_def = def_quality['def_xg_per60'].mean()

def weighted_def_quality(group):
    total_toi = group['toi'].sum()
    if total_toi == 0:
        return np.nan
    m = group.merge(def_quality[['opp_def_pairing', 'def_xg_per60']], on='opp_def_pairing', how='left')
    return (m['def_xg_per60'] * m['toi']).sum() / total_toi

matchup_quality = (
    shifts.groupby(['team', 'off_line'])
    .apply(weighted_def_quality, include_groups=False)
    .reset_index(name='avg_def_faced_xg60')
)

line_adj = line_raw.merge(matchup_quality, on=['team', 'off_line'])
line_adj['xg_per60_adj'] = line_adj['xg_per60_raw'] * (league_avg_def / line_adj['avg_def_faced_xg60'])

pivot = line_adj.pivot(index='team', columns='off_line', values='xg_per60_adj').reset_index()
pivot.columns.name = None
pivot = pivot.rename(columns={'first_off': 'first_line_adj_xg60', 'second_off': 'second_line_adj_xg60'})
pivot['disparity_ratio'] = pivot['first_line_adj_xg60'] / pivot['second_line_adj_xg60']
pivot['disparity_diff']  = pivot['first_line_adj_xg60'] - pivot['second_line_adj_xg60']

raw_pivot = line_raw.pivot(index='team', columns='off_line', values='xg_per60_raw').reset_index()
raw_pivot.columns.name = None
raw_pivot = raw_pivot.rename(columns={'first_off': 'first_line_raw_xg60', 'second_off': 'second_line_raw_xg60'})

toi_pivot = line_raw.pivot(index='team', columns='off_line', values='total_toi').reset_index()
toi_pivot.columns.name = None
toi_pivot = toi_pivot.rename(columns={'first_off': 'first_line_toi_s', 'second_off': 'second_line_toi_s'})

pivot = pivot.merge(raw_pivot, on='team').merge(toi_pivot, on='team')

# ═══════════════════════════════════════════════════════════════════════════════
# PART 3 — MERGE & EXPORT
# ═══════════════════════════════════════════════════════════════════════════════

pivot = pivot.merge(
    pd.DataFrame({'team': composite.index, 'composite_disparity_score': composite.values, 'tier': tier.values}),
    on='team'
)

pivot = pivot.sort_values('disparity_ratio', ascending=False).reset_index(drop=True)
pivot['rank'] = pivot.index + 1

float_cols = pivot.select_dtypes(include='float').columns
pivot[float_cols] = pivot[float_cols].round(4)

out_cols = [
    'rank', 'team',
    'first_line_adj_xg60', 'second_line_adj_xg60',
    'disparity_ratio', 'disparity_diff',
    'first_line_raw_xg60', 'second_line_raw_xg60',
    'first_line_toi_s', 'second_line_toi_s',
    'composite_disparity_score', 'tier',
]

pivot[out_cols].to_csv('../data/whl_2025_line_disparityv3.csv', index=False)
print(f"Saved whl_2025_line_disparityv3.csv  ({len(pivot)} teams, {len(out_cols)} columns)")
print(pivot[['rank', 'team', 'disparity_ratio', 'composite_disparity_score', 'tier']].to_string(index=False))