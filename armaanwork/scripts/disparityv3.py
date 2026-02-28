"""
Offensive Line Quality Disparity
=================================
For each team, quantifies how much better their first offensive line is
compared to their secondary line — i.e. how top-heavy their offence is.

Methodology
-----------
1. Filter to even-strength only (first_off / second_off lines; exclude
   PP_up, PP_kill_dwn, empty_net_line which distort xG rates).

2. Build an xG rate per 60 minutes for each line per team:
       xG_per60 = (total_xg_generated / total_toi_seconds) * 3600

3. Adjust for defensive matchup quality faced:
   - For every shift, the opposing defensive pairing (first_def vs second_def)
     represents a harder or easier matchup.
   - We compute each def pairing's league-wide xG-against rate (how many xG
     they typically concede per 60).
   - A line facing tougher defence (lower xG-against rate) gets a positive
     adjustment; facing weaker defence gets a negative one.
   - Adjusted xG_per60 = raw_xG_per60 * (league_avg_def_rate / matchup_def_rate)

4. Compute the offensive line quality disparity ratio:
       disparity_ratio = first_line_adj_xg60 / second_line_adj_xg60
   A ratio > 1 means the first line outperforms the second.
   The further above 1, the more top-heavy the team's offence is.

5. Rank teams from largest to smallest disparity ratio.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_excel('../whl_2025.xlsx')

# ── Step 1: Filter to even-strength only ─────────────────────────────────────
ES_LINES = {'first_off', 'second_off'}
ES_DEF   = {'first_def', 'second_def'}

# Build a long-form table: one row = one (team, off_line) shift
# We process home and away separately then stack.

home = df[df['home_off_line'].isin(ES_LINES) & df['away_def_pairing'].isin(ES_DEF)].copy()
home = home.rename(columns={
    'home_team':        'team',
    'away_team':        'opponent',
    'home_off_line':    'off_line',
    'away_def_pairing': 'opp_def_pairing',
    'home_xg':          'xg_for',
    'away_xg':          'xg_against',
})
home['side'] = 'home'

away = df[df['away_off_line'].isin(ES_LINES) & df['home_def_pairing'].isin(ES_DEF)].copy()
away = away.rename(columns={
    'away_team':        'team',
    'home_team':        'opponent',
    'away_off_line':    'off_line',
    'home_def_pairing': 'opp_def_pairing',
    'away_xg':          'xg_for',
    'home_xg':          'xg_against',
})
away['side'] = 'away'

shifts = pd.concat([
    home[['team', 'opponent', 'off_line', 'opp_def_pairing', 'toi', 'xg_for', 'xg_against', 'side']],
    away[['team', 'opponent', 'off_line', 'opp_def_pairing', 'toi', 'xg_for', 'xg_against', 'side']],
], ignore_index=True)

# ── Step 2: Raw xG per 60 per team × line ────────────────────────────────────
line_raw = shifts.groupby(['team', 'off_line']).agg(
    total_xg=('xg_for', 'sum'),
    total_toi=('toi',    'sum'),
    shifts=('toi',       'count'),
).reset_index()

line_raw['xg_per60_raw'] = (line_raw['total_xg'] / line_raw['total_toi']) * 3600

# ── Step 3: Defensive matchup adjustment ─────────────────────────────────────
# For each def pairing, compute how many xG it concedes per 60 (league-wide).
# This is the xg_for from the offensive team's perspective, grouped by the
# defensive pairing they faced.
def_quality = shifts.groupby('opp_def_pairing').agg(
    def_xg_conceded=('xg_for', 'sum'),
    def_toi=('toi', 'sum'),
).reset_index()
def_quality['def_xg_per60'] = (def_quality['def_xg_conceded'] / def_quality['def_toi']) * 3600

league_avg_def = def_quality['def_xg_per60'].mean()

# Per team × line: weighted average def quality faced
# (weight by TOI so short shifts don't skew)
def weighted_def_quality(group):
    total_toi = group['toi'].sum()
    if total_toi == 0:
        return np.nan
    merged = group.merge(def_quality[['opp_def_pairing', 'def_xg_per60']],
                         on='opp_def_pairing', how='left')
    return (merged['def_xg_per60'] * merged['toi']).sum() / total_toi

matchup_quality = (
    shifts.groupby(['team', 'off_line'])
    .apply(weighted_def_quality, include_groups=False)
    .reset_index(name='avg_def_faced_xg60')
)

line_adj = line_raw.merge(matchup_quality, on=['team', 'off_line'])

# Adjustment: scale up if faced tougher-than-average defence, scale down if easier
# adj_xg60 = raw_xg60 * (league_avg / matchup_quality)
# Higher matchup quality (easier defence) → ratio < 1 → penalises
# Lower matchup quality (harder defence)  → ratio > 1 → rewards
line_adj['adj_factor']   = league_avg_def / line_adj['avg_def_faced_xg60']
line_adj['xg_per60_adj'] = line_adj['xg_per60_raw'] * line_adj['adj_factor']

# ── Step 4: Pivot to first vs second and compute ratio ───────────────────────
pivot = line_adj.pivot(index='team', columns='off_line', values='xg_per60_adj').reset_index()
pivot.columns.name = None
pivot = pivot.rename(columns={'first_off': 'first_line_adj_xg60',
                               'second_off': 'second_line_adj_xg60'})

pivot['disparity_ratio'] = pivot['first_line_adj_xg60'] / pivot['second_line_adj_xg60']
pivot['disparity_diff']  = pivot['first_line_adj_xg60'] - pivot['second_line_adj_xg60']

# Also attach raw for reference
raw_pivot = line_raw.pivot(index='team', columns='off_line', values='xg_per60_raw').reset_index()
raw_pivot.columns.name = None
raw_pivot = raw_pivot.rename(columns={'first_off': 'first_line_raw_xg60',
                                       'second_off': 'second_line_raw_xg60'})
pivot = pivot.merge(raw_pivot, on='team')

# Attach TOI for context
toi_pivot = line_raw.pivot(index='team', columns='off_line', values='total_toi').reset_index()
toi_pivot.columns.name = None
toi_pivot = toi_pivot.rename(columns={'first_off': 'first_line_toi_s',
                                       'second_off': 'second_line_toi_s'})
pivot = pivot.merge(toi_pivot, on='team')

# ── Step 5: Rank ──────────────────────────────────────────────────────────────
pivot = pivot.sort_values('disparity_ratio', ascending=False).reset_index(drop=True)
pivot['rank'] = pivot.index + 1

# Round floats
float_cols = pivot.select_dtypes(include='float').columns
pivot[float_cols] = pivot[float_cols].round(4)

# ── Export CSV ────────────────────────────────────────────────────────────────
out_cols = [
    'rank', 'team',
    'first_line_adj_xg60', 'second_line_adj_xg60',
    'disparity_ratio', 'disparity_diff',
    'first_line_raw_xg60', 'second_line_raw_xg60',
    'first_line_toi_s', 'second_line_toi_s',
]
pivot[out_cols].to_csv('../data/whl_2025_line_disparityv3.csv', index=False)
print("Saved ../data/whl_2025_line_disparityv3.csv")
print(pivot[['rank', 'team', 'first_line_adj_xg60', 'second_line_adj_xg60',
             'disparity_ratio']].to_string(index=False))

# ── Plot ──────────────────────────────────────────────────────────────────────
BG       = '#0A0A0A'
BG_PANEL = '#111111'
GRID     = '#222222'
WHITE    = '#F0F0F0'
DIM      = '#777777'
CYAN     = '#00C2FF'
PURPLE   = '#7B61FF'
GOLD     = '#FFD700'
DIVIDER  = '#2A2A2A'

plt.rcParams.update({
    'figure.facecolor': BG,
    'axes.facecolor':   BG_PANEL,
    'axes.edgecolor':   GRID,
    'axes.labelcolor':  WHITE,
    'xtick.color':      DIM,
    'ytick.color':      WHITE,
    'text.color':       WHITE,
    'grid.color':       GRID,
    'grid.linewidth':   0.5,
    'font.family':      'monospace',
})

plot = pivot.sort_values('disparity_ratio', ascending=True)
teams  = [t.replace('_', ' ').title() for t in plot['team']]
ratio  = plot['disparity_ratio'].values
first  = plot['first_line_adj_xg60'].values
second = plot['second_line_adj_xg60'].values
ranks  = plot['rank'].values[::-1]  # reversed for bottom-to-top display

fig, axes = plt.subplots(1, 2, figsize=(18, 11), gridspec_kw={'width_ratios': [1.4, 1]})
fig.patch.set_facecolor(BG)
fig.subplots_adjust(wspace=0.05)

# ── Left: Disparity ratio bar chart ──────────────────────────────────────────
ax = axes[0]
ax.set_facecolor(BG_PANEL)

league_avg_ratio = ratio.mean()
bar_colors = [CYAN if r >= league_avg_ratio else PURPLE for r in ratio]
bars = ax.barh(teams, ratio, color=bar_colors, height=0.65, zorder=3)

ax.axvline(1.0, color='#444444', linewidth=1.0, zorder=4, linestyle='-')
ax.axvline(league_avg_ratio, color=GOLD, linewidth=0.9, linestyle='--', zorder=5, alpha=0.8)
ax.text(league_avg_ratio + 0.005, len(teams) - 0.5,
        f'avg {league_avg_ratio:.3f}', fontsize=7, color=GOLD, va='top')

for bar, val in zip(bars, ratio):
    ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
            f'{val:.3f}×', va='center', ha='left', fontsize=7, color=WHITE, alpha=0.9)

ax.set_xlabel('Disparity Ratio  (first line adj.xG/60  ÷  second line adj.xG/60)', fontsize=8, color=DIM, labelpad=6)
ax.tick_params(axis='y', labelsize=8)
ax.tick_params(axis='x', labelsize=8)
ax.xaxis.grid(True, zorder=0, alpha=0.35)
ax.set_axisbelow(True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color(GRID)
ax.spines['bottom'].set_color(GRID)

# ── Right: Grouped bars — first vs second line adj xG/60 ─────────────────────
ax2 = axes[1]
ax2.set_facecolor(BG_PANEL)

y = np.arange(len(teams))
h = 0.3
ax2.barh(y + h/2, first,  h, color=CYAN,   label='1st line adj.xG/60', zorder=3, alpha=0.9)
ax2.barh(y - h/2, second, h, color=PURPLE, label='2nd line adj.xG/60', zorder=3, alpha=0.9)

ax2.set_yticks(y)
ax2.set_yticklabels([])   # shared y-axis, labels already on left
ax2.tick_params(axis='x', labelsize=8)
ax2.xaxis.grid(True, zorder=0, alpha=0.35)
ax2.set_axisbelow(True)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['left'].set_color(DIVIDER)
ax2.spines['bottom'].set_color(GRID)
ax2.set_xlabel('Adjusted xG per 60 min', fontsize=8, color=DIM, labelpad=6)
ax2.legend(loc='lower right', fontsize=7.5, facecolor='#1A1A1A',
           edgecolor=GRID, labelcolor=WHITE, framealpha=0.9)

# ── Shared title ──────────────────────────────────────────────────────────────
fig.text(0.05, 0.975, 'OFFENSIVE LINE QUALITY DISPARITY',
         fontsize=15, fontweight='bold', color=WHITE, va='top')
fig.text(0.05, 0.945,
         'First line vs second line  |  Even-strength only  |  xG/60 adjusted for TOI and defensive matchup quality',
         fontsize=8.5, color=DIM, va='top')

plt.tight_layout(rect=[0, 0, 1, 0.935])
plt.savefig('../disparity_plots/whl_2025_line_disparity_v3.png', dpi=600, bbox_inches='tight', facecolor=BG)
plt.close()
print("Saved whl_2025_line_disparity3.png")