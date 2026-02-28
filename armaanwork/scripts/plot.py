"""
Cyan bars = positive disparity (favours team), red bars = negative
Dashed gold line = league average with its value labelled
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import numpy as np
from pathlib import Path

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv('../whl_2025_disparity_detailed.csv', index_col=0)
df = df[df.index != 'LEAGUE_AVERAGE'].copy()

# ── Theme ─────────────────────────────────────────────────────────────────────
BG        = '#0A0A0A'
BG_PANEL  = '#111111'
GRID      = '#222222'
WHITE     = '#F0F0F0'
DIM       = '#666666'
POS       = '#00C2FF'   # cyan  – above zero
NEG       = '#FF4C6A'   # red   – below zero
ZERO_LINE = '#444444'
ACCENT    = '#7B61FF'   # purple – neutral/composite

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

# ── Disparity metrics to plot ─────────────────────────────────────────────────
METRICS = [
    ('goal_disparity_pg',       'Goal Disparity',           'Goals scored minus goals conceded per game',                 POS),
    ('shot_disparity_pg',       'Shot Disparity',           'Shots attempted minus shots faced per game',                 POS),
    ('xg_disparity_pg',         'xG Disparity',             'Expected goals for minus against per game',                  POS),
    ('shot_quality_disparity',  'Shot Quality Disparity',   'xG/shot for minus xG/shot against (chance danger level)',    ACCENT),
    ('conversion_disparity',    'Conversion Disparity',     'Goals/shot for minus goals/shot against (finishing edge)',   ACCENT),
    ('finishing_disparity_pg',  'Finishing Disparity',      'Goals minus expected goals per game (luck / skill vs xG)',   ACCENT),
    ('penalty_disparity_pg',    'Penalty Disparity',        'Penalties taken minus opponent penalties per game',          NEG),
    ('home_away_split',         'Home / Away Split',        'Home goal margin minus away goal margin (home dependence)',  ACCENT),
    ('composite_disparity_score','Composite Disparity Score','Z-score average across 5 key metrics (overall team quality)', '#FFD700'),
]

output_dir = Path('../disparity_plots')
output_dir.mkdir(exist_ok=True)

def bar_color(values, base_color, metric):
    """Colour bars: positive = cyan/accent, negative = red, unless metric is always-styled."""
    always_styled = {'penalty_disparity_pg', 'home_away_split'}
    if metric in always_styled:
        return [base_color if v >= 0 else NEG for v in values]
    return [POS if v >= 0 else NEG for v in values]


for col, title, subtitle, base_color in METRICS:
    data = df[col].sort_values(ascending=True).dropna()
    teams = [t.replace('_', ' ').title() for t in data.index]
    values = data.values

    fig, ax = plt.subplots(figsize=(14, 9))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG_PANEL)

    colors = bar_color(values, base_color, col)
    bars = ax.barh(teams, values, color=colors, height=0.65, zorder=3)

    # Zero line
    ax.axvline(0, color=ZERO_LINE, linewidth=1.2, zorder=4)

    # Grid
    ax.xaxis.grid(True, zorder=0, alpha=0.4)
    ax.set_axisbelow(True)

    # Value labels on bars
    for bar, val in zip(bars, values):
        x_off = 0.0005 if val >= 0 else -0.0005
        ha = 'left' if val >= 0 else 'right'
        label_x = bar.get_width() + x_off
        ax.text(label_x, bar.get_y() + bar.get_height() / 2,
                f'{val:+.3f}', va='center', ha=ha,
                fontsize=6.5, color=WHITE, alpha=0.85)

    # Axes
    ax.tick_params(axis='y', labelsize=8, pad=4)
    ax.tick_params(axis='x', labelsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(GRID)
    ax.spines['bottom'].set_color(GRID)

    # League average line
    league_avg = values.mean()
    ax.axvline(league_avg, color='#FFD700', linewidth=0.9,
               linestyle='--', zorder=4, alpha=0.7)
    ax.text(league_avg, len(teams) - 0.3, f' avg {league_avg:+.3f}',
            fontsize=7, color='#FFD700', va='top', alpha=0.85)

    # Title block
    fig.text(0.13, 0.96, title.upper(),
             fontsize=15, fontweight='bold', color=WHITE, va='top')
    fig.text(0.13, 0.925, subtitle,
             fontsize=8.5, color=DIM, va='top')

    # Legend
    pos_patch = mpatches.Patch(color=POS, label='Positive (favours team)')
    neg_patch = mpatches.Patch(color=NEG, label='Negative (against team)')
    avg_line  = plt.Line2D([0], [0], color='#FFD700', linestyle='--', linewidth=1, label='League average')
    ax.legend(handles=[pos_patch, neg_patch, avg_line],
              loc='lower right', fontsize=7.5,
              facecolor='#1A1A1A', edgecolor=GRID,
              labelcolor=WHITE, framealpha=0.9)

    # X label
    ax.set_xlabel(col, fontsize=8, color=DIM, labelpad=6)

    plt.tight_layout(rect=[0, 0, 1, 0.93])

    fname = output_dir / f'{col}.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f"  Saved {fname}")

print(f"\nAll {len(METRICS)} plots saved to ./{output_dir}/")