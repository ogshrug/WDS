import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm
from matplotlib.lines import Line2D
from scipy import stats

# ── Register Fira Sans from local path ───────────────────────────────────────
FIRA_PATH = '../../Fira_Sans'
FIRA_VARIANTS = [
    'Regular', 'Bold', 'Italic', 'BoldItalic',
    'Medium', 'MediumItalic', 'Light', 'LightItalic',
    'SemiBold', 'SemiBoldItalic', 'ExtraLight', 'ExtraLightItalic',
    'Thin', 'ThinItalic', 'Black', 'BlackItalic',
    'ExtraBold', 'ExtraBoldItalic',
]
loaded = []
for v in FIRA_VARIANTS:
    path = f'{FIRA_PATH}/FiraSans-{v}.ttf'
    if os.path.exists(path) and os.path.getsize(path) > 0:
        fm.fontManager.addfont(path)
        loaded.append(v)

if loaded:
    plt.rcParams['font.family'] = 'Fira Sans'
    FONT = 'Fira Sans'
    print(f"Fira Sans loaded: {loaded}")
else:
    # Fallback to Poppins if Fira Sans not found
    for variant in ['Regular', 'Bold', 'Medium', 'Light', 'Italic', 'BoldItalic']:
        p = f'/usr/share/fonts/truetype/google-fonts/Poppins-{variant}.ttf'
        if os.path.exists(p):
            fm.fontManager.addfont(p)
    plt.rcParams['font.family'] = 'Poppins'
    FONT = 'Poppins'
    print("Fira Sans not found — falling back to Poppins")

# ── Data — single CSV (whl_2025_line_disparityv3.csv) ────────────────────────
df = pd.read_csv('../data/whl_2025_line_disparityv3.csv')

x     = df['disparity_ratio'].values
y     = df['composite_disparity_score'].values
teams = [t.replace('_', ' ').title() for t in df['team']]

# ── Regression ────────────────────────────────────────────────────────────────
slope, intercept, r, p, se = stats.linregress(x, y)
x_fit = np.linspace(x.min() - 0.03, x.max() + 0.03, 300)
y_fit = slope * x_fit + intercept
r2    = r ** 2

# ── Palette — warm dark background, ember/sage accent ────────────────────────
BG        = '#0F0E0C'   # very dark warm black
PANEL     = '#161410'   # warm dark panel
GRID_C    = '#242018'   # subtle warm grid
WHITE     = '#F5F0E8'   # warm off-white text
DIM       = '#C8BFB0'   # near-white warm dim
REF_LINE  = '#5A5040'   # warm reference lines

# Tier dot colours — earthy, high-contrast
TIER_COLORS = {
    '1 - Elite':           '#E8A838',   # amber gold
    '2 - Above Average':   '#6BBF6A',   # sage green
    '3 - Average':         '#7BB8D4',   # dusty blue
    '4 - Below Average':   '#D47A5A',   # terracotta
    '5 - Poor':            '#C45C7A',   # muted rose
}
TIER_LABELS = {
    '1 - Elite':           'Elite',
    '2 - Above Average':   'Above Average',
    '3 - Average':         'Average',
    '4 - Below Average':   'Below Average',
    '5 - Poor':            'Poor',
}

dot_colors = [TIER_COLORS[t] for t in df['tier']]

# ── Figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(15, 9.5))
fig.patch.set_facecolor(BG)
ax.set_facecolor(PANEL)

xpad = 0.055
ypad = 0.30
xlim = (x.min() - xpad, x.max() + xpad)
ylim = (y.min() - ypad, y.max() + ypad)
ax.set_xlim(*xlim)
ax.set_ylim(*ylim)

# ── Quadrant fills ────────────────────────────────────────────────────────────
xmid, ymid = 1.0, 0.0
kw = dict(zorder=0, alpha=0.18)
ax.fill_betweenx([ymid, ylim[1]], xlim[0], xmid,  color='#6BBF6A', **kw)  # TL: balanced+strong
ax.fill_betweenx([ymid, ylim[1]], xmid,  xlim[1], color='#E8A838', **kw)  # TR: top-heavy+strong
ax.fill_betweenx([ylim[0], ymid], xlim[0], xmid,  color='#7BB8D4', **kw)  # BL: balanced+weak
ax.fill_betweenx([ylim[0], ymid], xmid,  xlim[1], color='#C45C7A', **kw)  # BR: top-heavy+weak

# Quadrant labels
qkw = dict(fontsize=8, color='#D8D0C4', style='italic', zorder=3, ha='center',
           fontproperties=fm.FontProperties(family=FONT, style='italic', size=8))
ax.text(0.935, ylim[1] - 0.08, 'Balanced\n& Strong',   va='top',    **qkw)
ax.text(1.335, ylim[1] - 0.08, 'Top-Heavy\n& Strong',  va='top',    **qkw)
ax.text(0.935, ylim[0] + 0.06, 'Balanced\n& Weak',     va='bottom', **qkw)
ax.text(1.335, ylim[0] + 0.06, 'Top-Heavy\n& Weak',    va='bottom', **qkw)

# Reference lines
ax.axvline(xmid, color=REF_LINE, linewidth=1.1, linestyle='--', zorder=2)
ax.axhline(ymid, color=REF_LINE, linewidth=1.1, linestyle='--', zorder=2)

# ── Trend line ────────────────────────────────────────────────────────────────
ax.plot(x_fit, y_fit, color='#F5F0E8', linewidth=1.4,
        linestyle=(0, (6, 3)), alpha=0.55, zorder=4)

# ── Scatter dots ──────────────────────────────────────────────────────────────
ax.scatter(x, y, c=dot_colors, s=100, zorder=5,
           edgecolors='#2A2520', linewidths=0.8)

# ── Team labels ───────────────────────────────────────────────────────────────
OFFSETS = {
    'Brazil':        (-0.004, +0.12),
    'Thailand':      (-0.004, +0.12),
    'Netherlands':   (+0.005, +0.11),
    'Pakistan':      (+0.005, -0.17),
    'Usa':           (-0.004, +0.11),
    'Uae':           (+0.005, -0.16),
    'Switzerland':   (-0.004, -0.16),
    'Rwanda':        (+0.005, +0.10),
    'Mongolia':      (-0.004, -0.17),
    'Guatemala':     (+0.005, +0.11),
    'Vietnam':       (-0.004, -0.16),
    'Oman':          (+0.005, +0.11),
    'Kazakhstan':    (-0.004, +0.11),
    'France':        (-0.004, -0.16),
    'Iceland':       (+0.005, -0.16),
}
fp_label = fm.FontProperties(family=FONT, size=6.8)
for xi, yi, label in zip(x, y, teams):
    ox, oy = OFFSETS.get(label, (0.005, 0.10))
    ax.text(xi + ox, yi + oy, label,
            fontproperties=fp_label, color=WHITE,
            alpha=0.88, ha='center', va='bottom', zorder=6)

# ── Axes styling ──────────────────────────────────────────────────────────────
fp_axis  = fm.FontProperties(family=FONT, size=9)
fp_tick  = fm.FontProperties(family=FONT, size=8)

ax.set_xlabel(
    'Offensive Line Quality Disparity Ratio\n'
    '(1st Line adj. xG/60  ÷  2nd Line adj. xG/60  |  even-strength, TOI & matchup-adjusted)',
    fontproperties=fp_axis, color=DIM, labelpad=10
)
ax.set_ylabel(
    'Overall Team Strength\n'
    '(Composite Disparity Score — z-score avg. across 5 metrics)',
    fontproperties=fp_axis, color=DIM, labelpad=10
)

for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontproperties(fp_tick)
    label.set_color('#D8D0C4')

for spine in ax.spines.values():
    spine.set_color(GRID_C)
ax.grid(True, color=GRID_C, linewidth=0.6, alpha=0.8)
ax.set_axisbelow(True)

# ── Legend ────────────────────────────────────────────────────────────────────
fp_leg = fm.FontProperties(family=FONT, size=8)

tier_handles = [
    mpatches.Patch(facecolor=TIER_COLORS[t], edgecolor='#2A2520',
                   linewidth=0.6, label=TIER_LABELS[t])
    for t in TIER_COLORS
]
trend_handle = Line2D([0], [0], color=WHITE, linewidth=1.4,
                      linestyle=(0, (6, 3)), alpha=0.55,
                      label=f'Trend line  (r = {r:.2f},  p = {p:.2f})')
ref_handle   = Line2D([0], [0], color=REF_LINE, linewidth=1.1,
                      linestyle='--', label='League averages (x = 1.0,  y = 0.0)')

leg1 = ax.legend(handles=tier_handles,
                 title='Overall Team Tier', title_fontsize=8,
                 prop=fp_leg, loc='upper left',
                 facecolor='#1C1914', edgecolor=GRID_C,
                 labelcolor=WHITE, framealpha=0.92)
leg1.get_title().set_color(WHITE)
leg1.get_title().set_fontproperties(
    fm.FontProperties(family=FONT, weight='bold', size=8))
ax.add_artist(leg1)

leg2 = ax.legend(handles=[trend_handle, ref_handle],
                 prop=fp_leg, loc='lower left',
                 facecolor='#1C1914', edgecolor=GRID_C,
                 labelcolor=WHITE, framealpha=0.92)
ax.add_artist(leg2)

# ── Annotation box — key finding ─────────────────────────────────────────────
finding = (
    f"r = {r:.2f}  |  R² = {r2:.3f}  |  p = {p:.2f}\n"
    "No significant relationship detected.\n"
    "Line depth disparity shows no correlation\nwith team strength."
)
ax.text(0.985, 0.97, finding,
        transform=ax.transAxes, ha='right', va='top',
        fontproperties=fm.FontProperties(family=FONT, weight='bold', size=7.5),
        color=WHITE, alpha=0.95,
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#1C1914',
                  edgecolor=GRID_C, alpha=0.9))

# ── Title block ───────────────────────────────────────────────────────────────
fig.text(0.07, 0.975,
         'Does Offensive Line Balance Drive Team Success?',
         fontproperties=fm.FontProperties(family=FONT, weight='bold', size=15),
         color=WHITE, va='top')
fig.text(0.07, 0.942,
         f'WHL 2025  ·  32 teams  ·  Elite teams appear across both balanced and top-heavy lines — '
         f'line depth disparity shows no correlation with overall team strength (r = {r:.2f}).',
         fontproperties=fm.FontProperties(family=FONT, weight='bold', size=8.5),
         color=WHITE, va='top')

# ── Caption ───────────────────────────────────────────────────────────────────
fig.text(0.07, 0.018,
         'Disparity ratio: 1st line adj.xG/60 ÷ 2nd line adj.xG/60 (even-strength only, adjusted for TOI and '
         'opposing defensive pairing quality).  Team strength: composite z-score across goal, shot, xG, '
         'shot quality and conversion disparities (per game, normalised across all 32 teams).',
         fontproperties=fm.FontProperties(family=FONT, size=6.5),
         color='#A09080', va='bottom')

plt.tight_layout(rect=[0, 0.05, 1, 0.93])
plt.savefig('../disparity_plots/SNAASZ.png', dpi=160, bbox_inches='tight', facecolor=BG)
plt.close()
print("SNAASZm.png")