"""
WHSDSC 2026 — Phase 1b (FINAL)
================================
Top 10 teams with highest offensive line quality disparity
between first and second offensive lines.

Methodology
-----------
1. Filter to even-strength shifts only (first_off / second_off vs first_def / second_def).
   PP, PK, and empty-net rows excluded — they use different personnel and skew rates.

2. Compute raw xG/60 per line per team:
   xG/60 = (sum of xg_for / sum of toi) × 3600

3. Adjust for defensive matchup quality:
   - first_def concedes xGA/60 = 2.170  (tougher)
   - second_def concedes xGA/60 = 2.351 (weaker)
   - adj_factor = league_avg_def_xGA60 / opponent_def_xGA60
   - Lines that faced the tougher first_def get boosted; lines that faced
     the easier second_def get penalized. This is critical because first
     lines face first_def pairings 51.6% of the time vs 49.2% for second
     lines — without adjustment, first lines are systematically understated.

4. Compute disparity ratio = first_adj_xg60 / second_adj_xg60
   Per webinar clarification: "top 10 = largest disparity" = highest ratio.

Validated:
- All 32 teams have both first_off and second_off data (no missing)
- TOI split is nearly 50/50 (0.502–0.509) so TOI weighting is fair
- Defensive adjustment direction confirmed: adj_factor > 1 for first_def ✓
"""

import pandas as pd
import numpy as np

# ── 0. Load ───────────────────────────────────────────────────────────────────
df = pd.read_excel("whl_2025.xlsx")

ES_LINES = {'first_off', 'second_off'}
ES_DEF   = {'first_def', 'second_def'}

# ── 1. Filter to even-strength only ──────────────────────────────────────────
home = df[
    df['home_off_line'].isin(ES_LINES) &
    df['away_def_pairing'].isin(ES_DEF)
].copy()
home = home.rename(columns={
    'home_team':      'team',
    'home_off_line':  'off_line',
    'away_def_pairing':'opp_def',
    'home_xg':        'xg_for',
})

away = df[
    df['away_off_line'].isin(ES_LINES) &
    df['home_def_pairing'].isin(ES_DEF)
].copy()
away = away.rename(columns={
    'away_team':      'team',
    'away_off_line':  'off_line',
    'home_def_pairing':'opp_def',
    'away_xg':        'xg_for',
})

shifts = pd.concat([
    home[['team', 'off_line', 'opp_def', 'toi', 'xg_for']],
    away[['team', 'off_line', 'opp_def', 'toi', 'xg_for']],
], ignore_index=True)

# ── 2. Defensive matchup quality (league-wide) ────────────────────────────────
dq = (shifts.groupby('opp_def')
      .agg(xg=('xg_for','sum'), toi=('toi','sum'))
      .reset_index())
dq['def_xga60'] = dq['xg'] / dq['toi'] * 3600
league_avg_def  = dq['def_xga60'].mean()
dq_dict = dict(zip(dq['opp_def'], dq['def_xga60']))

# ── 3. Weighted avg defensive quality faced per team-line ─────────────────────
def weighted_def_quality(group):
    total_toi = group['toi'].sum()
    if total_toi == 0:
        return np.nan
    return (group['opp_def'].map(dq_dict) * group['toi']).sum() / total_toi

matchup_q = (shifts
    .groupby(['team', 'off_line'])
    .apply(weighted_def_quality, include_groups=False)
    .reset_index(name='avg_def_faced'))

# ── 4. Raw xG/60 per team-line ────────────────────────────────────────────────
line_raw = (shifts
    .groupby(['team', 'off_line'])
    .agg(total_xg=('xg_for','sum'), total_toi=('toi','sum'))
    .reset_index())
line_raw['xg60_raw'] = line_raw['total_xg'] / line_raw['total_toi'] * 3600

# ── 5. Defensive adjustment ───────────────────────────────────────────────────
line_adj = line_raw.merge(matchup_q, on=['team', 'off_line'])
line_adj['adj_factor'] = league_avg_def / line_adj['avg_def_faced']
line_adj['xg60_adj']   = line_adj['xg60_raw'] * line_adj['adj_factor']

# ── 6. Pivot and compute disparity ratio ─────────────────────────────────────
pivot = (line_adj
    .pivot(index='team', columns='off_line', values='xg60_adj')
    .reset_index())
pivot.columns.name = None
pivot = pivot.rename(columns={
    'first_off':  'first_adj_xg60',
    'second_off': 'second_adj_xg60',
})

# ratio = first / second: higher = first line dominates, second lags behind
pivot['disparity_ratio'] = pivot['first_adj_xg60'] / pivot['second_adj_xg60']
pivot = pivot.sort_values('disparity_ratio', ascending=False).reset_index(drop=True)
pivot['rank'] = pivot.index + 1

# ── 7. Print results ──────────────────────────────────────────────────────────
print("=" * 68)
print("PHASE 1b — OFFENSIVE LINE DISPARITY (all 32 teams)")
print("Metric: first_adj_xG60 / second_adj_xG60  (higher = more top-heavy)")
print("=" * 68)
print(f"{'Rk':<4} {'Team':<15} {'1st adj xG/60':>14} {'2nd adj xG/60':>14} {'Ratio':>7}")
print("─" * 58)
for _, row in pivot.iterrows():
    marker = " ◄" if row['rank'] <= 10 else ""
    print(f"{int(row['rank']):<4} {row['team']:<15} "
          f"{row['first_adj_xg60']:>14.4f} "
          f"{row['second_adj_xg60']:>14.4f} "
          f"{row['disparity_ratio']:>7.4f}{marker}")

print("\n" + "=" * 68)
print("PHASE 1b — TOP 10 SUBMISSION")
print("=" * 68)
for _, row in pivot.head(10).iterrows():
    print(f"  {int(row['rank'])}. {row['team']}")

# ── 8. Export ─────────────────────────────────────────────────────────────────
pivot.to_csv("phase1b_line_disparity.csv", index=False)
print("\nSaved: phase1b_line_disparity.csv")