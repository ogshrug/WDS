# --- LOGIC FOR AWAY COLUMNS (Road Resilience & System Strength) ---
# 1. Road Resilience Score: Aggregate team xG when playing as 'away_team'.
#    - Compare 'Away xG/60' vs 'Home xG/60'.
#    - ACTION: Teams with the smallest "Home-Road Gap" get a Reliability Bonus.
#      They are 'System-Strong' and play well regardless of environment.
#
# 2. Defensive Opponent-Adjustment: Use away_def_pairing to 'weight' offensive success.
#    - Scoring against an opponent's 'first_def' is worth more Power Points
#      than scoring against their 'second_def'.
#    - ACTION: Create a 'Difficulty-Adjusted Goal' metric.
#
# 3. The "Last Change" Penalty: On the road, teams cannot control line matchups.
#    - If a team's 'away_off_line' (1st) still dominates while being 'hunted'
#      by the home coach, they are a Top-Tier Juggernaut.
#
# 4. Data Normalization: Combine Home and Away stats into a single 'Neutral Table'.
#    - This ensures a team's Power Ranking is based on their WHOLE season,
#      not just a favorable home schedule.
#%%

# --- LOGIC FOR GOALIE COLUMNS (The Gatekeeper Factor) ---
# 1. GSAx (Goals Saved Above Expected): Compare 'Actual Goals Allowed' vs 'Total xG'.
#    - Formula: GSAx = Total xG - Actual Goals.
#    - ACTION: High GSAx = Elite Goalie. Low GSAx = Weak Link.
#
# 2. Standings vs. Process: Identify teams 'carried' by their goalie.
#    - If a team wins despite being out-shot (low xG share), their rank is FRAGILE.
#    - ACTION: Weight 'Team xG' higher than 'Actual Wins' to find sustainable power.
#
# 3. Goalie Split: Check if a team has a clear 'Starter' vs 'Backup'.
#    - Does the team's Win % drop significantly when the backup goalie is in?
#    - ACTION: Create a 'Roster Reliability' score based on the gap between goalies.
#
# 4. The 'Sieve' Alert: Flag teams with high xG suppression (great defense)
#    but high Goals Against (bad goalie). These are 'Sleepers' if they swap goalies.
#%%
# --- LOGIC FOR TOI (The Normalization Key) ---
# 1. Rate Normalization: NEVER use raw xG or Goals for rankings.
#    Always calculate (Stat / toi) * 3600 to get the 'Per 60' rate.
#
# 2. Fatigue Analysis: Track total TOI for 'first_def' and 'first_off'.
#    Teams with extreme workloads for top units should get a 'Sustainability Penalty'
#    in long-term power rankings.
#
# 3. Small Sample Filter: Ignore or down-weight lines with less than
#    a certain threshold of total TOI (e.g., 500 seconds) to avoid 'fluke' stats.
#
# 4. Efficiency Mapping: Combine TOI with xG to see which lines are the
#    most 'lethal' per minute played.
#%%
# --- LOGIC FOR SHOTS & ASSISTS (Style & Luck Filter) ---
# 1. Shooting Percentage: (home_goals / home_shots).
#    - Identify 'Sustainability': If a team's shooting % is way above the league average,
#      expect their Power Ranking to drop later (regression).
#
# 2. Playmaking Grade: (home_assists / home_goals).
#    - High ratios indicate 'System Teams' with high puck movement.
#    - Low ratios indicate 'Individualist Teams' (reliant on solo efforts/turnovers).
#
# 3. Chaos Generator: Identify teams with high 'Shots per 60' but low xG.
#    - These teams play a 'dirty' game—relying on rebounds and volume rather than skill.
#
# 4. Assist Map: Link assists to off_line.
#    - Does the 1st line rely on assists while the 2nd line scores solo?
#    - ACTION: Use this to determine which line is easier to 'scout' and shut down.
#%%
# --- LOGIC FOR PENALTIES (The Discipline & Chaos Metric) ---
# 1. Discipline Rating: Calculate 'Penalty Minutes per 60'.
#    - Identify teams that 'beat themselves'. A high-penalizing team
#      should have their Power Score docked for 'Unreliability'.
#
# 2. Special Teams Exposure: Compare 'home_penalties' vs 'away_penalties'.
#    - If away_penalties >> home_penalties, the team is mentally fragile on the road.
#
# 3. Penalty-Adjusted xG: Create a 'Clean-Play xG' by filtering out records
#    where home_penalty == 1. This shows how good a team is when playing fair.
#
# 4. The 'Instigator' Factor: Does a team draw more penalties than they take?
#    - Compare 'home_penalties' vs 'away_penalties' in the same game.
#    - Teams that 'draw' penalties have high 'Functional Aggression'.
#%%
--- LOGIC FOR went_ot (The Volatility Filter) ---
# 1. Regulation Performance: Use this to isolate 'Regulation Goal Differential'.
#    A team winning 5-0 in regulation is significantly stronger than a team
#    winning 1-0 in OT. The former shows dominance; the latter shows a coin-flip.
#
# 2. 'The Paper Tiger' Check: Identify teams with high standings but high OT win rates.
#    If a team relies on OT/Shootouts, their Power Ranking should be ADJUSTED DOWN
#    as OT results are less repeatable than 5-on-5 play.
#
# 3. 'The Resilience' Factor: Boost teams with high OT Loss counts.
#    In the standings, they look like losers (0 wins), but in reality,
#    they are competitive enough to hold elite teams to a draw for 60 minutes.
#
# 4. Usage Normalization: Since OT adds extra 'toi', always use 'per 60 minutes'
#    rates (e.g., xG/60) to ensure OT minutes don't artificially inflate total stats.

 --- LOGIC FOR home_off_line (The Roster Strength Factor) ---
# 1. Roster Depth: Compare 'first_off' vs 'second_off' xG/60.
#    - 'One-Line Wonders': Teams with a huge drop-off in quality (e.g., 1st line 3.0 xG, 2nd line 0.5 xG).
#    - 'Balanced Giants': Teams where both lines produce consistently.
#    ACTION: Reward 'Balanced' teams with a higher stability score in rankings.
#
# 2. Situational Power: Isolate 'PP_up' (Power Play) records.
#    - Standing might be low, but if 'PP_up' xG/60 is top 5, they are a 'Danger Team'.
#    ACTION: Add a 'Special Teams Grade' to the final Power Ranking.
#
# 3. 5-on-5 Purity: Filter for 'first_off' and 'second_off' only to find 'Even-Strength' dominance.
#    - This is the most repeatable part of hockey.
#    ACTION: Use Even-Strength xG Differential as 50% of the total Power Ranking weight.
#
# 4. Tactical Matchups: Link with 'away_def_pairing' to see which lines 'crush' weaker defenders.
#    - Identify teams that successfully hunt mismatches (e.g., first_off vs. opponent's second_def).
# --- LOGIC FOR AWAY COLUMNS (Road Resilience & System Strength) ---
# 1. Road Resilience Score: Aggregate team xG when playing as 'away_team'.
#    - Compare 'Away xG/60' vs 'Home xG/60'.
#    - ACTION: Teams with the smallest "Home-Road Gap" get a Reliability Bonus.
#      They are 'System-Strong' and play well regardless of environment.
#
# 2. Defensive Opponent-Adjustment: Use away_def_pairing to 'weight' offensive success.
#    - Scoring against an opponent's 'first_def' is worth more Power Points
#      than scoring against their 'second_def'.
#    - ACTION: Create a 'Difficulty-Adjusted Goal' metric.
#
# 3. The "Last Change" Penalty: On the road, teams cannot control line matchups.
#    - If a team's 'away_off_line' (1st) still dominates while being 'hunted'
#      by the home coach, they are a Top-Tier Juggernaut.
#
# 4. Data Normalization: Combine Home and Away stats into a single 'Neutral Table'.
#    - This ensures a team's Power Ranking is based on their WHOLE season,
#      not just a favorable home schedule.