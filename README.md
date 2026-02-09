# WHL Strategic Analytics: The "Deception & Sustainability" Model

## 1. Project Overview
This research project provides a multi-dimensional analysis of **Western Hockey League (WHL)** performance, moving beyond traditional standings to identify **True Contenders**, **Paper Tigers**, and **Strategic Archetypes**. By processing raw play-by-play data, this model quantifies team identities through the lens of puck possession, goaltending efficiency, and roster sustainability.

---

## 2. The Analytical Pillars

### I. The Deception Gap (Luck vs. Dominance)
Traditional win percentages are often inflated by overtime (OT) results, which are statistically high-variance. We isolate **Regulation Win Percentage** and compare it to total wins to calculate the **"Deception Gap."**
* **Pure Juggernauts:** Teams with high possession (Corsi) and low deception gaps (dominance in 60 minutes).
* **Paper Tigers:** Teams reliant on OT/Shootout "coin-flips" to maintain their position in the standings.

### II. The Gatekeeper Factor (GSAx)
We utilize **Goals Saved Above Expected (GSAx)** to decouple defensive system quality from individual goaltending performance.
* **The Sieve Alert:** Identifies teams with elite defensive structures (low xGA) being sabotaged by sub-par goaltending.
* **Goalie Heroes:** Teams masking defensive flaws through elite "Gatekeeper" performance.

### III. Fatigue & Roster Sustainability
Using **Time On Ice (TOI) Normalization**, we calculate a **Fatigue Ratio** for every offensive unit.
* **Workload Status:** Dynamically flags **"FATIGUED"** units (Top 25% of league usage).
* **Sustainability Risk:** Identifies "One-Line Wonders" overworking their top stars, signaling a high probability of performance collapse in compressed playoff schedules.

---

## 3. Key Findings (February 2026)

### The "Hidden Contenders"
Based on our **PowerScore** (Weighted 50% Defense System / 50% Goaltending), we identified teams undervalued by the current standings. These teams exhibit:
1.  **Negative Deception Gap:** Underperforming their true talent due to OT bad luck.
2.  **"Fresh" Workload Status:** High-performing second lines with low fatigue ratios.
3.  **High 5-on-5 Purity:** Consistent elite performance from the top-six forward group.

### Strategic Archetypes Identified:

| Archetype | Characteristics | Strategic Outlook |