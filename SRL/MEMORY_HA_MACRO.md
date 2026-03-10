# HA Macro Memory (Curated)

Last updated: 2026-03-08
Purpose: compact expert memory for heterogeneous-agent macro modeling (Huggett / Krusell-Smith / HANK), with implementation guidance for this repo.

## 1) Canonical papers (must-know)
1. Huggett (1993), *The risk-free rate in heterogeneous-agent incomplete-insurance economies*, JEDC, 17(5-6):953-969. DOI: 10.1016/0165-1889(93)90024-M
   - Link: https://www.sciencedirect.com/science/article/pii/016518899390024M
   - Core idea: incomplete insurance + borrowing constraint lowers equilibrium risk-free rate.

2. Aiyagari (1994), *Uninsured Idiosyncratic Risk and Aggregate Saving*, QJE 109(3):659-684. DOI: 10.2307/2118417
   - Link: https://academic.oup.com/qje/article/109/3/659/1838287
   - Core idea: precautionary savings in GE with production; wealth distribution affects aggregate capital.

3. Krusell & Smith (1998), *Income and Wealth Heterogeneity in the Macroeconomy*, JPE 106(5):867-896. DOI: 10.1086/250034
   - Link: https://ideas.repec.org/a/ucp/jpolec/v106y1998i5p867-896.html
   - Core idea: "approximate aggregation" via low-dimensional perceived law of motion.

4. Kaplan, Moll, Violante (2018), *Monetary Policy According to HANK*, AER 108(3):697-743. DOI: 10.1257/aer.20160042
   - Link: https://www.nber.org/papers/w21897
   - Core idea: distributional/GE channels dominate intertemporal-substitution channel in HANK.

## 2) Computational methods (state of practice)
1. Reiter (2009), projection + perturbation for HA GE. DOI: 10.1016/j.jedc.2008.08.010
   - Link: https://www.sciencedirect.com/science/article/abs/pii/S0165188908001528
   - Use when: around steady state with aggregate shocks, high-dimensional distribution states.

2. Boppart, Krusell, Mitman (2018), MIT-shock derivative method. NBER WP: https://www.nber.org/papers/w24138
   - Use when: you want linear IRFs from nonlinear transition solver, without explicit recursive law derivation.

3. Auclert, Bardoczy, Rognlie, Straub (2021), Sequence-Space Jacobian (SSJ), Econometrica 89(5):2375-2408.
   - Link: https://www.nber.org/papers/w26123
   - Core tool for: fast solution/estimation of HA models around SS; Jacobian composition + inversion.

4. Achdou, Han, Lasry, Lions, Moll (2022), continuous-time ABH approach, RESTUD 89(1):45-86.
   - Link: https://www.nber.org/papers/w23732
   - Core tool for: PDE/HJB-KFE formulation; efficient finite-difference implementations and theory.

5. Bilal (2023), Master Equation perturbations (FAME/SAME).
   - Link: https://www.nber.org/papers/w31103
   - Use when: many distributional moments/prices matter; higher-order perturbations needed.

## 3) Toolchains worth using
1. Sequence-Jacobian package (shade-econ)
   - Repo: https://github.com/shade-econ/sequence-jacobian
   - Best for: one-/two-asset HANK, KS, estimation, nonlinear perfect-foresight transitions.

2. HARK (Econ-ARK)
   - Docs: https://docs.econ-ark.org/
   - Repo: https://github.com/econ-ark/HARK
   - Best for: modular household block prototyping, lifecycle/consumption-saving variants.

3. Benjamin Moll code archive (continuous-time HA)
   - Link: https://benjaminmoll.com/codes/
   - Best for: reference implementations for Huggett/Aiyagari PDE methods.

4. HANS toolbox
   - Link: https://hans-econ.com/
   - Best for: nonlinear transition paths and large deviations in discrete-time HA models.

## 4) Project-specific implementation rules (for this repo)
1. Huggett market-clearing in training should use bracket interpolation (ir_lo/ir_hi/w_r), not pure argmin over r-grid.
2. For sigma == 1 utility, never use `jnp.where` with CRRA/log branches; use explicit Python branch to avoid NaNs.
3. During warm-up, keep distribution fixed but keep price mapping logic identical to adaptive phase.
4. Always track both raw loss and smoothed loss (MA/EMA) due Monte Carlo trajectory noise.
5. Diagnose policy quality with: monotonicity in b, positive c floor compliance, market-clearing residual path.

## 5) Immediate reading order for this project
1. Huggett (1993) -> Krusell-Smith (1998) -> HANK (Kaplan-Moll-Violante 2018)
2. Auclert et al. (2021 SSJ) + Reiter (2009)
3. Achdou et al. (2022) and Bilal (2023) for advanced solver design
4. Then map code choices to `SRL/hugget_jax.py`, `SRL/krusell_smith_jax.py`, `SRL/one_account_hank_jax.py`

## 6) Related to current SRL line
- CEPR DP20980 (2025): *Structural Reinforcement Learning for Heterogeneous Agent Macroeconomics*.
- Link: https://cepr.org/publications/dp20980
- Relevance: directly aligned with this repository's method choice (price-based state reduction + structural RL).
