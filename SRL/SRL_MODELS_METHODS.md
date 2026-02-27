# SRL: Calculation Methods for the Three Benchmark Models

**Language: English.** This document records the **specific calculation methods** used for the three heterogeneous-agent models in the Structural Reinforcement Learning (SRL) paper and notebooks.

---

## 1. Huggett (1993) Model with Aggregate Risk

### 1.1 Model summary

- **Agents:** Continuum of households with uninsured idiosyncratic labor income risk; save in bonds. Bonds are in **zero net supply**.
- **Individual state:** $(b, y)$ — bond holdings and idiosyncratic income.
- **Aggregate state:** TFP $z_t$ (stochastic); in equilibrium, the cross-sectional distribution $G_t(b,y)$.
- **Budget (per period):** $c + b' = (1+r)b + y\,z$, with borrowing constraint $b \geq \underline{b}$.

### 1.2 Calculation method (SRL)

- **State space:** Discretize $(b,y)$ on a finite grid (e.g. $n_b \times n_y$ points). Idiosyncratic $y$ and aggregate $z$ are discretized (e.g. Tauchen) with transition matrices $T_y$, $T_z$.
- **Policy:** Consumption and saving rules $c(b,y,z,r)$, $b'(b,y,z,r)$ depend on **current** price $r$ (and optionally short price history). Policy is represented on the grid; off-grid values by interpolation.
- **Equilibrium:** At each date, interest rate $r_t$ clears the bond market: aggregate saving equals zero. Given a candidate policy, simulate paths of $(z_t, r_t)$, update distribution via the **histogram method** (Young 2010): $g_{t+1} = A_{\pi(z_t,r_t)}^T g_t$, where $A_\pi$ is the individual-state transition matrix induced by the policy. At each $t$, solve for $r_t$ such that $\sum_{b,y} b'(b,y,z_t,r_t)\, g_t(b,y) = 0$.
- **Optimization:** **Structural policy gradient (SPG).** Value of a policy is approximated by Monte Carlo over simulated $(z,p)$ paths; gradient of value w.r.t. policy parameters is computed by **differentiating through** the known individual transition $A_\pi$; price process is learned from simulation only. Maximize expected discounted utility over policy parameters (e.g. grid values or parametrized rules).

### 1.3 Key equations

- Preferences: $E_0 \sum_{t\ge0} \beta^t u(c_t)$, with $u(c) = c^{1-\sigma}/(1-\sigma)$ (CRRA).
- Budget: $c_{i,t} + b_{i,t+1} = (1+r_t)b_{i,t} + y_{i,t}z_t$; $b \geq \underline{b}$.
- Market clearing: $\int b'(b,y,z_t)\,dG_t(b,y) = 0$ at all $t$.

### 1.4 Implementation notes (notebook)

- **hugget.ipynb:** Calibration (e.g. Table 2 in SRL); grids for $b$, $y$, $z$, $r$; Tauchen for $y$ (level AR(1), $E[y]=1$) and $z$ (log AR(1)); truncation horizon $T$ and $c_{\min}$ floor.
- **Policy:** Returns **raw** $(c, b')$ (continuous); no lottery inside the policy. Young or soft weights are applied only when updating the distribution (e.g. `update_G_direct`, `update_G_pi_direct`). Helper `b_next_to_grid_lottery` is used only when discrete $b'$ is needed (e.g. agent-level simulation).
- **Distribution update:** $G_{t+1}$ is computed from $G_t$ and the policy **without** building the full transition matrix $A_\pi$: `update_G_direct` (NumPy, Young weights) and `update_G_pi_direct` (PyTorch, soft weights).
- **Two-phase training:** Warm-up with fixed $G_0$ (steady-state under initial $\theta$ at mid $(z,r)$); after $N_{\text{warm-up}}$, $G$ evolves each period. See main repo README Appendix (SRL) for details.

---

## 2. Krusell–Smith (1998) Model

### 2.1 Model summary

- **Agents:** Households with idiosyncratic labor productivity; save in **productive capital** rented to a representative firm.
- **Individual state:** $(k, y)$ — capital and idiosyncratic labor productivity.
- **Aggregate:** Total capital $K_t = \int k\,dG_t(k,y)$; TFP $z_t$. Factor prices: $r_t = \alpha z_t K_t^{\alpha-1}L_t^{1-\alpha} - \delta$, $w_t = (1-\alpha)z_t K_t^\alpha L_t^{-\alpha}$ (with $L_t=1$ inelastic).
- **Budget:** $c + k' = (1+r)k + w\,y$; $k \geq 0$.

### 2.2 Calculation method (SRL)

- **State space:** Discretize $(k,y)$; $y$ and $z$ discretized (e.g. Tauchen). Policies depend on **current** prices $(r,w)$ (and optionally $z$ or price history).
- **Equilibrium:** Given policy, aggregate capital is $K_t = \sum_{k,y} k\, g_t(k,y)$ (or integral over distribution). Prices follow $r_t = r(K_t,z_t)$, $w_t = w(K_t,z_t)$ from firm FOCs. Distribution evolves: $g_{t+1} = A_{\pi(z_t,r_t,w_t)}^T g_t$. No separate market-clearing root-find for a price; $K_t$ is determined by the distribution.
- **Optimization:** Same SPG idea: value by Monte Carlo over $(z,r,w)$ paths; exact gradient w.r.t. policy via differentiation through $A_\pi$; price dynamics from simulation. Solve for policy that maximizes expected discounted utility.

### 2.3 Key equations

- Production: $Y_t = z_t K_t^\alpha L_t^{1-\alpha}$, $L_t=1$; $r_t = \alpha Y_t/K_t - \delta$, $w_t = (1-\alpha)Y_t/L_t$.
- Budget: $c + k' = (1+r_t)k + w_t y$; $k \geq 0$.
- Aggregation: $K_t = \int k\,dG_t(k,y)$.

### 2.4 Implementation notes (notebook)

- **krusell_smith.ipynb:** Calibration (SRL Section 4.2, Appendix A.2, Tables 4–5); grids for $k$, $y$, $z$, $r$, $w$; Tauchen for $y$ (level AR(1), $E[y]=1$) and $z$ (log AR(1)); same truncation and $c_{\min}$ as elsewhere.
- **Policy:** Returns **raw** $(c, n, b')$ (continuous); `policy_from_grid_ks` and `b_next_to_grid_lottery` follow the same design as Huggett (lottery only when mapping to grid or when discrete $b'$ is needed).
- **Distribution update:** $d_{t+1}$ from $d_t$ via `update_d_direct` (NumPy, Young) and `update_d_pi_direct_ks` (PyTorch, soft weights) **without** building the full $A_\pi$. Aggregate capital $K = d \cdot b$; $(r, w) = \texttt{K_to_prices}(K, z)$.
- **Two-phase training:** Warm-up with fixed $d_0$ (steady-state under initial $\theta$); then $d$ evolves. SPG hyperparameters in calibration cell; “Using Trained Policy” cell with try/except. See main repo README Appendix (SRL) for details.

---

## 3. One-Account HANK Model (with Forward-Looking Phillips Curve)

### 3.1 Model summary

- **Households:** Heterogeneous in wealth and idiosyncratic income; **endogenous labor** $n_t$. Single asset (bonds). Budget: $c_t + b_{t+1} = (1+r_t)b_t + w_t y_t n_t + d_t - T_t$, $b_{t+1} \geq 0$.
- **Firms:** Monopolistic competition; prices subject to Rotemberg adjustment costs → **forward-looking Phillips curve**.
- **Policy:** Taylor rule $1+i_t = \bar{R}(1+\Pi_t)^\phi e^{e_t}$; Fisher $R_t = (1+i_t)/(1+\Pi_t)$; monetary shock $e_t$.
- **Aggregate:** Bond market clearing; labor market clearing; dividends $d_t$ and lump-sum taxes $T_t$ (e.g. $r_t B = T_t$).

### 3.2 Calculation method (SRL)

- **State space:** Discretize $(b,y)$; discretize $z$, $e$ (and possibly other aggregates). Policies $c(b,y,r,w,z)$, $n(b,y,r,w,z)$ (and possibly other states) depend on **current** prices $(r,w)$ and aggregates.
- **Equilibrium:** For a given policy, at each $t$: (i) aggregate labor supply and bond demand from the distribution and policy; (ii) labor and bond market clearing give $(r_t,w_t)$; (iii) Phillips curve and Taylor rule determine inflation and nominal rate; (iv) distribution updates via $g_{t+1} = A_{\pi}^T g_t$.
- **Optimization:** SPG again: value by Monte Carlo over paths of $(z,e,r,w,\ldots)$; exact policy gradient via $A_\pi$; price and aggregate dynamics from simulation. Household and firm problems can be solved **jointly** (same SPG for consumption–labor and for price-setting).

### 3.3 Key equations

- Household: $\max E_0 \sum \beta^t u(c_t,n_t)$ s.t. $c_t + b_{t+1} = (1+r_t)b_t + w_t y_t n_t + d_t - T_t$, $b_{t+1} \geq 0$.
- Taylor: $1+i_t = \bar{R}(1+\Pi_t)^\phi e^{e_t}$; real rate $R_t = (1+i_t)/(1+\Pi_t)$.
- Phillips curve (Rotemberg): forward-looking in inflation; firms set prices given demand and adjustment costs.

### 3.4 Implementation notes (notebook)

- **one_account_hank.ipynb:** Calibration (SRL Table 6, Appendix A.3); preferences (CRRA in $c$, separable labor, Frisch elasticity); grids for $b$, $y$, $z$, $e$, $r$, $w$; can use a simplified price process for a household-only block before plugging in the full NK block.

---

## 4. Common SRL Ingredients (All Three Models)

| Step | Description |
|------|--------------|
| **1. Discretize individual state** | $(b,y)$ or $(k,y)$ on a finite grid; income processes by Tauchen or similar. |
| **2. Restrict policy to low-dimensional state** | Policy depends on current (and optionally lagged) prices, not on the full distribution. |
| **3. Simulate economy** | For a given policy, iterate: draw $z_{t+1}$ (and $e_{t+1}$ in HANK); update distribution $g_{t+1} = A_\pi^T g_t$; solve market clearing for prices $p_t$; record $(z_t,p_t)$. |
| **4. Value by Monte Carlo** | Approximate $v^\pi(s,z,p)$ by averaging over $N$ simulated paths; truncate at horizon $T$. |
| **5. Structural policy gradient** | Differentiate value w.r.t. policy parameters using the **known** transition $A_\pi$; no gradient through price process. Optimize (e.g. stochastic gradient ascent) to maximize value. |
| **6. Equilibrium** | Restricted perceptions equilibrium: policy optimal given simulated price process; prices consistent with market clearing given that policy. |

---

## References (in-document)

- SRL paper (SRL.tex): Setup (Section 2), SRL/SPG (Section 3), Experiments (Section 4).
- Notebooks: **hugget.ipynb**, **krusell_smith.ipynb**, **one_account_hank.ipynb** in this folder.
