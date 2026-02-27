# Neoclassical Growth Model: Solution Methods

This repo presents macroeconomic solutions (variables, policy functions, steady-state) for the neoclassical growth model in several ways.

## Task list

For the listed models, present solutions in the following ways:

- **(i)** Closed-form: pen-and-paper formulas using first-order conditions  
- **(ii)** Bellman equations: pen-and-paper formulas using Bellman equations  
- **(iii)** Numeric policy function iteration  
- **(iv)** Polynomial policy function iteration  
- **(v)** One-layer policy function  
- **(vi)** Multilayer policy function  
- **(vii)** Policy function that feeds closed-form model parameters/gradients into weight matrix optimization  

---

## What has been done (by method)

**(i) Closed-form**  
No dedicated file. Closed-form derivations (Euler equation, steady-state \(k^*\), \(c^*\)) are in `closed_form_neoclassical_growth.tex` (and PDF); full policy \(c(k)\) has no closed form in general.

**(ii) Bellman equations**  
Discrete-time Bellman is stated in `notebooks/growth-pfi.ipynb`. Pen-and-paper Bellman/HJB and steady state are in `closed_form_neoclassical_growth.tex`.

**(iii) Numeric policy function iteration**  
`notebooks/growth-pfi.ipynb` — VFI and Howard policy iteration on a capital grid; \(V(k)\), \(c(k)\), \(k'(k)\), steady state, plots.

**(iv) Polynomial policy function iteration**  
`notebooks/growth-proj.ipynb` — (1) Chebyshev projection (HJB residual zero at collocation nodes). (2) Sparse grid (nested Clenshaw–Curtis + spline).

**(v) One-layer policy function**  
Not implemented; use `growth-nn.ipynb` with one hidden layer if needed.

**(vi) Multilayer policy function**  
`notebooks/growth-nn.ipynb` — multilayer NN for \(V(k)\), HJB residual minimization, \(c(k)\) from FOC.

**(vii) Closed-form in weight optimization**  
`notebooks/growth-nn.ipynb` — loss uses FOC \(c = (V'(k))^{-1/\gamma}\), HJB equation, and model parameters.

---

## Notebooks (short names)

| File | Description |
|------|-------------|
| `notebooks/growth-pfi.ipynb` | Numeric PFI (VFI + Howard) |
| `notebooks/growth-proj.ipynb` | Projection (Chebyshev + sparse grid) |
| `notebooks/growth-nn.ipynb` | Neural network (HJB) |
| `notebooks/growth-compare.ipynb` | Compare all methods on one grid |

## Running the notebooks (higher-density grids)

Grids and training are set for **more precise** numerical solutions:

- **growth-pfi**: capital grid `n_k = 600` (was 200)
- **growth-proj**: Chebyshev degree 12, sparse grid level 6, fine grid 500 points
- **growth-nn**: 2000 epochs, 5000 samples per iteration
- **growth-compare**: same densities; comparison grid 600 points, NN 1500 epochs

Run under the `.venv` environment:

```bash
# Activate venv then run all notebooks (execute and save outputs)
source .venv/bin/activate   # or: .venv\Scripts\activate on Windows
mkdir -p .jupyter
export JUPYTER_CONFIG_DIR=".jupyter" JUPYTER_DATA_DIR=".jupyter/data"
jupyter nbconvert --execute --to notebook --inplace notebooks/growth-pfi.ipynb
jupyter nbconvert --execute --to notebook --inplace notebooks/growth-proj.ipynb
jupyter nbconvert --execute --to notebook --inplace notebooks/growth-nn.ipynb
jupyter nbconvert --execute --to notebook --inplace notebooks/growth-compare.ipynb
```

Or open and run in Jupyter Lab: `jupyter lab notebooks/`.  
**Note:** With `n_k=600`, VFI in growth-pfi and growth-compare can take ~10–15 minutes each.

## Other files

- `closed_form_neoclassical_growth.tex` / `.pdf` — closed-form and Bellman derivations, steady state, and notes on numerical methods.
- `SRL/` — Huggett (1993) model with aggregate risk and SRL/SPG solution (see Appendix details below).

---

## Appendix (SRL): Huggett model implementation details

The following summarizes the calibration, discretization, and algorithmic choices from **Appendix A.1** of the SRL paper for the Huggett application. The notebook `SRL/hugget.ipynb` implements these where applicable.

### Calibration (Table 2)

| Parameter | Symbol | Value | Description |
|-----------|--------|-------|-------------|
| β | beta | 0.96 | Discount factor |
| σ | sigma | 2 (main text); 0.6 in Table 2 | Coefficient of relative risk aversion |
| ρy | rho_y | 0.6 | Autocorrelation of (log) labor income |
| νy | nu_y | 0.2 | Innovation volatility of labor income |
| ρz | rho_z | 0.9 | Persistence of AR(1) for log TFP z |
| νz | nu_z | 0.02 | Volatility of AR(1) for z |
| B | B | 0 | Total bond supply (zero net supply) |
| b | b_min | -1 | Borrowing constraint |
| r̄ | r_bar | 0.038 | Mean interest rate (PE only) |
| ρr | rho_r | 0.8 | Autocorrelation of r (PE only) |
| νr | nu_r | 0.02 | Volatility of r (PE only) |

One period is interpreted as one year. Preferences are isoelastic; idiosyncratic and aggregate income follow log AR(1) processes, discretized via Tauchen.

### Partial equilibrium (PE) specification

In PE, the interest rate is exogenous and Markov. The process is mean-reverting with square-root volatility (CIR-type in discrete time):

- \( r_{t+1} = (1-\rho_r)\bar{r} + \rho_r r_t + \nu_r \sqrt{\max(r_t,0)}\,\varepsilon_t \), \( \varepsilon_t \sim N(0,1) \).

Parameter values are chosen so that the unconditional distribution of r and implied aggregate bond holdings in PE are broadly consistent with GE. For numerical implementation, the PE interest rate is discretized on a grid (Table 3); the paper uses the CIR discretization method of Farmer and Toda (2017) to preserve positivity.

### Discretization (Table 3)

| Parameter | Value | Description |
|-----------|-------|-------------|
| nb | 200 | Number of bond grid points |
| bmax | 50 | Upper bound of bond grid |
| ny | 3 | Number of idiosyncratic income (y) grid points |
| nr | 20 | Number of interest rate grid points |
| rL, rH | 0.01, 0.06 | Bounds of r grid |
| nz | 30 | Number of aggregate TFP (z) grid points |

Individual state (b, y): bonds on a 1D grid with nb points and upper bound bmax; income y takes ny values. Aggregate side: r on [rL, rH] with nr points; z discretized with nz points using a standard Tauchen procedure. These choices balance accuracy and cost.

### Simulation horizon and truncation

- Lifetime utility is approximated by truncating the infinite sum at horizon \( T_{\text{trunc}} \) with tolerance \( e_{\text{trunc}} \): choose \( T_{\text{trunc}} = \min\{T : \beta^T < e_{\text{trunc}}\} \).
- Baseline: \( e_{\text{trunc}} = 10^{-3} \), giving \( T_{\text{trunc}} = 170 \) (contribution of periods beyond that is bounded by \( 10^{-3} \) in present value).
- Minimum consumption floor \( c_{\min} = 10^{-3} \) to avoid evaluating utility at (or very close to) zero.

### Training schedule and learning rate (SPG)

- Exponentially decaying learning rate: \( \ell_t = \ell_{\text{ini}} \cdot \ell_{\text{decay}}^{t_0} \), where \( t_0 = \max(t - N_{\text{warm-up}}, 0) / (N_{\text{epoch}} - N_{\text{warm-up}}) \), so the rate is constant during warm-up and then decays.
- Huggett: \( N_{\text{epoch}} = 1000 \), \( N_{\text{warm-up}} = 50 \), \( \ell_{\text{ini}} = 10^{-3} \), \( \ell_{\text{decay}} = 0.5 \), exponential scheduler.
- Convergence: when the change in policy parameters across epochs falls below \( e_{\text{converge}} = 3\times 10^{-4} \).

### Sampling, batching, and memory

- Data are used in minibatches. Per update: effective data size = \( N_{\text{sample}} \times N_{\text{update}} \) (number of trajectories × time steps per trajectory).
- Baseline Huggett: \( N_{\text{sample}} = 512 \) trajectories per batch. Mini-batching keeps memory manageable while giving enough variation for stable gradient estimates.

### Initialization and warm-up

- Policy is initialized so that the initial aggregate savings schedule is at least weakly responsive to the interest rate (see paper Footnote 20).
- **Warm-up** (first \( N_{\text{warm-up}} \) epochs): cross-sectional distribution g is fixed at a simple initial guess \( g_0 \) and not updated; only the policy is updated to move away from the initial guess.
- **After warm-up**: distribution is updated endogenously (e.g. \( G_{t+1} = \texttt{update\_G\_pi\_direct}(\theta, G_t, \ldots) \)); initial \( G_0 \) for each trajectory can be uniform or the steady-state under the current policy.
- **Steady-state \( g_0 \) for warm-up:** \( g_0 \) is computed as the invariant distribution of the transition at a fixed (z, r) (e.g. grid midpoints) under the **initial** policy θ via power iteration (\( g \leftarrow A^\top g \) normalized); that \( g_0 \) is then fixed for all trajectories during the first \( N_{\text{warm-up}} \) epochs.

### Policy output and distribution update (implementation)

- **Policy returns raw (c, b′):** The policy function \( \pi(b,y,r,z) \to (c, b') \) returns **continuous** consumption and next-period bonds (e.g. \( b' = (1+r)b + yz - c \) clipped to \( [\underline{b}, b_{\max}] \)). No lottery or rounding is applied inside the policy; this keeps the policy deterministic and simplifies market clearing (e.g. aggregate saving uses \( G \cdot b' \) directly).
- **Lottery only in transition:** When the distribution \( g \) must be updated on a **discrete** b grid, we use either:
  - **Young (lottery) weights:** For each state \( j \), \( b'_j \) is split between the two neighbouring grid points with weights \( (1-\lambda, \lambda) \); mass is scattered via \( \texttt{np.add.at} \) so that \( g_{t+1} \) is obtained **without building** the full transition matrix \( A_\pi \). The helper \( \texttt{b\_next\_to\_grid\_lottery} \) is used only when a discrete \( b' \) is needed (e.g. agent-level simulation).
  - **Soft weights (SPG):** In the differentiable training path, \( w_b[j, ib] \propto \exp(-(b'_j - b_{\text{grid}}[ib])^2/(2\sigma^2)) \) (normalized) so that \( b'_j \) distributes mass smoothly over the grid (differentiable in θ).
- **Direct \( g_t \to g_{t+1} \) without building A:** The notebook implements \( \texttt{update\_G\_direct} \) (NumPy) and \( \texttt{update\_G\_pi\_direct} \) (PyTorch) so that \( g_{t+1} \) is computed from \( g_t \) and the policy by scattering mass (Young or soft weights) and applying \( T_y \), **without** forming the \( J\times J \) matrix \( A_\pi \). This reduces memory and matches the “map \( G_t \) to \( G_{t+1} \) via \( \pi \)” design.

### Mapping continuous b′ to the discrete b grid (updating g)

The policy outputs a **continuous** next-period bond \( b' = (1+r)b + yz - c \), while the distribution g is defined on a **discrete** b grid. To update g, the mass at each continuous \( b' \) must be assigned to grid points.

**Soft weights (used in SPG code):** For each state j with implied \( b'_j \), compute distances to all grid points \( b_{\text{grid}}[ib] \), then Gaussian weights and normalize:

- \( \text{dist}[j, ib] = b'_j - b_{\text{grid}}[ib] \)
- \( w_b[j, ib] \propto \exp(-\text{dist}^2 / (2\sigma^2)) \), normalized so \( \sum_{ib} w_b[j, ib] = 1 \).

So \( b'_j \) distributes probability mass smoothly over the grid (differentiable in θ). The transition matrix entry from \( j = (ib, iy) \) to \( j' = (ib', iy') \) would be \( A_\pi[j', j] = w_b[j, ib'] \times T_y[iy, iy'] \); in practice we compute \( g_{t+1} \) directly via \( \texttt{update\_G\_pi\_direct} \) without forming \( A_\pi \).

**Young (two-point) weights (used in NumPy path):** If \( b' \in [b_i, b_{i+1}] \), assign mass \( (1-\lambda) \) to \( b_i \) and \( \lambda \) to \( b_{i+1} \) with \( \lambda = (b' - b_i)/(b_{i+1}-b_i) \). Implemented in \( \texttt{update\_G\_direct} \) and optionally \( \texttt{b\_next\_to\_grid\_lottery} \) when discrete \( b' \) is required.

---

## Appendix (SRL): Krusell–Smith notebook

The notebook **SRL/krusell_smith.ipynb** implements the Krusell–Smith (1998) model with the same implementation conventions as Huggett where applicable:

- **Calibration:** SRL Section 4.2 and Appendix A.2 (Tables 4–5): \( \beta = 0.95 \), \( \sigma = 3 \), production \( \alpha, \delta \), capital grid, price grids (r, w), Tauchen for y and z.
- **Policy:** Raw \( (c, n, b') \) from the consumption grid θ; labour \( n = 1 \) in the SPG grid policy. \( \texttt{policy\_from\_grid\_ks} \) returns continuous \( (c, n, b') \); \( \texttt{b\_next\_to\_grid\_lottery} \) is used only when discrete \( b' \) is needed.
- **Distribution update:** Direct \( d_{t+1} \) from \( d_t \) via \( \texttt{update\_d\_direct} \) (NumPy, Young weights) and \( \texttt{update\_d\_pi\_direct\_ks} \) (PyTorch, soft weights) **without** building the full transition matrix. Aggregate capital \( K = d \cdot b \); prices \( (r, w) = \texttt{K\_to\_prices}(K, z) \).
- **Two-phase training:** Warm-up with fixed \( d_0 \) (steady-state under initial θ at mid z, r, w); after \( N_{\text{warm-up}} \), \( d \) evolves each period. SPG hyperparameters (e.g. \( N_{\text{epoch}}, N_{\text{warm-up}}, \ell_{\text{ini}}, \ell_{\text{decay}}, N_{\text{sample}}, e_{\text{converge}} \)) are defined in the calibration cell and used in the training loop; \( \texttt{steady\_state\_d0\_ks} \) provides the initial \( d_0 \) for warm-up.
- **Using trained policy:** A “Using Trained Policy” cell wraps usage in try/except so that the notebook does not assume `theta` exists before training is run.

### Speed optimizations (Huggett and Krusell–Smith)

- **Market clearing (Huggett):** `P_star_detach` is **vectorized**: one forward pass computes aggregate saving \( S(\texttt{ir}) \) for all interest-rate grid points, then picks the index that minimizes \( |S - B| \). This avoids multiple Python calls and `.item()` in a bisection loop. A bisection variant `P_star_detach_bisection` remains available if needed (e.g. very large `nr` and memory-bound).
- **Distribution update (Krusell–Smith):** `update_d_pi_direct_ks` uses the same **Q @ Ty** pattern as Huggett’s `update_G_pi_direct`: no loop over \( J \), no full \( J\times J \) matrix; steady-state \( d_0 \) is computed by iterating `update_d_pi_direct_ks` instead of building \( A_\pi \).
- **Steady-state G0/d0:** Warm-up initial distribution is computed by iterating the direct update; **early stopping** (e.g. when \( \max|G_{new} - G| < 10^{-6} \)) and a modest iteration cap (e.g. 150) reduce cost.
- **Further options:** Use a coarser SPG grid (`nb_spg`, `nr_spg`, `nz_spg`) or shorter `T_horizon` for faster runs; on CPU, `torch.set_num_threads(n)` can help; PyTorch 2’s `torch.compile` can be tried on the objective or update step (profile first).
