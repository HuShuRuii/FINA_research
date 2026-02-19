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

## Other files

- `closed_form_neoclassical_growth.tex` / `.pdf` — closed-form and Bellman derivations, steady state, and notes on numerical methods.
