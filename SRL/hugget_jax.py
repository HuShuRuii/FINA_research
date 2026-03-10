#!/usr/bin/env python3
"""
Huggett (1993) SRL/SPG — JAX implementation.

Same economic setup as hugget.ipynb / run_hugget_cluster.py: policy π(b,y,r,z)→(c,b'),
r = P*(G,z) with gradient stopped. Implemented in JAX for potential speedup via JIT
and XLA compilation (CPU/GPU/TPU).

Dependencies: numpy, scipy, jax, jaxlib, optax
  pip install numpy scipy jax jaxlib optax

Usage:
  python hugget_jax.py [--epochs N] [--quick] [--out_dir DIR]
  python hugget_jax.py --benchmark   # run short train and print timing

Output (default: hugget_output/):
  - loss_curve.png           # L(θ) vs epoch
  - consumption_policy_grid.png  # c(b,y,r,z) for various r,z
  - consumption_vs_r.png     # c vs r given fixed b,y,z
  - c_grid.npy, *_grid.npy   # full policy grid; loss_hist.txt
"""
from __future__ import annotations

import argparse
import os
import time
from typing import Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

import jax
import jax.numpy as jnp
from jax import lax

import optax


# ---------- Calibration (SRL Table 2 & 3) ----------
def get_calibration(quick: bool = False) -> dict:
    beta, sigma = 0.96, 2.0
    rho_y, nu_y = 0.6, 0.2
    rho_z, nu_z = 0.9, 0.02
    B, b_min = 0.0, -1.0
    nb, b_max = 200, 50.0
    ny, nr, nz = 3, 20, 30
    r_min, r_max = 0.01, 0.06
    c_min = 1e-3
    T_trunc = 170
    e_trunc = 1e-3
    if quick:
        N_epoch, N_warmup = 30, 5
        N_sample, e_converge = 64, 1e-3
    else:
        N_epoch, N_warmup = 1000, 50
        N_sample, e_converge = 512, 3e-4
    lr_ini, lr_decay = 1e-3, 0.5
    return {
        "beta": beta, "sigma": sigma, "rho_y": rho_y, "nu_y": nu_y,
        "rho_z": rho_z, "nu_z": nu_z, "B": B, "b_min": b_min,
        "nb": nb, "b_max": b_max, "ny": ny, "nr": nr, "nz": nz,
        "r_min": r_min, "r_max": r_max, "c_min": c_min,
        "T_trunc": T_trunc, "e_trunc": e_trunc,
        "N_epoch": N_epoch, "N_warmup": N_warmup, "lr_ini": lr_ini, "lr_decay": lr_decay,
        "N_sample": N_sample, "e_converge": e_converge,
    }


# ---------- Tauchen (NumPy, then convert to JAX arrays) ----------
def tauchen_ar1(rho: float, sigma_innov: float, n_states: int, m: float = 3.0, mean: float = 0.0):
    std = sigma_innov / np.sqrt(1 - rho**2)
    if mean == 0:
        x_min, x_max = -m * std, m * std
    else:
        x_min = max(1e-6, mean - m * std)
        x_max = mean + m * std
    x_grid = np.linspace(x_min, x_max, n_states)
    step = (x_max - x_min) / (n_states - 1) if n_states > 1 else 1.0
    mu_i = (1 - rho) * mean + rho * x_grid
    z_lo = (x_grid - mu_i[:, None] + step / 2) / sigma_innov
    z_hi = (x_grid - mu_i[:, None] - step / 2) / sigma_innov
    P = np.zeros((n_states, n_states))
    P[:, 0] = norm.cdf(z_lo[:, 0])
    P[:, -1] = 1 - norm.cdf(z_hi[:, -1])
    if n_states > 2:
        P[:, 1:-1] = norm.cdf(z_lo[:, 1:-1]) - norm.cdf(z_hi[:, 1:-1])
    P = P / P.sum(axis=1, keepdims=True)
    return x_grid, P


# ---------- JAX helpers ----------
def theta_to_c(theta: jnp.ndarray, c_min_val: float) -> jnp.ndarray:
    """θ -> c = clamp(θ, c_min)."""
    return jnp.maximum(theta, c_min_val)


def u_jax(c: jnp.ndarray, sig: float, c_min_val: float) -> jnp.ndarray:
    """CRRA u(c); clamp c for numerical safety."""
    c = jnp.maximum(c, c_min_val)
    if abs(sig - 1.0) < 1e-8:
        return jnp.log(c)
    return (c ** (1 - sig)) / (1 - sig)


def init_theta(
    b_grid: jnp.ndarray,
    y_grid: jnp.ndarray,
    z_grid: jnp.ndarray,
    r_grid: jnp.ndarray,
    save_frac: float = 0.2,
    c_min_val: float = 1e-3,
) -> jnp.ndarray:
    """Initial consumption grid (used directly as theta)."""
    nb_t, ny_t = b_grid.shape[0], y_grid.shape[0]
    nz_t, nr_t = z_grid.shape[0], r_grid.shape[0]
    # b_flat (nb*ny,), y_flat (nb*ny,)
    b_flat = jnp.repeat(b_grid, ny_t)
    y_flat = jnp.tile(y_grid, nb_t)
    # cash (nz, nr, nb*ny) -> (nz, nr, nb, ny)
    cash = (
        b_flat[None, None, :] * (1 + r_grid[None, :, None])
        + y_flat[None, None, :] * z_grid[:, None, None]
    )
    cash = cash.reshape(nz_t, nr_t, nb_t, ny_t)
    return jnp.maximum((1 - save_frac) * cash, c_min_val)


def _G_to_mat(G: jnp.ndarray, nb: int, ny: int) -> jnp.ndarray:
    """Ensure G is (nb, ny)."""
    return jnp.reshape(G, (nb, ny))


def update_G_pi_direct(
    theta: jnp.ndarray,
    G: jnp.ndarray,
    iz: int,
    ir: int,
    b_grid: jnp.ndarray,
    y_grid: jnp.ndarray,
    z_grid: jnp.ndarray,
    r_grid: jnp.ndarray,
    Ty: jnp.ndarray,
    nb: int,
    ny: int,
    b_min: float,
    b_max: float,
    c_min_val: float,
) -> jnp.ndarray:
    """One-step G update. b' from budget; lottery to adjacent b grid points."""
    G = _G_to_mat(G, nb, ny)
    c = theta_to_c(theta, c_min_val)  # (nz, nr, nb, ny)
    c_val = c[iz, ir, :, :].ravel()
    b_next = (
        (1 + r_grid[ir]) * jnp.repeat(b_grid, ny)
        + jnp.tile(y_grid, nb) * z_grid[iz]
        - c_val
    )
    b_next = jnp.clip(b_next, b_min, b_max)
    # Lottery: linear weights to two adjacent grid points
    idx_hi = jnp.clip(
        jnp.searchsorted(b_grid, b_next, side="right"), 1, nb - 1
    )
    idx_lo = idx_hi - 1
    denom = jnp.maximum(b_grid[idx_hi] - b_grid[idx_lo], 1e-20)
    w_hi = (b_next - b_grid[idx_lo]) / denom
    w_lo = 1.0 - w_hi
    eye = jnp.eye(nb)
    w_b = w_lo[:, None] * eye[idx_lo] + w_hi[:, None] * eye[idx_hi]
    M = jnp.transpose(w_b.reshape(nb, ny, nb), (2, 0, 1))
    Q = (M * G[None, :, :]).sum(axis=1)
    G_new = Q @ Ty
    return G_new / (G_new.sum() + 1e-20)


def update_G_from_c_and_r(
    c_t: jnp.ndarray,
    r_star: jnp.ndarray,
    G: jnp.ndarray,
    b_grid: jnp.ndarray,
    y_grid: jnp.ndarray,
    z_grid: jnp.ndarray,
    iz: int,
    Ty: jnp.ndarray,
    nb: int,
    ny: int,
    b_min: float,
    b_max: float,
) -> jnp.ndarray:
    """One-step G update using an interpolated c_t and scalar r_star."""
    G = _G_to_mat(G, nb, ny)
    b_next = (
        (1 + r_star) * jnp.repeat(b_grid, ny)
        + jnp.tile(y_grid, nb) * z_grid[iz]
        - c_t.ravel()
    )
    b_next = jnp.clip(b_next, b_min, b_max)
    idx_hi = jnp.clip(jnp.searchsorted(b_grid, b_next, side="right"), 1, nb - 1)
    idx_lo = idx_hi - 1
    denom = jnp.maximum(b_grid[idx_hi] - b_grid[idx_lo], 1e-20)
    w_hi = (b_next - b_grid[idx_lo]) / denom
    w_lo = 1.0 - w_hi
    eye = jnp.eye(nb)
    w_b = w_lo[:, None] * eye[idx_lo] + w_hi[:, None] * eye[idx_hi]
    M = jnp.transpose(w_b.reshape(nb, ny, nb), (2, 0, 1))
    Q = (M * G[None, :, :]).sum(axis=1)
    G_new = Q @ Ty
    return G_new / (G_new.sum() + 1e-20)


def P_star_bracket(
    theta: jnp.ndarray,
    G: jnp.ndarray,
    iz: int,
    b_grid: jnp.ndarray,
    y_grid: jnp.ndarray,
    z_grid: jnp.ndarray,
    r_grid: jnp.ndarray,
    ny: int,
    B: float,
    b_min: float,
    b_max: float,
    c_min_val: float,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Market clearing by bracket interpolation on r-grid.
    Returns (r_star, ir_lo, ir_hi, w_r)."""
    nr = r_grid.shape[0]
    if nr == 1:
        z0 = jnp.int32(0)
        return r_grid[0], z0, z0, jnp.float32(0.0)
    nb_b = b_grid.shape[0]
    G_mat = _G_to_mat(G, nb_b, ny)
    z_val = z_grid[iz]
    c = theta_to_c(theta, c_min_val)  # (nz, nr, nb, ny)
    # c_all (nb*ny, nr)
    c_all = c[iz, :, :, :].transpose(1, 2, 0).reshape(nb_b * ny, nr)
    b_flat = jnp.repeat(b_grid, ny)
    y_flat = jnp.tile(y_grid, nb_b)
    resources = b_flat[:, None] * (1 + r_grid[None, :]) + (y_flat * z_val)[:, None]
    b_next_all = jnp.clip(resources - c_all, b_min, b_max)
    b_next_all = b_next_all.reshape(nb_b, ny, nr)
    S_all = (G_mat[:, :, None] * b_next_all).sum(axis=(0, 1))
    ge = S_all >= B
    has_ge = jnp.any(ge)
    first_ge = jnp.argmax(ge)
    ir_hi_raw = jnp.where(has_ge, first_ge, nr - 1)
    ir_hi = jnp.clip(ir_hi_raw, 1, nr - 1)
    ir_lo = jnp.clip(ir_hi - 1, 0, nr - 2)
    S_lo = S_all[ir_lo]
    S_hi = S_all[ir_hi]
    w_r = jnp.clip((B - S_lo) / jnp.maximum(S_hi - S_lo, 1e-20), 0.0, 1.0)
    r_star = r_grid[ir_lo] + w_r * (r_grid[ir_hi] - r_grid[ir_lo])
    return r_star, ir_lo, ir_hi, w_r


def market_clearing_stats(
    theta: jnp.ndarray,
    G: jnp.ndarray,
    iz: int,
    b_grid: jnp.ndarray,
    y_grid: jnp.ndarray,
    z_grid: jnp.ndarray,
    r_grid: jnp.ndarray,
    ny: int,
    B: float,
    b_min: float,
    b_max: float,
    c_min_val: float,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return (r_star, ir_lo, ir_hi, w_r, residual) at current (G, z)."""
    nr = r_grid.shape[0]
    nb_b = b_grid.shape[0]
    G_mat = _G_to_mat(G, nb_b, ny)
    z_val = z_grid[iz]
    c = theta_to_c(theta, c_min_val)
    c_all = c[iz, :, :, :].transpose(1, 2, 0).reshape(nb_b * ny, nr)
    b_flat = jnp.repeat(b_grid, ny)
    y_flat = jnp.tile(y_grid, nb_b)
    resources = b_flat[:, None] * (1 + r_grid[None, :]) + (y_flat * z_val)[:, None]
    b_next_all = jnp.clip(resources - c_all, b_min, b_max).reshape(nb_b, ny, nr)
    S_all = (G_mat[:, :, None] * b_next_all).sum(axis=(0, 1))
    ge = S_all >= B
    has_ge = jnp.any(ge)
    first_ge = jnp.argmax(ge)
    ir_hi_raw = jnp.where(has_ge, first_ge, nr - 1)
    ir_hi = jnp.clip(ir_hi_raw, 1, nr - 1)
    ir_lo = jnp.clip(ir_hi - 1, 0, nr - 2)
    S_lo = S_all[ir_lo]
    S_hi = S_all[ir_hi]
    w_r = jnp.clip((B - S_lo) / jnp.maximum(S_hi - S_lo, 1e-20), 0.0, 1.0)
    r_star = r_grid[ir_lo] + w_r * (r_grid[ir_hi] - r_grid[ir_lo])
    S_star = S_lo + w_r * (S_hi - S_lo)
    residual = S_star - B
    return r_star, ir_lo, ir_hi, w_r, residual


def simulate_diagnostics_path(
    theta: jnp.ndarray,
    G0: jnp.ndarray,
    key: jnp.ndarray,
    T_diag: int,
    b_grid: jnp.ndarray,
    y_grid: jnp.ndarray,
    z_grid: jnp.ndarray,
    r_grid: jnp.ndarray,
    Ty: jnp.ndarray,
    Tz: jnp.ndarray,
    nb: int,
    ny: int,
    B: float,
    b_min: float,
    b_max: float,
    c_min_val: float,
) -> dict:
    """Simulate one path and return diagnostics for debugging/visualization."""
    G = _G_to_mat(G0, nb, ny)
    iz = int(z_grid.shape[0] // 2)
    r_path = np.zeros(T_diag, dtype=float)
    z_path = np.zeros(T_diag, dtype=int)
    residual_path = np.zeros(T_diag, dtype=float)
    mean_b_path = np.zeros(T_diag, dtype=float)

    b_col = np.array(b_grid)[:, None]
    for t in range(T_diag):
        r_star, ir_lo, ir_hi, w_r, residual = market_clearing_stats(
            theta, G, iz, b_grid, y_grid, z_grid, r_grid, ny, B, b_min, b_max, c_min_val
        )
        c = theta_to_c(theta, c_min_val)
        c_iz = c[iz, :, :, :]
        c_t = (1 - w_r) * c_iz[ir_lo, :, :] + w_r * c_iz[ir_hi, :, :]
        G = update_G_from_c_and_r(c_t, r_star, G, b_grid, y_grid, z_grid, iz, Ty, nb, ny, b_min, b_max)

        r_path[t] = float(r_star)
        z_path[t] = int(iz)
        residual_path[t] = float(residual)
        mean_b_path[t] = float((np.array(G) * b_col).sum())

        key, subkey = jax.random.split(key)
        iz = int(jax.random.choice(subkey, z_grid.shape[0], p=Tz[iz, :]))

    return {
        "r_path": r_path,
        "z_idx_path": z_path,
        "z_path": np.array(z_grid)[z_path],
        "residual_path": residual_path,
        "mean_b_path": mean_b_path,
    }


def one_step_trajectory(
    carry: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, int, float],
    _t: int,
    static: dict,
) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, int, float], jnp.ndarray]:
    """Single time step: (G, iz, L_n) -> (G_new, iz_new, L_n + term).
    static: theta, b_grid, y_grid, z_grid, r_grid, Ty, Tz, nb, ny, nz_spg, nr_spg,
            beta, e_trunc, b_min, b_max, c_min, sigma, warm_up.
    """
    G, iz, L_n, key, beta_t = carry
    theta = static["theta"]
    b_grid = static["b_grid"]
    y_grid = static["y_grid"]
    z_grid = static["z_grid"]
    r_grid = static["r_grid"]
    Ty = static["Ty"]
    Tz = static["Tz"]
    nb = static["nb"]
    ny = static["ny"]
    nz_spg = static["nz_spg"]
    _nr_spg = static["nr_spg"]
    beta = static["beta"]
    e_trunc = static["e_trunc"]
    b_min = static["b_min"]
    b_max = static["b_max"]
    c_min_val = static["c_min"]
    sigma = static["sigma"]
    warm_up = static["warm_up"]

    t = _t
    weight = (beta ** t) * jnp.float32((beta ** t) >= e_trunc)

    # Market-clearing: bracket interpolation on r-grid (no gradient through prices)
    r_star, ir_lo, ir_hi, w_r = P_star_bracket(
        theta, G, iz, b_grid, y_grid, z_grid, r_grid, ny, static["B"],
        b_min, b_max, c_min_val,
    )
    r_star = lax.stop_gradient(r_star)
    ir_lo = lax.stop_gradient(ir_lo)
    ir_hi = lax.stop_gradient(ir_hi)
    w_r = lax.stop_gradient(w_r)

    c = theta_to_c(theta, c_min_val)
    c_iz = c[iz, :, :, :]
    c_t = (1 - w_r) * c_iz[ir_lo, :, :] + w_r * c_iz[ir_hi, :, :]
    G_stopped = lax.stop_gradient(G)
    term = weight * (G_stopped * u_jax(c_t, sigma, c_min_val)).sum()
    L_n_new = L_n + term

    if warm_up:
        G_new = G
    else:
        G_new = update_G_from_c_and_r(
            c_t, r_star, lax.stop_gradient(G),
            b_grid, y_grid, z_grid, iz, Ty, nb, ny,
            b_min, b_max,
        )
    G_new = lax.stop_gradient(G_new)

    key, subkey = jax.random.split(key)
    iz_new = jax.random.choice(subkey, nz_spg, p=Tz[iz, :])

    new_beta_t = beta_t * beta
    return (G_new, iz_new, L_n_new, key, new_beta_t), term


def spg_objective_single_traj(
    key: jnp.ndarray,
    theta: jnp.ndarray,
    G0: jnp.ndarray,
    iz0: jnp.ndarray,  # scalar (e.g. from vmap)
    T_horizon: int,
    b_grid: jnp.ndarray,
    y_grid: jnp.ndarray,
    z_grid: jnp.ndarray,
    r_grid: jnp.ndarray,
    Ty: jnp.ndarray,
    Tz: jnp.ndarray,
    nb: int,
    ny: int,
    nz_spg: int,
    nr_spg: int,
    beta: float,
    e_trunc: float,
    B: float,
    b_min: float,
    b_max: float,
    c_min_val: float,
    sigma: float,
    warm_up: bool,
) -> jnp.ndarray:
    """Single trajectory L_n and final distribution G_T."""
    static = {
        "theta": theta,
        "b_grid": b_grid,
        "y_grid": y_grid,
        "z_grid": z_grid,
        "r_grid": r_grid,
        "Ty": Ty,
        "Tz": Tz,
        "nb": nb,
        "ny": ny,
        "nz_spg": nz_spg,
        "nr_spg": nr_spg,
        "beta": beta,
        "e_trunc": e_trunc,
        "B": B,
        "b_min": b_min,
        "b_max": b_max,
        "c_min": c_min_val,
        "sigma": sigma,
        "warm_up": warm_up,
    }
    G = _G_to_mat(G0, nb, ny)
    L_n = jnp.float32(0.0)
    beta_t = jnp.float32(1.0)
    iz = jnp.int32(iz0) if iz0.shape == () else iz0
    carry = (G, iz, L_n, key, beta_t)
    (G_final, iz_final, L_n_final, key_final, _), L_terms = jax.lax.scan(
        lambda c, t: one_step_trajectory(c, t, static),
        carry,
        jnp.arange(T_horizon),
    )
    return L_terms.sum(), G_final.reshape(-1)


def spg_objective(
    theta: jnp.ndarray,
    key: jnp.ndarray,
    N_traj: int,
    T_horizon: int,
    G0: jnp.ndarray,
    b_grid: jnp.ndarray,
    y_grid: jnp.ndarray,
    z_grid: jnp.ndarray,
    r_grid: jnp.ndarray,
    Ty: jnp.ndarray,
    Tz: jnp.ndarray,
    nb: int,
    ny: int,
    nz_spg: int,
    nr_spg: int,
    beta: float,
    e_trunc: float,
    B: float,
    b_min: float,
    b_max: float,
    c_min_val: float,
    sigma: float,
    warm_up: bool,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Return (mean objective over trajectories, mean final distribution)."""
    key, k1, k2 = jax.random.split(key, 3)
    keys = jax.random.split(k1, N_traj)
    iz0s = jax.random.randint(k2, (N_traj,), 0, nz_spg)

    def body(key_i, iz0):
        return spg_objective_single_traj(
            key_i, theta, G0, iz0, T_horizon,
            b_grid, y_grid, z_grid, r_grid, Ty, Tz,
            nb, ny, nz_spg, nr_spg, beta, e_trunc, B, b_min, b_max, c_min_val, sigma, warm_up,
        )

    L_list, G_final_list = jax.vmap(body)(keys, iz0s)
    return jnp.mean(L_list), jnp.mean(G_final_list, axis=0)


def steady_state_G0(
    theta: jnp.ndarray,
    b_grid: jnp.ndarray,
    y_grid: jnp.ndarray,
    z_grid: jnp.ndarray,
    r_grid: jnp.ndarray,
    Ty: jnp.ndarray,
    nb: int,
    ny: int,
    nz_spg: int,
    nr_spg: int,
    b_min: float,
    b_max: float,
    c_min_val: float,
    n_iter: int = 150,
    tol: float = 1e-6,
) -> jnp.ndarray:
    """Steady-state G at mid (z, r); no grad, used as warm-up G0."""
    iz_mid = nz_spg // 2
    ir_mid = nr_spg // 2 if nr_spg > 0 else 0
    G = jnp.ones((nb, ny)) / (nb * ny)

    def cond(carry):
        G_cur, G_prev, it = carry
        return (it < n_iter) & (jnp.abs(G_cur - G_prev).max() >= tol)

    def body(carry):
        G_cur, _G_prev, it = carry
        G_new = update_G_pi_direct(
            theta, G_cur, iz_mid, ir_mid,
            b_grid, y_grid, z_grid, r_grid, Ty, nb, ny,
            b_min, b_max, c_min_val,
        )
        return (G_new, G_cur, it + 1)

    G_final, _, _ = jax.lax.while_loop(
        cond,
        body,
        (G, G + 1.0, 0),
    )
    return G_final.reshape(-1)


def policy_from_grid(b, iy, iz, ir_or_r, c_grid, b_grid, y_grid, z_grid, r_grid, c_min_val=1e-3):
    """b: continuous (lottery). ir_or_r: int = grid index; float = r value (linear interp in r).
    c_grid shape (nb, ny, nz, nr). Returns (c, b_next)."""
    b = np.atleast_1d(np.asarray(b, dtype=float))
    nb, nr = len(b_grid), len(r_grid)
    if isinstance(ir_or_r, (int, np.integer)):
        ir_lo = int(np.clip(ir_or_r, 0, nr - 1))
        ir_hi = ir_lo
        w_r = 0.0
        r_use = float(r_grid[ir_lo])
    else:
        r_val = float(ir_or_r)
        r_use = r_val
        ir_lo = int(np.clip(np.searchsorted(r_grid, r_val, side="right") - 1, 0, nr - 2))
        ir_hi = ir_lo + 1
        w_r = (r_val - r_grid[ir_lo]) / max(r_grid[ir_hi] - r_grid[ir_lo], 1e-20)
    b_c = np.clip(b, b_grid[0], b_grid[-1])
    j_hi = np.clip(np.searchsorted(b_grid, b_c), 1, nb - 1)
    j_lo = j_hi - 1
    w_b = (b_c - b_grid[j_lo]) / np.maximum(b_grid[j_hi] - b_grid[j_lo], 1e-20)
    c = (1 - w_r) * ((1 - w_b) * c_grid[j_lo, iy, iz, ir_lo] + w_b * c_grid[j_hi, iy, iz, ir_lo])
    c += w_r * ((1 - w_b) * c_grid[j_lo, iy, iz, ir_hi] + w_b * c_grid[j_hi, iy, iz, ir_hi])
    c = np.maximum(c, c_min_val)
    c_total = (1 + r_use) * b + y_grid[iy] * z_grid[iz]
    b_next = np.clip(c_total - c, b_grid[0], b_grid[-1])
    c = np.maximum(c_total - b_next, c_min_val)
    if c.size == 1:
        return float(c.ravel()[0]), float(b_next.ravel()[0])
    return c, b_next


def main():
    parser = argparse.ArgumentParser(description="Huggett SRL/SPG in JAX")
    parser.add_argument("--epochs", type=int, default=None, help="Max epochs")
    parser.add_argument("--quick", action="store_true", help="Short run")
    parser.add_argument("--out_dir", type=str, default="hugget_output", help="Output dir for figures and grids")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--benchmark", action="store_true", help="Run short train and print timing")
    parser.add_argument("--n_sample", type=int, default=None, help="Override number of trajectories per epoch")
    parser.add_argument("--n_warmup", type=int, default=None, help="Override warm-up epochs")
    parser.add_argument("--lr_ini", type=float, default=None, help="Override initial learning rate")
    parser.add_argument("--lr_decay", type=float, default=None, help="Override lr decay factor")
    parser.add_argument("--log_every", type=int, default=10, help="Print progress every N epochs")
    parser.add_argument("--diag_steps", type=int, default=200, help="Length of post-training diagnostic simulation path")
    args = parser.parse_args()

    quick = args.quick or args.benchmark
    if args.benchmark:
        args.epochs = args.epochs or 20
    cal = get_calibration(quick=quick)
    if args.epochs is not None:
        cal["N_epoch"] = args.epochs
    if args.n_sample is not None:
        cal["N_sample"] = args.n_sample
    if args.n_warmup is not None:
        cal["N_warmup"] = args.n_warmup
    if args.lr_ini is not None:
        cal["lr_ini"] = args.lr_ini
    if args.lr_decay is not None:
        cal["lr_decay"] = args.lr_decay

    beta = cal["beta"]
    sigma = cal["sigma"]
    c_min = cal["c_min"]
    b_min = cal["b_min"]
    b_max = cal["b_max"]
    ny = cal["ny"]
    nb, nr, nz = cal["nb"], cal["nr"], cal["nz"]
    r_min, r_max = cal["r_min"], cal["r_max"]
    B = cal["B"]

    # Grids (NumPy for Tauchen)
    np.random.seed(args.seed)
    b_grid_np = np.linspace(b_min, b_max, nb)
    r_grid_np = np.linspace(r_min, r_max, nr)
    y_grid_np, Ty_np = tauchen_ar1(cal["rho_y"], cal["nu_y"], ny, m=3, mean=1.0)
    invariant_y = np.linalg.matrix_power(Ty_np.T, 200)[:, 0]
    y_grid_np = y_grid_np / (y_grid_np @ invariant_y)
    log_z_grid, Tz_np = tauchen_ar1(cal["rho_z"], cal["nu_z"], nz)
    z_grid_np = np.exp(log_z_grid)
    invariant_z = np.linalg.matrix_power(Tz_np.T, 200)[:, 0]
    z_grid_np = z_grid_np / (z_grid_np @ invariant_z)

    # JAX arrays: use full paper grids for training/reproduction
    nb_spg, nr_spg, nz_spg = nb, nr, nz
    b_grid = jnp.array(b_grid_np)
    r_grid = jnp.array(r_grid_np)
    z_grid = jnp.array(z_grid_np)
    y_grid = jnp.array(y_grid_np)
    Ty = jnp.array(Ty_np)
    Tz = jnp.array(Tz_np)
    nz_spg = Tz.shape[0]
    nr_spg = r_grid.shape[0]
    J = nb_spg * ny

    key = jax.random.PRNGKey(args.seed)
    key_init, key_train = jax.random.split(key)

    theta = init_theta(b_grid, y_grid, z_grid, r_grid, save_frac=0.2, c_min_val=c_min)
    G0_steady = steady_state_G0(
        theta, b_grid, y_grid, z_grid, r_grid, Ty,
        nb_spg, ny, nz_spg, nr_spg, b_min, b_max, c_min,
    )
    T_horizon = cal["T_trunc"]

    # Objective and gradient
    def objective_fn(theta, key, G0, warm_up):
        return spg_objective(
            theta, key, cal["N_sample"], T_horizon, G0,
            b_grid, y_grid, z_grid, r_grid, Ty, Tz,
            nb_spg, ny, nz_spg, nr_spg, beta, cal["e_trunc"], B,
            b_min, b_max, c_min, sigma, warm_up,
        )

    def loss_with_aux(theta, key, G0, warm_up):
        L_val, G_end_mean = objective_fn(theta, key, G0, warm_up)
        return -L_val, G_end_mean

    value_and_grad_fn = jax.jit(
        jax.value_and_grad(loss_with_aux, argnums=0, has_aux=True),
        static_argnums=(3,),
    )

    # Optimizer: optax Adam with custom lr schedule (warm-up flat, then exponential decay)
    def lr_schedule(step):
        denom = max(cal["N_epoch"] - cal["N_warmup"], 1)
        t0 = jnp.maximum(step - cal["N_warmup"], 0) / denom
        return cal["lr_ini"] * (cal["lr_decay"] ** t0)

    optimizer = optax.adam(learning_rate=lr_schedule)
    opt_state = optimizer.init(theta)
    loss_hist = []

    print("Huggett JAX: nb=%d, ny=%d, nz_spg=%d, nr_spg=%d, N_epoch=%d, N_sample=%d"
          % (nb_spg, ny, nz_spg, nr_spg, cal["N_epoch"], cal["N_sample"]), flush=True)
    print("JAX device:", jax.default_backend(), flush=True)

    start = time.perf_counter()
    G0_adaptive = G0_steady
    for epoch in range(cal["N_epoch"]):
        warm_up = epoch < cal["N_warmup"]
        key_train, key_epoch = jax.random.split(key_train)
        G0_phase = G0_steady if warm_up else G0_adaptive

        theta_old = theta
        (loss_val, G_end_mean), g = value_and_grad_fn(theta, key_epoch, G0_phase, warm_up)
        L_val = -loss_val
        updates, opt_state = optimizer.update(g, opt_state)
        theta = optax.apply_updates(theta, updates)

        if not warm_up:
            G0_adaptive = jax.lax.stop_gradient(G_end_mean)

        loss_hist.append(float(L_val))
        param_change = float(jnp.abs(theta - theta_old).max())
        lr_t = lr_schedule(epoch)
        if param_change < cal["e_converge"]:
            print("Converged at epoch %d, |Δθ|_max = %.2e" % (epoch + 1, param_change), flush=True)
            break
        if (epoch + 1) % args.log_every == 0 or (epoch + 1) <= 5 or (epoch + 1) == cal["N_warmup"]:
            phase = "warm-up (G fixed)" if warm_up else "G evolves"
            print("Epoch %d, L(θ) = %.6f, lr = %.2e, |Δθ| = %.2e, %s"
                  % (epoch + 1, L_val, lr_t, param_change, phase), flush=True)

    elapsed = time.perf_counter() - start
    print("Training done in %.2f s (%d epochs)" % (elapsed, len(loss_hist)), flush=True)
    if args.benchmark:
        print("BENCHMARK: %.2f s total, %.4f s/epoch" % (elapsed, elapsed / max(1, len(loss_hist))), flush=True)

    # ---------- Save visualizations and full grid to out_dir ----------
    os.makedirs(args.out_dir, exist_ok=True)
    c_grid_jax = theta_to_c(theta, c_min)  # (nz, nr, nb, ny)
    c_grid_np = np.array(c_grid_jax).transpose(2, 3, 0, 1)  # (nb, ny, nz, nr) for policy_from_grid
    b_grid_np = np.array(b_grid)
    y_grid_np = np.array(y_grid)
    z_grid_np = np.array(z_grid)
    r_grid_np = np.array(r_grid)

    def policy_cur(b, iy, iz, ir):
        return policy_from_grid(b, iy, iz, ir, c_grid_np, b_grid_np, y_grid_np, z_grid_np, r_grid_np, c_min_val=c_min)

    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.unicode_minus"] = False

    # 1. Loss curve
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    ax.plot(loss_hist, color="tab:blue")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("L(θ)")
    ax.set_title("SPG training loss (JAX)")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "loss_curve.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved %s" % os.path.join(args.out_dir, "loss_curve.png"), flush=True)

    # 2. Consumption policy c(b, y, r, z) — grid of c vs b for different r, z
    fig, axs = plt.subplots(3, 3, figsize=(18, 9))
    ir_indices = [0, nr_spg // 2, nr_spg - 1] if nr_spg >= 3 else list(range(nr_spg))
    iz_indices = [0, len(z_grid_np) // 2, len(z_grid_np) - 1]
    for row, ir_v in enumerate(ir_indices[:3]):
        for col, iz_v in enumerate(iz_indices):
            ax = axs[row, col]
            iy_v = 1
            b_lin = np.linspace(b_min, b_max, 100)
            c_ge, _ = policy_cur(b_lin, iy_v, iz_v, ir_v)
            label = "y=%.3f, r=%.3f, z=%.2f" % (y_grid_np[iy_v], r_grid_np[ir_v], z_grid_np[iz_v])
            ax.plot(b_lin, c_ge, label=label)
            ax.set_xlabel("Bond holdings b")
            ax.set_ylabel("Consumption c_GE(b,y,r,z)")
            ax.grid(alpha=0.3)
            ax.legend()
    fig.suptitle("SRL/GE: Consumption policy c_GE(b, y, r, z)")
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(os.path.join(args.out_dir, "consumption_policy_grid.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved %s" % os.path.join(args.out_dir, "consumption_policy_grid.png"), flush=True)

    # 2a. Zoomed low-asset region where curvature is usually strongest
    fig, axs = plt.subplots(3, 3, figsize=(18, 9))
    b_zoom_max = min(6.0, b_max)
    for row, ir_v in enumerate(ir_indices[:3]):
        for col, iz_v in enumerate(iz_indices):
            ax = axs[row, col]
            iy_v = 1
            b_lin = np.linspace(b_min, b_zoom_max, 200)
            c_ge, _ = policy_cur(b_lin, iy_v, iz_v, ir_v)
            label = "y=%.3f, r=%.3f, z=%.2f" % (y_grid_np[iy_v], r_grid_np[ir_v], z_grid_np[iz_v])
            ax.plot(b_lin, c_ge, label=label)
            ax.set_xlabel("Bond holdings b (zoom)")
            ax.set_ylabel("Consumption c_GE")
            ax.grid(alpha=0.3)
            ax.legend()
    fig.suptitle("SRL/GE: Consumption policy (low-b zoom)")
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(os.path.join(args.out_dir, "consumption_policy_grid_zoom_low_b.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved %s" % os.path.join(args.out_dir, "consumption_policy_grid_zoom_low_b.png"), flush=True)

    # 2b. Policy curvature diagnostics: detect near-linear shape in b-dimension
    fig, axs = plt.subplots(1, 3, figsize=(18, 4))
    curvature_lines = []
    for k, ir_v in enumerate(ir_indices[:3]):
        ax = axs[k]
        iy_v = 1
        iz_v = len(z_grid_np) // 2
        b_lin = np.linspace(b_min, b_max, 300)
        c_ge, _ = policy_cur(b_lin, iy_v, iz_v, ir_v)
        dc = np.gradient(c_ge, b_lin)
        d2c = np.gradient(dc, b_lin)
        curvature_lines.append((ir_v, float(np.mean(np.abs(d2c))), float(np.quantile(np.abs(d2c), 0.95))))
        ax.plot(b_lin, d2c, linewidth=1.5)
        ax.axhline(0.0, color="k", linestyle="--", linewidth=1)
        ax.set_title("d2c/db2 at r=%.4f" % r_grid_np[ir_v])
        ax.set_xlabel("b")
        ax.set_ylabel("curvature")
        ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "consumption_curvature_b.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved %s" % os.path.join(args.out_dir, "consumption_curvature_b.png"), flush=True)

    # 3. c vs r — given b, y, z, how c changes with r
    iz_mid = len(z_grid_np) // 2
    iy_s, ib_s = 1, nb_spg // 2
    b_val = float(b_grid_np[ib_s])
    c_r_curve = [policy_cur(b_val, iy_s, iz_mid, ir_v)[0] for ir_v in range(nr_spg)]
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    ax.plot(r_grid_np, c_r_curve, "-o")
    ax.set_xlabel("r")
    ax.set_ylabel("c(b,y,r,z)")
    ax.set_title("Consumption vs r (b=%.2f, y=%.2f, z=%.2f)" % (b_val, y_grid_np[iy_s], z_grid_np[iz_mid]))
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "consumption_vs_r.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved %s" % os.path.join(args.out_dir, "consumption_vs_r.png"), flush=True)

    # 3b. Continuous-r interpolation curve for finer shape diagnostics
    r_fine = np.linspace(r_min, r_max, 200)
    c_r_fine = [policy_cur(b_val, iy_s, iz_mid, float(rv))[0] for rv in r_fine]
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    ax.plot(r_fine, c_r_fine, "-", linewidth=2)
    ax.set_xlabel("r")
    ax.set_ylabel("c(b,y,r,z)")
    ax.set_title("Consumption vs r (continuous interpolation)")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "consumption_vs_r_fine.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved %s" % os.path.join(args.out_dir, "consumption_vs_r_fine.png"), flush=True)

    # 3c. Savings policy b'(b) at fixed y,z,r to inspect nonlinearity directly
    fig, axs = plt.subplots(1, 3, figsize=(18, 4))
    b_lin = np.linspace(b_min, b_max, 300)
    for k, ir_v in enumerate(ir_indices[:3]):
        ax = axs[k]
        c_line, b_next = policy_cur(b_lin, iy_s, iz_mid, ir_v)
        ax.plot(b_lin, b_next, label="b'(b)")
        ax.plot(b_lin, b_lin, "--", color="k", linewidth=1, label="45-degree")
        ax.set_title("Savings policy at r=%.4f" % r_grid_np[ir_v])
        ax.set_xlabel("b")
        ax.set_ylabel("b'")
        ax.grid(alpha=0.3)
        ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "savings_policy_vs_b.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved %s" % os.path.join(args.out_dir, "savings_policy_vs_b.png"), flush=True)

    # 5. Post-training diagnostics path: p/r trajectory, clearing residual, mean assets
    key_train, key_diag = jax.random.split(key_train)
    diag = simulate_diagnostics_path(
        theta, G0_adaptive, key_diag, args.diag_steps,
        b_grid, y_grid, z_grid, r_grid, Ty, Tz,
        nb_spg, ny, B, b_min, b_max, c_min,
    )
    np.save(os.path.join(args.out_dir, "r_path.npy"), diag["r_path"])
    np.save(os.path.join(args.out_dir, "z_path.npy"), diag["z_path"])
    np.save(os.path.join(args.out_dir, "market_clearing_residual.npy"), diag["residual_path"])
    np.save(os.path.join(args.out_dir, "mean_b_path.npy"), diag["mean_b_path"])
    print(
        "Diagnostics: residual mean=%.3e, max_abs=%.3e"
        % (float(np.mean(diag["residual_path"])), float(np.max(np.abs(diag["residual_path"])))),
        flush=True,
    )

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.plot(diag["r_path"], label="r_t / p_t")
    ax.set_xlabel("t")
    ax.set_ylabel("rate")
    ax.set_title("Simulated price trajectory")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "price_trajectory.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved %s" % os.path.join(args.out_dir, "price_trajectory.png"), flush=True)

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.plot(diag["residual_path"])
    ax.axhline(0.0, color="k", linestyle="--", linewidth=1)
    ax.set_xlabel("t")
    ax.set_ylabel("S(r_t,z_t)-B")
    ax.set_title("Market-clearing residual along simulated path")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "market_clearing_residual.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved %s" % os.path.join(args.out_dir, "market_clearing_residual.png"), flush=True)

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.plot(diag["mean_b_path"])
    ax.set_xlabel("t")
    ax.set_ylabel("E_t[b]")
    ax.set_title("Mean bond holdings along simulated path")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "mean_b_trajectory.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved %s" % os.path.join(args.out_dir, "mean_b_trajectory.png"), flush=True)

    # 4. Full grid and loss history as files
    np.save(os.path.join(args.out_dir, "c_grid.npy"), c_grid_np)
    np.save(os.path.join(args.out_dir, "b_grid.npy"), b_grid_np)
    np.save(os.path.join(args.out_dir, "y_grid.npy"), y_grid_np)
    np.save(os.path.join(args.out_dir, "z_grid.npy"), z_grid_np)
    np.save(os.path.join(args.out_dir, "r_grid.npy"), r_grid_np)
    with open(os.path.join(args.out_dir, "loss_hist.txt"), "w") as f:
        f.write("\n".join(map(str, loss_hist)))
    with open(os.path.join(args.out_dir, "policy_diagnostics.txt"), "w") as f:
        f.write("Linear-shape diagnostics (y=mid, z=mid)\n")
        for ir_v, mean_abs_d2, p95_abs_d2 in curvature_lines:
            c_line, _ = policy_cur(np.array(b_grid_np), 1, len(z_grid_np) // 2, ir_v)
            coeff = np.polyfit(b_grid_np, c_line, 1)
            fit = np.polyval(coeff, b_grid_np)
            denom = float(np.sum((c_line - np.mean(c_line)) ** 2))
            r2 = 1.0 - float(np.sum((c_line - fit) ** 2)) / max(denom, 1e-20)
            f.write(
                "ir=%d r=%.6f R2_linear=%.8f mean_abs_d2=%.8e p95_abs_d2=%.8e\n"
                % (ir_v, r_grid_np[ir_v], r2, mean_abs_d2, p95_abs_d2)
            )
    print("Saved %s (c_grid, grids, loss_hist.txt)" % args.out_dir, flush=True)


if __name__ == "__main__":
    main()
