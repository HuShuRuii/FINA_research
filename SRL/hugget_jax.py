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
    iz0: int | None = None,
) -> dict:
    """Simulate one path and return diagnostics for debugging/visualization."""
    G = _G_to_mat(G0, nb, ny)
    iz = int(iz0) if iz0 is not None else int(z_grid.shape[0] // 2)
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


def simulate_path_no_grad_single_traj(
    key: jnp.ndarray,
    theta: jnp.ndarray,
    G0: jnp.ndarray,
    iz0: jnp.ndarray,
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
    B: float,
    b_min: float,
    b_max: float,
    c_min_val: float,
    warm_up: bool,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Simulate a full path with detached prices and detached stored states.
    先做完整前向路径模拟：价格更新截断梯度，缓存下来的状态也不保留整条反向图。"""
    G0_mat = _G_to_mat(G0, nb, ny)
    iz0 = jnp.int32(iz0) if jnp.ndim(iz0) == 0 else iz0

    def body(carry, _t):
        G_cur, iz_cur, key_cur = carry
        r_star, ir_lo, ir_hi, w_r = P_star_bracket(
            theta, G_cur, iz_cur, b_grid, y_grid, z_grid, r_grid, ny, B, b_min, b_max, c_min_val
        )
        r_star = lax.stop_gradient(r_star)
        ir_lo = lax.stop_gradient(ir_lo)
        ir_hi = lax.stop_gradient(ir_hi)
        w_r = lax.stop_gradient(w_r)

        c = theta_to_c(theta, c_min_val)
        c_iz = c[iz_cur, :, :, :]
        c_t = (1.0 - w_r) * c_iz[ir_lo, :, :] + w_r * c_iz[ir_hi, :, :]
        if warm_up:
            G_next = G_cur
        else:
            G_next = update_G_from_c_and_r(
                c_t, r_star, lax.stop_gradient(G_cur),
                b_grid, y_grid, z_grid, iz_cur, Ty, nb, ny, b_min, b_max,
            )
        G_next = lax.stop_gradient(G_next)
        key_next, subkey = jax.random.split(key_cur)
        iz_next = jax.random.choice(subkey, nz_spg, p=Tz[iz_cur, :])
        return (G_next, iz_next, key_next), (G_cur, iz_cur)

    (G_final, iz_final, _), (G_hist, iz_hist) = jax.lax.scan(
        body,
        (G0_mat, iz0, key),
        jnp.arange(T_horizon),
    )
    G_path = jnp.concatenate([G_hist, G_final[None, :, :]], axis=0)
    iz_path = jnp.concatenate([iz_hist, jnp.asarray([iz_final], dtype=jnp.int32)], axis=0)
    return G_path, iz_path, G_final.reshape(-1)


def truncated_time_batch_objective_single_traj(
    theta: jnp.ndarray,
    G_path: jnp.ndarray,
    iz_path: jnp.ndarray,
    sample_ts: jnp.ndarray,
    T_horizon: int,
    n_update: int,
    g_grad_window: int,
    b_grid: jnp.ndarray,
    y_grid: jnp.ndarray,
    z_grid: jnp.ndarray,
    r_grid: jnp.ndarray,
    Ty: jnp.ndarray,
    ny: int,
    B: float,
    b_min: float,
    b_max: float,
    c_min_val: float,
    sigma: float,
    beta: float,
    e_trunc: float,
    warm_up: bool,
) -> jnp.ndarray:
    """Replay only sampled time steps with a truncated G-gradient window.
    只对抽样时点做短窗口回放，从而保留局部分布梯度、避免整条 170 期全量反传。"""
    window = max(int(g_grad_window), 1)
    n_update_eff = max(int(n_update), 1)
    scale = jnp.asarray(T_horizon / n_update_eff, dtype=jnp.float32)

    def term_for_t(t_idx):
        t_idx = jnp.int32(t_idx)
        start = jnp.maximum(t_idx - (window - 1), 0)
        window_len = t_idx - start + 1
        G_start = lax.stop_gradient(lax.dynamic_index_in_dim(G_path, start, axis=0, keepdims=False))

        def window_body(carry, rel_step):
            G_cur = carry
            active = rel_step < window_len
            global_t = start + rel_step
            iz_cur = lax.dynamic_index_in_dim(iz_path, global_t, axis=0, keepdims=False)
            r_star, ir_lo, ir_hi, w_r = P_star_bracket(
                theta, G_cur, iz_cur, b_grid, y_grid, z_grid, r_grid, ny, B, b_min, b_max, c_min_val
            )
            r_star = lax.stop_gradient(r_star)
            ir_lo = lax.stop_gradient(ir_lo)
            ir_hi = lax.stop_gradient(ir_hi)
            w_r = lax.stop_gradient(w_r)

            c = theta_to_c(theta, c_min_val)
            c_iz = c[iz_cur, :, :, :]
            c_t = (1.0 - w_r) * c_iz[ir_lo, :, :] + w_r * c_iz[ir_hi, :, :]
            weight = (beta ** global_t) * jnp.float32((beta ** global_t) >= e_trunc)
            term = weight * (G_cur * u_jax(c_t, sigma, c_min_val)).sum()
            if warm_up:
                G_next = G_cur
            else:
                G_next = update_G_from_c_and_r(
                    c_t, r_star, G_cur,
                    b_grid, y_grid, z_grid, iz_cur, Ty, b_grid.shape[0], ny, b_min, b_max,
                )
            G_out = jnp.where(active, G_next, G_cur)
            term_out = jnp.where(active & (global_t == t_idx), term, jnp.float32(0.0))
            return G_out, term_out

        _, term_hist = jax.lax.scan(window_body, G_start, jnp.arange(window))
        return scale * term_hist.sum()

    return jax.vmap(term_for_t)(sample_ts).mean()


def spg_objective_single_traj(
    key: jnp.ndarray,
    theta: jnp.ndarray,
    G0: jnp.ndarray,
    iz0: jnp.ndarray,  # scalar (e.g. from vmap)
    T_horizon: int,
    n_update: int,
    g_grad_window: int,
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
    """Single-trajectory objective plus final distribution.
    单条轨迹的目标值与终端分布。"""
    key_path, key_sample = jax.random.split(key)
    G_path, iz_path, G_final = simulate_path_no_grad_single_traj(
        key_path, theta, G0, iz0, T_horizon,
        b_grid, y_grid, z_grid, r_grid, Ty, Tz,
        nb, ny, nz_spg, B, b_min, b_max, c_min_val, warm_up,
    )
    n_update_eff = min(max(int(n_update), 1), int(T_horizon))
    sample_ts = jax.random.choice(key_sample, T_horizon, shape=(n_update_eff,), replace=False)
    L_n = truncated_time_batch_objective_single_traj(
        theta, G_path, iz_path, sample_ts, T_horizon, n_update_eff, g_grad_window,
        b_grid, y_grid, z_grid, r_grid, Ty, ny, B, b_min, b_max, c_min_val, sigma, beta, e_trunc, warm_up,
    )
    return L_n, G_final


def spg_objective(
    theta: jnp.ndarray,
    key: jnp.ndarray,
    N_traj: int,
    T_horizon: int,
    n_update: int,
    g_grad_window: int,
    G0_batch: jnp.ndarray,
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
    """Return mean objective and mean terminal distribution across trajectories.
    返回跨轨迹平均目标值以及平均终端分布。"""
    key, k1, k2 = jax.random.split(key, 3)
    keys = jax.random.split(k1, N_traj)
    iz0s = jax.random.randint(k2, (N_traj,), 0, nz_spg)
    if G0_batch.ndim == 1:
        G0_batch = jnp.broadcast_to(G0_batch[None, :], (N_traj, G0_batch.shape[0]))

    def body(key_i, iz0, G0_i):
        return spg_objective_single_traj(
            key_i, theta, G0_i, iz0, T_horizon, n_update, g_grad_window,
            b_grid, y_grid, z_grid, r_grid, Ty, Tz,
            nb, ny, nz_spg, nr_spg, beta, e_trunc, B, b_min, b_max, c_min_val, sigma, warm_up,
        )

    L_list, G_final_list = jax.vmap(body)(keys, iz0s, G0_batch)
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


def build_uniform_b_G0(b_grid: jnp.ndarray, invariant_y: jnp.ndarray, ny: int) -> jnp.ndarray:
    """Construct a broad initial distribution with uniform b mass and invariant y mass."""
    del ny  # kept for interface symmetry with other G0 builders
    w_b = jnp.ones_like(b_grid) / jnp.maximum(b_grid.shape[0], 1)
    w_y = invariant_y / jnp.maximum(invariant_y.sum(), 1e-20)
    G = w_b[:, None] * w_y[None, :]
    return (G / jnp.maximum(G.sum(), 1e-20)).reshape(-1)


def build_high_b_G0(b_grid: jnp.ndarray, invariant_y: jnp.ndarray, ny: int, high_power: float = 6.0) -> jnp.ndarray:
    """Construct an initial distribution concentrated on high-b region."""
    del ny  # kept for interface symmetry with other G0 builders
    b01 = (b_grid - b_grid.min()) / jnp.maximum(b_grid.max() - b_grid.min(), 1e-20)
    w_b = jnp.maximum(b01, 1e-6) ** high_power
    w_b = w_b / jnp.maximum(w_b.sum(), 1e-20)
    w_y = invariant_y / jnp.maximum(invariant_y.sum(), 1e-20)
    G = w_b[:, None] * w_y[None, :]
    return (G / jnp.maximum(G.sum(), 1e-20)).reshape(-1)


def sample_training_G0_batch(
    key: jnp.ndarray,
    n_traj: int,
    G0_anchor: jnp.ndarray,
    G0_uniform: jnp.ndarray,
    G0_high_b: jnp.ndarray,
    broad_share: float,
    beta_concentration: float,
) -> jnp.ndarray:
    """Sample a batch of broad but GE-valid initial distributions for post-warm-up training.

    Each trajectory starts from a convex combination of the steady-state anchor and a broad
    perturbation spanning uniform to high-b cross sections. This broadens coverage while
    keeping the original GE/SPG objective unchanged.
    """
    share = jnp.clip(jnp.asarray(broad_share, dtype=G0_anchor.dtype), 0.0, 1.0)
    conc = jnp.maximum(jnp.asarray(beta_concentration, dtype=G0_anchor.dtype), 1e-3)
    high_b_weight = jax.random.beta(key, conc, conc, shape=(n_traj, 1))
    G_broad = (1.0 - high_b_weight) * G0_uniform[None, :] + high_b_weight * G0_high_b[None, :]
    G0_batch = (1.0 - share) * G0_anchor[None, :] + share * G_broad
    return G0_batch / jnp.maximum(G0_batch.sum(axis=1, keepdims=True), 1e-20)


def interpolate_c_at_r(
    theta: jnp.ndarray,
    iz: int,
    r_star: jnp.ndarray,
    r_grid: jnp.ndarray,
    c_min_val: float,
) -> jnp.ndarray:
    """Interpolate policy in r for a fixed z, returning c_t with shape (nb, ny)."""
    nr = r_grid.shape[0]
    c = theta_to_c(theta, c_min_val)
    if nr == 1:
        return c[iz, 0, :, :]
    ir_hi = jnp.clip(jnp.searchsorted(r_grid, r_star, side="right"), 1, nr - 1)
    ir_lo = ir_hi - 1
    denom = jnp.maximum(r_grid[ir_hi] - r_grid[ir_lo], 1e-20)
    w_r = (r_star - r_grid[ir_lo]) / denom
    c_iz = c[iz, :, :, :]
    return (1.0 - w_r) * c_iz[ir_lo, :, :] + w_r * c_iz[ir_hi, :, :]


def stationary_distribution_given_r(
    theta: jnp.ndarray,
    r_star: jnp.ndarray,
    G_init: jnp.ndarray,
    iz: int,
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
    n_iter: int = 500,
    tol: float = 1e-10,
) -> jnp.ndarray:
    """Invariant distribution for fixed (z, r) and current policy."""
    G = _G_to_mat(G_init, nb, ny)
    c_t = interpolate_c_at_r(theta, iz, r_star, r_grid, c_min_val)
    for _ in range(n_iter):
        G_new = update_G_from_c_and_r(c_t, r_star, G, b_grid, y_grid, z_grid, iz, Ty, nb, ny, b_min, b_max)
        if float(jnp.max(jnp.abs(G_new - G))) < tol:
            G = G_new
            break
        G = G_new
    return G.reshape(-1)


def solve_huggett_steady_state(
    theta: jnp.ndarray,
    G_init: jnp.ndarray,
    b_grid: jnp.ndarray,
    y_grid: jnp.ndarray,
    z_grid: jnp.ndarray,
    r_grid: jnp.ndarray,
    Ty: jnp.ndarray,
    nb: int,
    ny: int,
    B: float,
    b_min: float,
    b_max: float,
    c_min_val: float,
    n_outer: int = 80,
    r_tol: float = 1e-8,
    g_tol: float = 1e-10,
) -> dict:
    """Solve deterministic steady state at z≈1 by fixed point in (G, r)."""
    iz_ss = int(jnp.argmin(jnp.abs(z_grid - 1.0)))
    r_ss = jnp.array(r_grid[r_grid.shape[0] // 2], dtype=jnp.float32)
    G_ss = G_init
    outer_hist = []
    for _ in range(n_outer):
        G_prev = G_ss
        r_prev = r_ss
        G_ss = stationary_distribution_given_r(
            theta, r_ss, G_ss, iz_ss,
            b_grid, y_grid, z_grid, r_grid, Ty, nb, ny,
            b_min, b_max, c_min_val,
        )
        r_ss, ir_lo, ir_hi, w_r, residual = market_clearing_stats(
            theta, G_ss, iz_ss, b_grid, y_grid, z_grid, r_grid, ny, B, b_min, b_max, c_min_val
        )
        outer_hist.append((float(r_ss), float(residual), float(jnp.max(jnp.abs(G_ss - G_prev)))))
        if float(jnp.abs(r_ss - r_prev)) < r_tol and float(jnp.max(jnp.abs(G_ss - G_prev))) < g_tol:
            break
    G_mat = np.array(_G_to_mat(G_ss, nb, ny))
    b_mass = G_mat.sum(axis=1)
    return {
        "iz_ss": iz_ss,
        "z_ss": float(z_grid[iz_ss]),
        "r_ss": float(r_ss),
        "residual": float(residual),
        "G_ss": np.array(G_ss),
        "b_mass": b_mass,
        "outer_hist": np.array(outer_hist, dtype=float),
        "ir_lo": int(ir_lo),
        "ir_hi": int(ir_hi),
        "w_r": float(w_r),
    }


def policy_change_summary(
    c_final_np: np.ndarray,
    c_init_np: np.ndarray,
    b_grid_np: np.ndarray,
) -> dict:
    """Summarize how much the trained policy moved relative to initialization."""
    diff = np.abs(c_final_np - c_init_np)
    rel = diff / np.maximum(np.abs(c_init_np), 1e-8)
    abs_by_b = diff.mean(axis=(1, 2, 3))
    return {
        "diff": diff,
        "rel": rel,
        "abs_by_b": abs_by_b,
        "mean_abs": float(diff.mean()),
        "median_abs": float(np.median(diff)),
        "p90_abs": float(np.quantile(diff, 0.9)),
        "p99_abs": float(np.quantile(diff, 0.99)),
        "mean_rel": float(rel.mean()),
        "median_rel": float(np.median(rel)),
        "p90_rel": float(np.quantile(rel, 0.9)),
        "p99_rel": float(np.quantile(rel, 0.99)),
        "share_abs_le_1e-4": float((diff <= 1e-4).mean()),
        "share_abs_le_1e-3": float((diff <= 1e-3).mean()),
        "share_abs_le_1e-2": float((diff <= 1e-2).mean()),
        "share_abs_le_5e-2": float((diff <= 5e-2).mean()),
        "share_abs_le_1e-1": float((diff <= 1e-1).mean()),
        "share_b_mean_abs_le_1e-2": float((abs_by_b <= 1e-2).mean()),
        "mass_change_b_le_2": float(abs_by_b[b_grid_np <= 2.0].mean()) if np.any(b_grid_np <= 2.0) else float("nan"),
        "mass_change_b_2_10": float(abs_by_b[(b_grid_np > 2.0) & (b_grid_np <= 10.0)].mean())
        if np.any((b_grid_np > 2.0) & (b_grid_np <= 10.0)) else float("nan"),
        "mass_change_b_gt_10": float(abs_by_b[b_grid_np > 10.0].mean()) if np.any(b_grid_np > 10.0) else float("nan"),
    }


def validate_solution(
    r_path: np.ndarray,
    residual_path: np.ndarray,
    r_grid_np: np.ndarray,
    residual_tol: float,
    ss: dict | None = None,
) -> dict:
    """Check whether the learned GE solution stays away from invalid boundary outcomes."""
    if r_grid_np.shape[0] > 1:
        boundary_buffer = 0.5 * float(r_grid_np[1] - r_grid_np[0])
    else:
        boundary_buffer = 0.0
    r_min = float(r_grid_np[0])
    r_max = float(r_grid_np[-1])
    max_abs_resid = float(np.max(np.abs(residual_path)))
    touches_lower = bool(np.any(r_path <= r_min + boundary_buffer))
    touches_upper = bool(np.any(r_path >= r_max - boundary_buffer))
    ss_touches_lower = False
    ss_touches_upper = False
    ss_resid = None
    ss_r = None
    if ss is not None:
        ss_r = float(ss["r_ss"])
        ss_resid = float(ss["residual"])
        ss_touches_lower = bool(ss_r <= r_min + boundary_buffer)
        ss_touches_upper = bool(ss_r >= r_max - boundary_buffer)
    is_valid = (max_abs_resid <= residual_tol) and (not touches_lower) and (not touches_upper)
    if ss is not None:
        is_valid = is_valid and (abs(ss_resid) <= residual_tol) and (not ss_touches_lower) and (not ss_touches_upper)
    return {
        "is_valid": bool(is_valid),
        "max_abs_residual": max_abs_resid,
        "mean_residual": float(np.mean(residual_path)),
        "touches_lower_bound": touches_lower,
        "touches_upper_bound": touches_upper,
        "boundary_buffer": boundary_buffer,
        "r_min": r_min,
        "r_max": r_max,
        "steady_state_r": ss_r,
        "steady_state_residual": ss_resid,
        "steady_state_touches_lower": ss_touches_lower,
        "steady_state_touches_upper": ss_touches_upper,
        "residual_tol": float(residual_tol),
    }


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
    parser.add_argument("--n_update", type=int, default=16,
                        help="Number of sampled time steps per trajectory used in each gradient update")
    parser.add_argument("--g_grad_window", type=int, default=10,
                        help="Truncated horizon for distribution-gradient replay around each sampled time step")
    parser.add_argument("--g0_mode", type=str, default="steady_high_mix",
                        choices=["steady", "uniform", "high_b", "steady_high_mix"],
                        help="Warm-up initial distribution setup")
    parser.add_argument("--g0_high_mix_warmup", type=float, default=0.8,
                        help="Mix weight of high-b G0 during warm-up")
    parser.add_argument("--g0_high_mix_after", type=float, default=0.5,
                        help="High-b share in the adaptive post-warm-up GE anchor")
    parser.add_argument("--g0_high_power", type=float, default=6.0,
                        help="Concentration power for high-b initial distribution")
    parser.add_argument("--coverage_traj_share", type=float, default=0.25,
                        help="Share of post-warm-up trajectories started from broad GE-valid G0s")
    parser.add_argument("--coverage_decay_epochs", type=int, default=150,
                        help="Number of post-warm-up epochs over which broad-start coverage decays to zero")
    parser.add_argument("--post_warmup_broad_share", type=float, default=1.0,
                        help="Broadness of the extra coverage trajectories after warm-up")
    parser.add_argument("--post_warmup_beta_conc", type=float, default=0.7,
                        help="Beta concentration for uniform-vs-high-b broad perturbations")
    parser.add_argument("--explore_weight", type=float, default=0.0,
                        help="Deprecated. Must remain 0 to preserve the GE objective.")
    parser.add_argument("--explore_high_b_mix", type=float, default=0.6,
                        help="Deprecated and ignored")
    parser.add_argument("--solve_steady_state", action="store_true",
                        help="Solve and save deterministic steady state diagnostics after training")
    parser.add_argument("--residual_valid_tol", type=float, default=1e-6,
                        help="Validity tolerance for market-clearing residual checks")
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
    if args.explore_weight != 0.0:
        raise ValueError(
            "--explore_weight changes the GE objective and is disabled. "
            "Use the steady-state-anchored post-warm-up G0 broadening instead."
        )

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
    theta_init = theta
    G0_steady = steady_state_G0(
        theta, b_grid, y_grid, z_grid, r_grid, Ty,
        nb_spg, ny, nz_spg, nr_spg, b_min, b_max, c_min,
    )
    invariant_y_jax = jnp.array(invariant_y)
    G0_uniform = build_uniform_b_G0(b_grid, invariant_y_jax, ny)
    G0_high_b = build_high_b_G0(b_grid, invariant_y_jax, ny, high_power=args.g0_high_power)

    if args.g0_mode == "steady":
        G0_warmup_base = G0_steady
    elif args.g0_mode == "uniform":
        G0_warmup_base = G0_uniform
    elif args.g0_mode == "high_b":
        G0_warmup_base = G0_high_b
    else:  # steady_high_mix
        mix_w = float(np.clip(args.g0_high_mix_warmup, 0.0, 1.0))
        G0_warmup_base = (1.0 - mix_w) * G0_steady + mix_w * G0_high_b
        G0_warmup_base = G0_warmup_base / jnp.maximum(G0_warmup_base.sum(), 1e-20)
    T_horizon = cal["T_trunc"]

    # Objective and gradient
    def make_G0_batch(key, warm_up, G0_anchor, coverage_share):
        if warm_up:
            return jnp.broadcast_to(G0_warmup_base[None, :], (cal["N_sample"], G0_warmup_base.shape[0]))
        G0_anchor_batch = jnp.broadcast_to(G0_anchor[None, :], (cal["N_sample"], G0_anchor.shape[0]))
        if coverage_share <= 0.0:
            return G0_anchor_batch
        key_mask, key_cov = jax.random.split(key)
        G0_cov_batch = sample_training_G0_batch(
            key_cov,
            cal["N_sample"],
            G0_anchor,
            G0_uniform,
            G0_high_b,
            args.post_warmup_broad_share,
            args.post_warmup_beta_conc,
        )
        use_cov = jax.random.bernoulli(
            key_mask,
            p=jnp.clip(jnp.asarray(coverage_share, dtype=G0_anchor.dtype), 0.0, 1.0),
            shape=(cal["N_sample"], 1),
        )
        return jnp.where(use_cov, G0_cov_batch, G0_anchor_batch)

    def objective_fn(theta, key, G0_batch, warm_up):
        return spg_objective(
            theta, key, cal["N_sample"], T_horizon, args.n_update, args.g_grad_window, G0_batch,
            b_grid, y_grid, z_grid, r_grid, Ty, Tz,
            nb_spg, ny, nz_spg, nr_spg, beta, cal["e_trunc"], B,
            b_min, b_max, c_min, sigma, warm_up,
        )

    def loss_with_aux(theta, key, G0_batch, warm_up):
        L_main, G_end_mean = objective_fn(theta, key, G0_batch, warm_up)
        return -L_main, (G_end_mean, L_main)

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
    print(
        "Warm-up G0: mode=%s, high_mix=%.2f, high_power=%.1f"
        % (args.g0_mode, args.g0_high_mix_warmup, args.g0_high_power),
        flush=True,
    )
    print(
        "Post-warm-up GE anchor: high_mix=%.2f, coverage_traj_share=%.2f, coverage_decay_epochs=%d, coverage_broad_share=%.2f, beta_conc=%.2f"
        % (
            args.g0_high_mix_after,
            args.coverage_traj_share,
            args.coverage_decay_epochs,
            args.post_warmup_broad_share,
            args.post_warmup_beta_conc,
        ),
        flush=True,
    )
    print(
        "Time minibatch: n_update=%d, g_grad_window=%d, T_horizon=%d"
        % (args.n_update, args.g_grad_window, T_horizon),
        flush=True,
    )

    start = time.perf_counter()
    G0_adaptive = G0_warmup_base
    for epoch in range(cal["N_epoch"]):
        warm_up = epoch < cal["N_warmup"]
        key_train, key_epoch, key_g0 = jax.random.split(key_train, 3)
        if warm_up:
            G0_anchor = G0_warmup_base
            coverage_share_t = 0.0
        else:
            mix_a = float(np.clip(args.g0_high_mix_after, 0.0, 1.0))
            G0_anchor = (1.0 - mix_a) * G0_adaptive + mix_a * G0_high_b
            G0_anchor = G0_anchor / jnp.maximum(G0_anchor.sum(), 1e-20)
            if args.coverage_decay_epochs <= 0:
                coverage_share_t = 0.0
            else:
                post_warm_epoch = max(epoch - cal["N_warmup"], 0)
                decay = max(0.0, 1.0 - post_warm_epoch / float(args.coverage_decay_epochs))
                coverage_share_t = float(args.coverage_traj_share) * decay
        G0_batch = make_G0_batch(key_g0, warm_up, G0_anchor, coverage_share_t)

        theta_old = theta
        (loss_val, (G_end_mean, L_main)), g = value_and_grad_fn(theta, key_epoch, G0_batch, warm_up)
        L_total = -loss_val
        updates, opt_state = optimizer.update(g, opt_state)
        theta = optax.apply_updates(theta, updates)

        if not warm_up:
            G0_adaptive = jax.lax.stop_gradient(G_end_mean)

        loss_hist.append(float(L_total))
        param_change = float(jnp.abs(theta - theta_old).max())
        lr_t = lr_schedule(epoch)
        if param_change < cal["e_converge"]:
            print("Converged at epoch %d, |Δθ|_max = %.2e" % (epoch + 1, param_change), flush=True)
            break
        if (epoch + 1) % args.log_every == 0 or (epoch + 1) <= 5 or (epoch + 1) == cal["N_warmup"]:
            phase = "warm-up (G fixed)" if warm_up else "G evolves"
            print(
                "Epoch %d, L_total = %.6f, lr = %.2e, |Δθ| = %.2e, %s"
                % (epoch + 1, L_total, lr_t, param_change, phase),
                flush=True,
            )

    elapsed = time.perf_counter() - start
    print("Training done in %.2f s (%d epochs)" % (elapsed, len(loss_hist)), flush=True)
    if args.benchmark:
        print("BENCHMARK: %.2f s total, %.4f s/epoch" % (elapsed, elapsed / max(1, len(loss_hist))), flush=True)

    # ---------- Save visualizations and full grid to out_dir ----------
    os.makedirs(args.out_dir, exist_ok=True)
    c_grid_jax = theta_to_c(theta, c_min)  # (nz, nr, nb, ny)
    c_grid_np = np.array(c_grid_jax).transpose(2, 3, 0, 1)  # (nb, ny, nz, nr) for policy_from_grid
    c_init_np = np.array(theta_to_c(theta_init, c_min)).transpose(2, 3, 0, 1)
    b_grid_np = np.array(b_grid)
    y_grid_np = np.array(y_grid)
    z_grid_np = np.array(z_grid)
    r_grid_np = np.array(r_grid)
    G0_basis_mass = {
        "steady": np.array(_G_to_mat(G0_steady, nb_spg, ny)).sum(axis=1),
        "uniform": np.array(_G_to_mat(G0_uniform, nb_spg, ny)).sum(axis=1),
        "high_b": np.array(_G_to_mat(G0_high_b, nb_spg, ny)).sum(axis=1),
        "warmup": np.array(_G_to_mat(G0_warmup_base, nb_spg, ny)).sum(axis=1),
    }

    def policy_cur(b, iy, iz, ir):
        return policy_from_grid(b, iy, iz, ir, c_grid_np, b_grid_np, y_grid_np, z_grid_np, r_grid_np, c_min_val=c_min)

    def policy_init(b, iy, iz, ir):
        return policy_from_grid(b, iy, iz, ir, c_init_np, b_grid_np, y_grid_np, z_grid_np, r_grid_np, c_min_val=c_min)

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

    # 1a. Training G0 bases over b to document coverage design
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    for label, b_mass in G0_basis_mass.items():
        ax.plot(b_grid_np, b_mass, linewidth=2, label=label)
    ax.set_xlabel("b")
    ax.set_ylabel("mass")
    ax.set_title("Warm-up and post-warm-up G0 building blocks")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "training_g0_bases.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved %s" % os.path.join(args.out_dir, "training_g0_bases.png"), flush=True)

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

    # 2a-bis. Final policy versus initialization on the economically relevant low-b region
    fig, axs = plt.subplots(3, 3, figsize=(18, 9))
    for row, ir_v in enumerate(ir_indices[:3]):
        for col, iz_v in enumerate(iz_indices):
            ax = axs[row, col]
            iy_v = 1
            b_lin = np.linspace(b_min, b_zoom_max, 200)
            c_init_line, _ = policy_init(b_lin, iy_v, iz_v, ir_v)
            c_final_line, _ = policy_cur(b_lin, iy_v, iz_v, ir_v)
            ax.plot(b_lin, c_init_line, "--", linewidth=1.5, label="init")
            ax.plot(b_lin, c_final_line, linewidth=2.0, label="final")
            ax.set_xlabel("Bond holdings b (zoom)")
            ax.set_ylabel("Consumption c")
            ax.set_title("r=%.4f, z=%.2f" % (r_grid_np[ir_v], z_grid_np[iz_v]))
            ax.grid(alpha=0.3)
            if row == 0 and col == 0:
                ax.legend()
    fig.suptitle("Policy comparison versus initialization")
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(os.path.join(args.out_dir, "policy_vs_init_low_b.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved %s" % os.path.join(args.out_dir, "policy_vs_init_low_b.png"), flush=True)

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

    policy_change = policy_change_summary(c_grid_np, c_init_np, b_grid_np)
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.plot(b_grid_np, policy_change["abs_by_b"], linewidth=2, label="mean |c_final-c_init|")
    ax.set_xlabel("b")
    ax.set_ylabel("mean absolute change")
    ax.set_title("Policy movement versus initialization")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "policy_change_by_b.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved %s" % os.path.join(args.out_dir, "policy_change_by_b.png"), flush=True)

    # 5. Steady-state diagnostics and simulated GE path
    ss = None
    G0_diag = steady_state_G0(
        theta, b_grid, y_grid, z_grid, r_grid, Ty,
        nb_spg, ny, nz_spg, nr_spg, b_min, b_max, c_min,
    )
    iz_diag = int(len(z_grid_np) // 2)
    if args.solve_steady_state:
        ss = solve_huggett_steady_state(
            theta, G0_diag, b_grid, y_grid, z_grid, r_grid, Ty,
            nb_spg, ny, B, b_min, b_max, c_min,
        )
        G0_diag = jnp.array(ss["G_ss"])
        iz_diag = int(ss["iz_ss"])
        np.save(os.path.join(args.out_dir, "steady_state_G.npy"), ss["G_ss"])
        np.save(os.path.join(args.out_dir, "steady_state_b_mass.npy"), ss["b_mass"])
        np.save(os.path.join(args.out_dir, "steady_state_outer_hist.npy"), ss["outer_hist"])

        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        ax.plot(b_grid_np, ss["b_mass"], linewidth=2)
        ax.set_xlabel("b")
        ax.set_ylabel("steady-state mass")
        ax.set_title("Deterministic steady-state mass over b")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        fig.savefig(os.path.join(args.out_dir, "steady_state_b_mass.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("Saved %s" % os.path.join(args.out_dir, "steady_state_b_mass.png"), flush=True)

        fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        outer_hist = ss["outer_hist"]
        axs[0].plot(outer_hist[:, 0], linewidth=2)
        axs[0].set_ylabel("r")
        axs[0].set_title("Steady-state outer iterations")
        axs[0].grid(alpha=0.3)
        axs[1].plot(outer_hist[:, 1], linewidth=2, label="market residual")
        axs[1].plot(outer_hist[:, 2], linewidth=2, label="max |ΔG|")
        axs[1].set_xlabel("outer iteration")
        axs[1].set_ylabel("diagnostic")
        axs[1].grid(alpha=0.3)
        axs[1].legend()
        plt.tight_layout()
        fig.savefig(os.path.join(args.out_dir, "steady_state_outer_iterations.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("Saved %s" % os.path.join(args.out_dir, "steady_state_outer_iterations.png"), flush=True)

        cum_mass = np.cumsum(ss["b_mass"])
        support_1e6 = b_grid_np[ss["b_mass"] > 1e-6]
        support_1e4 = b_grid_np[ss["b_mass"] > 1e-4]
        with open(os.path.join(args.out_dir, "steady_state_summary.txt"), "w") as f:
            f.write("z_ss=%.8f\n" % ss["z_ss"])
            f.write("r_ss=%.8f\n" % ss["r_ss"])
            f.write("market_residual=%.8e\n" % ss["residual"])
            f.write("ir_lo=%d ir_hi=%d w_r=%.8f\n" % (ss["ir_lo"], ss["ir_hi"], ss["w_r"]))
            f.write("mass_b_le_2=%.8f\n" % float(np.sum(ss["b_mass"][b_grid_np <= 2.0])))
            f.write("mass_b_le_6=%.8f\n" % float(np.sum(ss["b_mass"][b_grid_np <= 6.0])))
            f.write("mass_b_gt_10=%.8f\n" % float(np.sum(ss["b_mass"][b_grid_np > 10.0])))
            f.write("mass_b_gt_20=%.8f\n" % float(np.sum(ss["b_mass"][b_grid_np > 20.0])))
            f.write("max_b_with_mass_gt_1e-6=%.8f\n" % float(support_1e6.max() if support_1e6.size else b_grid_np[0]))
            f.write("max_b_with_mass_gt_1e-4=%.8f\n" % float(support_1e4.max() if support_1e4.size else b_grid_np[0]))
            for q in [0.5, 0.9, 0.95, 0.99, 0.999]:
                ib = int(np.searchsorted(cum_mass, q, side="left"))
                ib = min(ib, len(b_grid_np) - 1)
                f.write("b_quantile_%0.3f=%.8f\n" % (q, b_grid_np[ib]))
        print("Saved %s" % os.path.join(args.out_dir, "steady_state_summary.txt"), flush=True)

    key_train, key_diag = jax.random.split(key_train)
    diag = simulate_diagnostics_path(
        theta, G0_diag, key_diag, args.diag_steps,
        b_grid, y_grid, z_grid, r_grid, Ty, Tz,
        nb_spg, ny, B, b_min, b_max, c_min,
        iz0=iz_diag,
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

    validity = validate_solution(diag["r_path"], diag["residual_path"], r_grid_np, args.residual_valid_tol, ss=ss)

    # 4. Full grid and loss history as files
    np.save(os.path.join(args.out_dir, "c_grid.npy"), c_grid_np)
    np.save(os.path.join(args.out_dir, "b_grid.npy"), b_grid_np)
    np.save(os.path.join(args.out_dir, "y_grid.npy"), y_grid_np)
    np.save(os.path.join(args.out_dir, "z_grid.npy"), z_grid_np)
    np.save(os.path.join(args.out_dir, "r_grid.npy"), r_grid_np)
    np.save(os.path.join(args.out_dir, "c_init_grid.npy"), c_init_np)
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
    with open(os.path.join(args.out_dir, "policy_change_summary.txt"), "w") as f:
        f.write("mean_abs=%.8e\n" % policy_change["mean_abs"])
        f.write("median_abs=%.8e\n" % policy_change["median_abs"])
        f.write("p90_abs=%.8e\n" % policy_change["p90_abs"])
        f.write("p99_abs=%.8e\n" % policy_change["p99_abs"])
        f.write("mean_rel=%.8e\n" % policy_change["mean_rel"])
        f.write("median_rel=%.8e\n" % policy_change["median_rel"])
        f.write("p90_rel=%.8e\n" % policy_change["p90_rel"])
        f.write("p99_rel=%.8e\n" % policy_change["p99_rel"])
        f.write("share_abs_le_1e-4=%.8f\n" % policy_change["share_abs_le_1e-4"])
        f.write("share_abs_le_1e-3=%.8f\n" % policy_change["share_abs_le_1e-3"])
        f.write("share_abs_le_1e-2=%.8f\n" % policy_change["share_abs_le_1e-2"])
        f.write("share_abs_le_5e-2=%.8f\n" % policy_change["share_abs_le_5e-2"])
        f.write("share_abs_le_1e-1=%.8f\n" % policy_change["share_abs_le_1e-1"])
        f.write("share_b_mean_abs_le_1e-2=%.8f\n" % policy_change["share_b_mean_abs_le_1e-2"])
        f.write("mean_abs_change_b_le_2=%.8e\n" % policy_change["mass_change_b_le_2"])
        f.write("mean_abs_change_b_2_10=%.8e\n" % policy_change["mass_change_b_2_10"])
        f.write("mean_abs_change_b_gt_10=%.8e\n" % policy_change["mass_change_b_gt_10"])
    with open(os.path.join(args.out_dir, "validation_summary.txt"), "w") as f:
        f.write("is_valid=%d\n" % int(validity["is_valid"]))
        f.write("residual_tol=%.8e\n" % validity["residual_tol"])
        f.write("diag_mean_residual=%.8e\n" % validity["mean_residual"])
        f.write("diag_max_abs_residual=%.8e\n" % validity["max_abs_residual"])
        f.write("touches_lower_bound=%d\n" % int(validity["touches_lower_bound"]))
        f.write("touches_upper_bound=%d\n" % int(validity["touches_upper_bound"]))
        f.write("steady_state_r=%s\n" % ("nan" if validity["steady_state_r"] is None else "%.8f" % validity["steady_state_r"]))
        f.write("steady_state_residual=%s\n" % ("nan" if validity["steady_state_residual"] is None else "%.8e" % validity["steady_state_residual"]))
        f.write("steady_state_touches_lower=%d\n" % int(validity["steady_state_touches_lower"]))
        f.write("steady_state_touches_upper=%d\n" % int(validity["steady_state_touches_upper"]))
    print(
        "Validity: %s (diag max_abs_residual=%.3e, touches_upper=%s)"
        % (
            "VALID" if validity["is_valid"] else "INVALID",
            validity["max_abs_residual"],
            validity["touches_upper_bound"] or validity["steady_state_touches_upper"],
        ),
        flush=True,
    )
    print("Saved %s (c_grid, grids, loss_hist.txt)" % args.out_dir, flush=True)


if __name__ == "__main__":
    main()
