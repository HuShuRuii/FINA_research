#!/usr/bin/env python3
"""
Forward-Looking Huggett (1993) — JAX implementation.

Policy depends on p_{t+1} (expected next-period price). At time t, agents use
π(b, y, r_t, p_{t+1}, z) to choose consumption. Inner loop (N_p): simulate with
current p-trajectory, gradient step, then set p_t <- r_t (realized). Outer loop:
N_epoch epochs.

══════════════════════════════════════════════════════════════════════════════
ASSUMPTIONS TO RE-CONFIRM (需重新確認的假設)
══════════════════════════════════════════════════════════════════════════════
1. First inner iter (k=0): No prior p-trajectory from simulation → assume
   P_{t+1}=P_t. Use π(b,y,r_t,r_t,z) — same as baseline Huggett.
   第一次內層迭代：無先前模擬的 p 軌跡，假設 P_{t+1}=P_t。

2. Last period (t=T-1): No realized p_{t+1} (no period T) → use π(b,y,r_t,r_t,z).
   最後一期：無實現的 p_{t+1}，使用 π(b,y,r_t,r_t,z)。

3. p_trajectory: List of indices into r_grid; p_trajectory[t+1] is the index
   used at time t for consumption (expected next price).
   p 軌跡：r_grid 的索引；p_trajectory[t+1] 為 t 期消費時使用的預期下期價格索引。

4. G0: Use an invariant distribution at mid (z, r) under the diagonal
   same-price policy slice as the default initial distribution.
   G0 預設使用中間 (z, r) 下、沿對角 same-price policy slice 的不變分佈。

5. Market clearing with ip_consume: When agents consume using expected p_{t+1}
   (index ip_consume), we find r_t such that S(r_t)=B given that consumption.
   當 ip_consume 設定時，在給定該消費行為下求市場出清 r_t。

Dependencies: numpy, scipy, jax, jaxlib, matplotlib
  pip install numpy scipy jax jaxlib matplotlib

Usage:
  python forward_looking_hugget_jax.py [--epochs N] [--quick] [--out_dir DIR]
"""
from __future__ import annotations

import argparse
import os
import time
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

import jax
import jax.numpy as jnp
from jax import lax


# ---------- Calibration ----------
def get_calibration(quick: bool = False) -> dict:
    beta, sigma = 0.96, 2.0
    rho_y, nu_y = 0.6, 0.2
    rho_z, nu_z = 0.9, 0.02
    B, b_min = 0.0, -1.0
    nb, b_max = 200, 50.0
    ny, nr, nz = 3, 20, 30
    r_min, r_max = 0.01, 0.06
    c_min = 1e-3
    e_trunc = 1e-3
    if quick:
        N_epoch_outer, N_p, T_traj, N_z_test, N_sample, N_theta_per_p = 10, 3, 20, 2, 4, 2
    else:
        N_epoch_outer, N_p, T_traj, N_z_test, N_sample, N_theta_per_p = 120, 10, 170, 4, 20, 5
    lr_ini = 1e-3
    p_damping = 0.1
    p_init = 0.038
    return {
        "beta": beta, "sigma": sigma, "rho_y": rho_y, "nu_y": nu_y,
        "rho_z": rho_z, "nu_z": nu_z, "B": B, "b_min": b_min,
        "nb": nb, "b_max": b_max, "ny": ny, "nr": nr, "nz": nz,
        "r_min": r_min, "r_max": r_max, "c_min": c_min, "e_trunc": e_trunc,
        "N_epoch_outer": N_epoch_outer, "N_p": N_p, "T_traj": T_traj,
        "N_z_test": N_z_test, "N_sample": N_sample, "N_theta_per_p": N_theta_per_p,
        "lr_ini": lr_ini, "p_damping": p_damping, "p_init": p_init,
    }


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
    """θ -> c = clamp(θ, c_min). Shape (nz, nr_cur, nr_next, nb, ny)."""
    return jnp.maximum(theta, c_min_val)


def u_jax(c: jnp.ndarray, sig: float, c_min_val: float) -> jnp.ndarray:
    c = jnp.maximum(c, c_min_val)
    return jnp.where(
        jnp.abs(sig - 1.0) < 1e-8,
        jnp.log(c),
        (c ** (1 - sig)) / (1 - sig),
    )


def init_theta(b_grid, y_grid, z_grid, r_grid, save_frac=0.2, c_min_val=1e-3):
    nb_t, ny_t = b_grid.shape[0], y_grid.shape[0]
    nz_t, nr_t = z_grid.shape[0], r_grid.shape[0]
    b_flat = jnp.repeat(b_grid, ny_t)
    y_flat = jnp.tile(y_grid, nb_t)
    cash = (
        b_flat[None, None, :] * (1 + r_grid[None, :, None])
        + y_flat[None, None, :] * z_grid[:, None, None]
    )
    cash = cash.reshape(nz_t, nr_t, nb_t, ny_t)
    c_base = jnp.maximum((1 - save_frac) * cash, c_min_val)  # (nz, nr_cur, nb, ny)
    # Forward-looking policy: duplicate base across expected-next-price dimension.
    return jnp.broadcast_to(c_base[:, :, None, :, :], (nz_t, nr_t, nr_t, nb_t, ny_t))


def _G_to_mat(G, nb, ny):
    return jnp.reshape(G, (nb, ny))


def interp_weights_1d(grid, x):
    """Linear interpolation weights on a 1D grid for a scalar x."""
    n = grid.shape[0]
    if n == 1:
        z0 = jnp.int32(0)
        return z0, z0, jnp.float32(0.0)
    x_clip = jnp.clip(x, grid[0], grid[-1])
    hi = jnp.clip(jnp.searchsorted(grid, x_clip, side="right"), 1, n - 1)
    lo = hi - 1
    denom = jnp.maximum(grid[hi] - grid[lo], 1e-20)
    w = (x_clip - grid[lo]) / denom
    return lo, hi, w


def interpolate_c_at_prices(theta, iz, r_cur_val, p_next_val, r_grid, c_min_val):
    """Bilinear interpolation over current r and expected next-period p."""
    c = theta_to_c(theta, c_min_val)
    ir_lo, ir_hi, w_r = interp_weights_1d(r_grid, r_cur_val)
    ip_lo, ip_hi, w_p = interp_weights_1d(r_grid, p_next_val)
    c00 = c[iz, ir_lo, ip_lo, :, :]
    c01 = c[iz, ir_lo, ip_hi, :, :]
    c10 = c[iz, ir_hi, ip_lo, :, :]
    c11 = c[iz, ir_hi, ip_hi, :, :]
    c_lo = (1.0 - w_p) * c00 + w_p * c01
    c_hi = (1.0 - w_p) * c10 + w_p * c11
    return (1.0 - w_r) * c_lo + w_r * c_hi


def update_G_from_c_and_r(c_t, r_star, G, b_grid, y_grid, z_grid, iz, Ty, nb, ny, b_min, b_max):
    """One-step G update using an interpolated policy slice and continuous r."""
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


def market_clearing_stats(theta, G, iz, b_grid, y_grid, z_grid, r_grid, ny, B, b_min, b_max, c_min_val,
                          p_next_val, use_same_r):
    """Continuous market-clearing via bracket interpolation on the current-r grid."""
    nr = r_grid.shape[0]
    nb_b = b_grid.shape[0]
    G_mat = _G_to_mat(G, nb_b, ny)
    z_val = z_grid[iz]
    c = theta_to_c(theta, c_min_val)  # (nz, nr_cur, nr_next, nb, ny)
    b_flat = jnp.repeat(b_grid, ny)
    y_flat = jnp.tile(y_grid, nb_b)
    resources = b_flat[:, None] * (1 + r_grid[None, :]) + (y_flat * z_val)[:, None]  # (nb*ny, nr)
    ir_idx = jnp.arange(nr, dtype=jnp.int32)

    def c_slice_same(_):
        return c[iz, ir_idx, ir_idx, :, :]

    def c_slice_given(p_val):
        ip_lo, ip_hi, w_p = interp_weights_1d(r_grid, p_val)
        c_lo = c[iz, ir_idx, ip_lo, :, :]
        c_hi = c[iz, ir_idx, ip_hi, :, :]
        return (1.0 - w_p) * c_lo + w_p * c_hi

    c_slice = lax.cond(use_same_r, c_slice_same, c_slice_given, p_next_val)
    c_slice = c_slice.transpose(1, 2, 0).reshape(nb_b * ny, nr)

    b_next_all = jnp.clip(resources - c_slice, b_min, b_max)
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
    S_star = S_lo + w_r * (S_hi - S_lo)
    residual = S_star - B
    return r_star, ir_lo, ir_hi, w_r, residual


def steady_state_G0(theta, b_grid, y_grid, z_grid, r_grid, Ty, nb, ny, nz_spg, nr_spg, b_min, b_max, c_min_val,
                    n_iter: int = 150, tol: float = 1e-6):
    """Invariant distribution at mid (z, r) under the diagonal same-price slice."""
    iz_mid = nz_spg // 2
    ir_mid = nr_spg // 2 if nr_spg > 0 else 0
    G = jnp.ones((nb, ny)) / (nb * ny)

    def cond(carry):
        G_cur, G_prev, it = carry
        return (it < n_iter) & (jnp.abs(G_cur - G_prev).max() >= tol)

    def body(carry):
        G_cur, _G_prev, it = carry
        c_t = theta_to_c(theta, c_min_val)[iz_mid, ir_mid, ir_mid, :, :]
        G_new = update_G_from_c_and_r(
            c_t, r_grid[ir_mid], G_cur,
            b_grid, y_grid, z_grid, iz_mid, Ty, nb, ny,
            b_min, b_max,
        )
        return (G_new, G_cur, it + 1)

    G_final, _, _ = jax.lax.while_loop(
        cond,
        body,
        (G, G + 1.0, 0),
    )
    return G_final.reshape(-1)


def init_p_path_values(T_traj: int, p_init_val: float) -> np.ndarray:
    """Initial expected-price path in levels, not grid indices."""
    return np.full((T_traj + 1,), p_init_val, dtype=np.float32)


def damped_update_p_path(p_old: np.ndarray, r_realized: np.ndarray, alpha: float) -> np.ndarray:
    """Damped fixed-point update p <- (1-alpha) p + alpha r."""
    p_target = np.array(p_old, copy=True)
    p_target[:-1] = r_realized
    p_target[-1] = r_realized[-1]
    return ((1.0 - alpha) * p_old + alpha * p_target).astype(np.float32)


def objective_one_trajectory(
    theta,
    z_trajectory: jnp.ndarray,  # (T,) int32
    p_path_values: jnp.ndarray,  # (T+1,) float32 - expected next prices in levels
    G0,
    b_grid, y_grid, z_grid, r_grid, Ty,
    nb, ny, nz_spg, nr_spg,
    beta, e_trunc, B, b_min, b_max, c_min_val, sigma,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Returns (L, r_realized, residuals, boundary_hits).
    Uses lax.scan for JIT compatibility.
    """
    T = z_trajectory.shape[0]

    def body_fn(carry, t):
        G_cur, L_cur = carry
        iz = z_trajectory[t]
        is_last = (t == T - 1)
        p_next_val = p_path_values[t + 1]

        r_t, ir_lo, ir_hi, w_r, residual = market_clearing_stats(
            theta, G_cur, iz, b_grid, y_grid, z_grid, r_grid, ny,
            B, b_min, b_max, c_min_val, p_next_val, is_last,
        )
        r_t = lax.stop_gradient(r_t)
        ir_lo = lax.stop_gradient(ir_lo)
        ir_hi = lax.stop_gradient(ir_hi)
        w_r = lax.stop_gradient(w_r)
        residual = lax.stop_gradient(residual)

        p_eval = jnp.where(is_last, r_t, p_next_val)
        c_t = interpolate_c_at_prices(theta, iz, r_t, p_eval, r_grid, c_min_val)
        weight = (beta ** t) * jnp.float32((beta ** t) >= e_trunc)
        term = weight * (lax.stop_gradient(G_cur) * u_jax(c_t, sigma, c_min_val)).sum()
        L_new = L_cur + term

        G_new = update_G_from_c_and_r(
            c_t, r_t, G_cur,
            b_grid, y_grid, z_grid, iz, Ty, nb, ny,
            b_min, b_max,
        )
        G_new = lax.stop_gradient(G_new)
        boundary_hit = jnp.float32((r_t <= r_grid[0] + 1e-12) | (r_t >= r_grid[-1] - 1e-12))
        return (G_new, L_new), (r_t, residual, boundary_hit)

    G = _G_to_mat(G0, nb, ny)
    L_init = jnp.float32(0.0)
    (G_final, L_n), (r_vals, residual_vals, boundary_hits) = lax.scan(
        body_fn,
        (G, L_init),
        jnp.arange(T),
    )
    return L_n, r_vals, residual_vals, boundary_hits


def generate_z_trajectory(T: int, nz_spg: int, Tz_np: np.ndarray) -> np.ndarray:
    """Sample z path. Returns (T,) int32 array of iz."""
    iz = np.random.randint(0, nz_spg)
    path = []
    for _ in range(T):
        path.append(iz)
        iz = np.random.choice(nz_spg, p=Tz_np[iz, :])
    return np.array(path, dtype=np.int32)


def run_inner_convergence(theta_flat, z_traj, N_p, r_grid_np, G0_steady, b_grid, y_grid, z_grid, r_grid, Ty,
                          nb_spg, ny, nz_spg, nr_spg, beta, e_trunc, B, b_min, b_max, c_min, sigma,
                          p_init_val: float, p_damping: float):
    """Run p-trajectory fixed-point inner loop with fixed theta; return r paths and convergence distances."""
    p_path_vals = init_p_path_values(len(z_traj), p_init_val)
    losses = []
    r_paths = []
    residual_paths = []
    boundary_paths = []
    p_paths = []
    for k in range(N_p):
        L_val, r_vals, residual_vals, boundary_hits = objective_one_trajectory(
            theta_flat.reshape(nz_spg, nr_spg, nr_spg, nb_spg, ny),
            jnp.array(z_traj), jnp.array(p_path_vals, dtype=jnp.float32), G0_steady,
            b_grid, y_grid, z_grid, r_grid, Ty, nb_spg, ny, nz_spg, nr_spg,
            beta, e_trunc, B, b_min, b_max, c_min, sigma,
        )
        losses.append(float(L_val))
        r_np = np.asarray(r_vals)
        r_paths.append(r_np)
        residual_paths.append(np.asarray(residual_vals))
        boundary_paths.append(np.asarray(boundary_hits))
        p_paths.append(np.asarray(p_path_vals[:-1], dtype=float))
        p_path_vals = damped_update_p_path(p_path_vals, r_np, p_damping)
    losses = np.asarray(losses)
    r_paths = np.asarray(r_paths)  # (N_p, T)
    residual_paths = np.asarray(residual_paths)  # (N_p, T)
    boundary_paths = np.asarray(boundary_paths)  # (N_p, T)
    p_paths = np.asarray(p_paths)  # (N_p, T)
    if N_p >= 2:
        delta = np.max(np.abs(r_paths[1:] - r_paths[:-1]), axis=1)  # (N_p-1,)
    else:
        delta = np.zeros((0,), dtype=float)
    return losses, r_paths, residual_paths, boundary_paths, p_paths, delta


def main():
    parser = argparse.ArgumentParser(description="Forward-Looking Huggett in JAX")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--out_dir", type=str, default="forward_looking_hugget_output")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_sample", type=int, default=None, help="Number of z trajectories per epoch")
    parser.add_argument("--n_p", type=int, default=None, help="Inner iterations per trajectory")
    parser.add_argument("--theta_steps_per_p", type=int, default=None, help="Policy gradient steps with p fixed before one p update")
    parser.add_argument("--p_damping", type=float, default=None, help="Damping coefficient alpha in p <- (1-alpha)p + alpha r")
    parser.add_argument("--p_init", type=float, default=None, help="Initial expected-price level for the whole p path")
    args = parser.parse_args()

    cal = get_calibration(quick=args.quick)
    if args.epochs is not None:
        cal["N_epoch_outer"] = args.epochs
    if args.n_sample is not None:
        cal["N_sample"] = args.n_sample
    if args.n_p is not None:
        cal["N_p"] = args.n_p
    if args.theta_steps_per_p is not None:
        cal["N_theta_per_p"] = args.theta_steps_per_p
    if args.p_damping is not None:
        cal["p_damping"] = args.p_damping
    if args.p_init is not None:
        cal["p_init"] = args.p_init

    beta = cal["beta"]
    sigma = cal["sigma"]
    c_min = cal["c_min"]
    b_min = cal["b_min"]
    b_max = cal["b_max"]
    ny = cal["ny"]
    nb, nr, nz = cal["nb"], cal["nr"], cal["nz"]
    r_min, r_max = cal["r_min"], cal["r_max"]
    B = cal["B"]
    e_trunc = cal["e_trunc"]
    N_epoch_outer = cal["N_epoch_outer"]
    N_p = cal["N_p"]
    N_sample = cal["N_sample"]
    N_theta_per_p = cal["N_theta_per_p"]
    T_traj = cal["T_traj"]
    N_z_test = cal["N_z_test"]
    lr_ini = cal["lr_ini"]
    p_damping = cal["p_damping"]
    p_init_val = cal["p_init"]

    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)

    # Grids
    b_grid_np = np.linspace(b_min, b_max, nb)
    r_grid_np = np.linspace(r_min, r_max, nr)
    y_grid_np, Ty_np = tauchen_ar1(cal["rho_y"], cal["nu_y"], ny, m=3, mean=1.0)
    invariant_y = np.linalg.matrix_power(Ty_np.T, 200)[:, 0]
    y_grid_np = y_grid_np / (y_grid_np @ invariant_y)
    log_z_grid, Tz_np = tauchen_ar1(cal["rho_z"], cal["nu_z"], nz)
    z_grid_np = np.exp(log_z_grid)
    invariant_z = np.linalg.matrix_power(Tz_np.T, 200)[:, 0]
    z_grid_np = z_grid_np / (z_grid_np @ invariant_z)

    # Use full Huggett grids for forward-looking training.
    nb_spg, nr_spg, nz_spg = nb, nr, nz
    Tz_sub = Tz_np
    b_grid = jnp.array(np.linspace(b_min, b_max, nb_spg))
    r_grid = jnp.array(np.linspace(r_min, r_max, nr_spg))
    z_grid = jnp.array(z_grid_np)
    y_grid = jnp.array(y_grid_np)
    Ty = jnp.array(Ty_np)
    r_grid_spg_np = np.array(r_grid)

    theta = init_theta(b_grid, y_grid, z_grid, r_grid, save_frac=0.2, c_min_val=c_min)
    G0_steady = steady_state_G0(
        theta, b_grid, y_grid, z_grid, r_grid, Ty,
        nb_spg, ny, nz_spg, nr_spg, b_min, b_max, c_min,
    )
    theta = theta.reshape(-1)  # flatten for Adam

    def loss_fn(theta_flat, z_traj, p_path_vals):
        theta_mat = theta_flat.reshape(nz_spg, nr_spg, nr_spg, nb_spg, ny)
        L, _, _, _ = objective_one_trajectory(
            theta_mat, z_traj, p_path_vals, G0_steady,
            b_grid, y_grid, z_grid, r_grid, Ty,
            nb_spg, ny, nz_spg, nr_spg,
            beta, e_trunc, B, b_min, b_max, c_min, sigma,
        )
        return -L

    def adam_step(theta_flat, m, v, g, step, lr, beta1=0.9, beta2=0.999, eps=1e-8):
        m_new = beta1 * m + (1 - beta1) * g
        v_new = beta2 * v + (1 - beta2) * (g ** 2)
        m_hat = m_new / (1 - beta1 ** (step + 1))
        v_hat = v_new / (1 - beta2 ** (step + 1))
        theta_new = theta_flat + lr * m_hat / (jnp.sqrt(v_hat) + eps)
        return theta_new, m_new, v_new

    grad_fn = jax.jit(jax.grad(loss_fn, argnums=0))

    m = jnp.zeros_like(theta)
    v = jnp.zeros_like(theta)
    loss_hist = []
    mean_abs_resid_hist = []
    boundary_share_hist = []

    print("Forward-Looking Huggett JAX: N_sample=%d, N_p=%d, N_theta_per_p=%d, N_epoch=%d, T_traj=%d"
          % (N_sample, N_p, N_theta_per_p, N_epoch_outer, T_traj))
    print("JAX device:", jax.default_backend())
    print("G0 initialized from mid-(z,r) invariant distribution")
    print("Damped p update: alpha=%.3f, p_init=%.4f" % (p_damping, p_init_val))

    start = time.perf_counter()
    global_step = 0
    for epoch in range(N_epoch_outer):
        epoch_losses = []
        epoch_abs_resid = []
        epoch_boundary_share = []
        for n in range(N_sample):
            # Same z path across inner loop for this sample.
            z_trajectory = generate_z_trajectory(T_traj, nz_spg, Tz_sub)
            z_trajectory_jax = jnp.array(z_trajectory, dtype=jnp.int32)
            p_path_values = init_p_path_values(T_traj, p_init_val)
            for k in range(N_p):
                p_path_jax = jnp.array(p_path_values, dtype=jnp.float32)
                for _ in range(N_theta_per_p):
                    g = grad_fn(theta, z_trajectory_jax, p_path_jax)
                    theta, m, v = adam_step(theta, m, v, g, global_step, lr_ini)
                    global_step += 1
                theta_mat = theta.reshape(nz_spg, nr_spg, nr_spg, nb_spg, ny)
                L_val, r_vals, residual_vals, boundary_hits = objective_one_trajectory(
                    theta_mat,
                    z_trajectory_jax, p_path_jax, G0_steady,
                    b_grid, y_grid, z_grid, r_grid, Ty,
                    nb_spg, ny, nz_spg, nr_spg,
                    beta, e_trunc, B, b_min, b_max, c_min, sigma,
                )
                epoch_losses.append(float(-L_val))
                epoch_abs_resid.append(float(np.mean(np.abs(np.asarray(residual_vals)))))
                epoch_boundary_share.append(float(np.mean(np.asarray(boundary_hits))))
                r_realized_np = np.asarray(r_vals)
                p_path_values = damped_update_p_path(p_path_values, r_realized_np, p_damping)
        loss_hist.append(np.mean(epoch_losses))
        mean_abs_resid_hist.append(np.mean(epoch_abs_resid))
        boundary_share_hist.append(np.mean(epoch_boundary_share))
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                "Epoch %d, mean L = %.4f, mean |mc residual| = %.4e, boundary share = %.3f"
                % (epoch + 1, loss_hist[-1], mean_abs_resid_hist[-1], boundary_share_hist[-1])
            )

    elapsed = time.perf_counter() - start
    print("Training done in %.2f s" % elapsed)

    # Convergence diagnostics with fixed theta (no gradient update in inner loop)
    test_z_trajectories = [generate_z_trajectory(T_traj, nz_spg, Tz_sub) for _ in range(N_z_test)]
    conv_results = []
    for traj_idx, z_traj in enumerate(test_z_trajectories):
        losses_conv, r_paths_conv, residual_paths_conv, boundary_paths_conv, p_paths_conv, delta_conv = run_inner_convergence(
            theta, z_traj, N_p, r_grid_spg_np, G0_steady, b_grid, y_grid, z_grid, r_grid, Ty,
            nb_spg, ny, nz_spg, nr_spg, beta, e_trunc, B, b_min, b_max, c_min, sigma,
            p_init_val, p_damping,
        )
        conv_results.append({
            "losses": losses_conv,
            "z_traj": z_traj,
            "r_paths": r_paths_conv,
            "residual_paths": residual_paths_conv,
            "boundary_paths": boundary_paths_conv,
            "p_paths": p_paths_conv,
            "delta": delta_conv,
        })
        print(
            "Test trajectory %d: L first=%.4f, last=%.4f, last |mc residual|=%.4e, last boundary share=%.3f"
            % (
                traj_idx + 1,
                float(losses_conv[0]),
                float(losses_conv[-1]),
                float(np.mean(np.abs(residual_paths_conv[-1]))),
                float(np.mean(boundary_paths_conv[-1])),
            )
        )

    # Save outputs
    os.makedirs(args.out_dir, exist_ok=True)
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.unicode_minus"] = False

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    for traj_idx, cres in enumerate(conv_results):
        axs[0, 0].plot(range(N_p), cres["losses"], label="z_traj %d" % (traj_idx + 1))
    axs[0, 0].set_xlabel("Inner iteration k")
    axs[0, 0].set_ylabel("L(θ)")
    axs[0, 0].set_title("Fixed-theta loss over inner loop")
    axs[0, 0].legend()
    axs[0, 0].grid(alpha=0.3)

    t_axis = np.arange(T_traj)
    for traj_idx, cres in enumerate(conv_results):
        axs[0, 1].plot(t_axis, cres["r_paths"][0], alpha=0.7, label="z_traj%d, k=0" % (traj_idx + 1))
        axs[0, 1].plot(t_axis, cres["r_paths"][-1], alpha=0.7, linestyle="--", label="z_traj%d, k=%d" % (traj_idx + 1, N_p - 1))
    axs[0, 1].set_xlabel("Time t")
    axs[0, 1].set_ylabel("r_t (realized)")
    axs[0, 1].set_title("Realized r_t: first vs last inner iteration")
    axs[0, 1].legend(ncol=2, fontsize=8)
    axs[0, 1].grid(alpha=0.3)

    axs[1, 0].plot(loss_hist, color="tab:blue")
    axs[1, 0].set_xlabel("Epoch")
    axs[1, 0].set_ylabel("Mean L")
    axs[1, 0].set_title("Training: mean L per epoch")
    axs[1, 0].grid(alpha=0.3)

    for traj_idx, cres in enumerate(conv_results):
        if cres["delta"].size > 0:
            axs[1, 1].plot(np.arange(1, N_p), cres["delta"], "-o", label="traj %d" % (traj_idx + 1))
    axs[1, 1].set_xlabel("inner iteration k")
    axs[1, 1].set_ylabel("max_t |r^(k)-r^(k-1)|")
    axs[1, 1].set_title("Inner-loop trajectory convergence distance")
    axs[1, 1].legend(fontsize=8)
    axs[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "forward_looking_results.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Plot all inner r trajectories for each fixed z path (convergence visibility)
    for traj_idx, cres in enumerate(conv_results):
        r_paths = cres["r_paths"]  # (N_p, T)
        residual_paths = cres["residual_paths"]  # (N_p, T)
        delta = cres["delta"]      # (N_p-1,)
        t_axis = np.arange(r_paths.shape[1])

        fig, axs = plt.subplots(2, 1, figsize=(10, 8))
        for k in range(r_paths.shape[0]):
            alpha = 0.25 + 0.75 * (k + 1) / max(r_paths.shape[0], 1)
            axs[0].plot(t_axis, r_paths[k], alpha=alpha, linewidth=1.2, label="k=%d" % k)
        axs[0].set_xlabel("t")
        axs[0].set_ylabel("r_t")
        axs[0].set_title("All inner realized r trajectories (fixed z path %d)" % (traj_idx + 1))
        axs[0].grid(alpha=0.3)
        axs[0].legend(ncol=4, fontsize=8)

        if delta.size > 0:
            axs[1].plot(np.arange(1, r_paths.shape[0]), delta, "-o")
        axs[1].set_xlabel("inner iteration k")
        axs[1].set_ylabel("max_t |r^(k)-r^(k-1)|")
        axs[1].set_title("Inner-loop trajectory convergence distance")
        axs[1].grid(alpha=0.3)

        plt.tight_layout()
        fig.savefig(os.path.join(args.out_dir, "trajectory_convergence_traj%02d.png" % (traj_idx + 1)),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

        np.save(os.path.join(args.out_dir, "conv_r_paths_traj%02d.npy" % (traj_idx + 1)), r_paths)
        np.save(os.path.join(args.out_dir, "conv_residual_paths_traj%02d.npy" % (traj_idx + 1)), residual_paths)
        np.save(os.path.join(args.out_dir, "conv_boundary_paths_traj%02d.npy" % (traj_idx + 1)), cres["boundary_paths"])
        np.save(os.path.join(args.out_dir, "conv_p_paths_traj%02d.npy" % (traj_idx + 1)), cres["p_paths"])
        np.save(os.path.join(args.out_dir, "conv_delta_traj%02d.npy" % (traj_idx + 1)), delta)

    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    ax.plot(loss_hist, color="tab:blue")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean L(θ)")
    ax.set_title("Forward-Looking Huggett: training loss")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "loss_curve.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    for traj_idx, cres in enumerate(conv_results):
        ax.plot(t_axis, cres["r_paths"][0], alpha=0.7, label="traj%d, first" % (traj_idx + 1))
        ax.plot(t_axis, cres["r_paths"][-1], alpha=0.7, linestyle="--", label="traj%d, last" % (traj_idx + 1))
    ax.set_xlabel("t")
    ax.set_ylabel("r_t")
    ax.set_title("First vs last inner realized r trajectories")
    ax.grid(alpha=0.3)
    ax.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "final_sim_first_vs_last.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    for traj_idx, cres in enumerate(conv_results):
        axs[0].plot(np.abs(cres["residual_paths"][-1]), label="traj %d" % (traj_idx + 1))
        axs[1].plot(np.arange(1, N_p + 1), cres["boundary_paths"].mean(axis=1), "-o", label="traj %d" % (traj_idx + 1))
    axs[0].set_xlabel("t")
    axs[0].set_ylabel("|market-clearing residual|")
    axs[0].set_title("Last inner iteration residual path")
    axs[0].grid(alpha=0.3)
    axs[0].legend(fontsize=8)
    axs[1].set_xlabel("inner iteration k")
    axs[1].set_ylabel("boundary-hit share")
    axs[1].set_title("Boundary-hit share across inner loop")
    axs[1].grid(alpha=0.3)
    axs[1].legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "final_sim_diagnostics.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    with open(os.path.join(args.out_dir, "loss_hist.txt"), "w") as f:
        f.write("\n".join(map(str, loss_hist)))
    with open(os.path.join(args.out_dir, "diagnostics_summary.txt"), "w") as f:
        f.write("g0_type=steady_state_mid_same_r\n")
        f.write("train_final_mean_loss=%.8f\n" % float(loss_hist[-1]))
        f.write("train_final_mean_abs_residual=%.8e\n" % float(mean_abs_resid_hist[-1]))
        f.write("train_final_boundary_share=%.8f\n" % float(boundary_share_hist[-1]))
        if conv_results:
            all_r = np.asarray([cres["r_paths"] for cres in conv_results])
            all_resid = np.asarray([cres["residual_paths"] for cres in conv_results])
            all_boundary = np.asarray([cres["boundary_paths"] for cres in conv_results])
            last_r = all_r[:, -1, :]
            last_resid = all_resid[:, -1, :]
            last_boundary = all_boundary[:, -1, :]
            f.write("conv_all_r_min=%.8f\n" % float(all_r.min()))
            f.write("conv_all_r_max=%.8f\n" % float(all_r.max()))
            f.write("conv_last_r_min=%.8f\n" % float(last_r.min()))
            f.write("conv_last_r_max=%.8f\n" % float(last_r.max()))
            f.write("conv_all_mean_abs_residual=%.8e\n" % float(np.mean(np.abs(all_resid))))
            f.write("conv_last_mean_abs_residual=%.8e\n" % float(np.mean(np.abs(last_resid))))
            f.write("conv_last_max_abs_residual=%.8e\n" % float(np.max(np.abs(last_resid))))
            f.write("conv_all_boundary_share=%.8f\n" % float(all_boundary.mean()))
            f.write("conv_last_boundary_share=%.8f\n" % float(last_boundary.mean()))
            f.write("conv_all_upper_share=%.8f\n" % float((all_r >= r_grid_np[-1] - 1e-10).mean()))
            f.write("conv_last_upper_share=%.8f\n" % float((last_r >= r_grid_np[-1] - 1e-10).mean()))
            f.write("conv_all_lower_share=%.8f\n" % float((all_r <= r_grid_np[0] + 1e-10).mean()))
            f.write("conv_last_lower_share=%.8f\n" % float((last_r <= r_grid_np[0] + 1e-10).mean()))
            for traj_idx, cres in enumerate(conv_results, start=1):
                last_delta = float(cres["delta"][-1]) if cres["delta"].size > 0 else 0.0
                f.write("traj_%02d_last_delta=%.8f\n" % (traj_idx, last_delta))
                f.write("traj_%02d_last_mean_abs_residual=%.8e\n" % (traj_idx, float(np.mean(np.abs(cres["residual_paths"][-1])))))
                f.write("traj_%02d_last_boundary_share=%.8f\n" % (traj_idx, float(np.mean(cres["boundary_paths"][-1]))))
    np.save(os.path.join(args.out_dir, "mean_abs_residual_hist.npy"), np.asarray(mean_abs_resid_hist))
    np.save(os.path.join(args.out_dir, "boundary_share_hist.npy"), np.asarray(boundary_share_hist))
    np.save(os.path.join(args.out_dir, "final_sim_r_paths.npy"), np.asarray([cres["r_paths"] for cres in conv_results]))
    np.save(os.path.join(args.out_dir, "final_sim_residual_paths.npy"), np.asarray([cres["residual_paths"] for cres in conv_results]))
    np.save(os.path.join(args.out_dir, "final_sim_boundary_paths.npy"), np.asarray([cres["boundary_paths"] for cres in conv_results]))
    np.save(os.path.join(args.out_dir, "theta_final.npy"), np.array(theta.reshape(nz_spg, nr_spg, nr_spg, nb_spg, ny)))
    print("Saved %s" % args.out_dir)


if __name__ == "__main__":
    main()
