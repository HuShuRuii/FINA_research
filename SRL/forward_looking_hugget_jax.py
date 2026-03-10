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

4. G0: Fixed at uniform (1/(nb*ny)). No steady-state warm-up.
   G0 固定為均勻分佈，無穩態預熱。

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
        N_epoch_outer, N_p, T_traj, N_z_test, N_sample = 10, 3, 20, 2, 4
    else:
        N_epoch_outer, N_p, T_traj, N_z_test, N_sample = 120, 10, 170, 4, 20
    lr_ini = 1e-3
    return {
        "beta": beta, "sigma": sigma, "rho_y": rho_y, "nu_y": nu_y,
        "rho_z": rho_z, "nu_z": nu_z, "B": B, "b_min": b_min,
        "nb": nb, "b_max": b_max, "ny": ny, "nr": nr, "nz": nz,
        "r_min": r_min, "r_max": r_max, "c_min": c_min, "e_trunc": e_trunc,
        "N_epoch_outer": N_epoch_outer, "N_p": N_p, "T_traj": T_traj,
        "N_z_test": N_z_test, "N_sample": N_sample, "lr_ini": lr_ini,
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


def P_star_best_ir(theta, G, iz, b_grid, y_grid, z_grid, r_grid, ny, B, b_min, b_max, c_min_val, ip_next, use_same_r):
    """Market-clearing current-r index.
    If use_same_r: use c[iz, ir, ir, :, :].
    Else: use c[iz, ir, ip_next, :, :].
    """
    nr = r_grid.shape[0]
    nb_b = b_grid.shape[0]
    G_mat = _G_to_mat(G, nb_b, ny)
    z_val = z_grid[iz]
    c = theta_to_c(theta, c_min_val)  # (nz, nr_cur, nr_next, nb, ny)
    b_flat = jnp.repeat(b_grid, ny)
    y_flat = jnp.tile(y_grid, nb_b)
    resources = b_flat[:, None] * (1 + r_grid[None, :]) + (y_flat * z_val)[:, None]  # (nb*ny, nr)
    ir_idx = jnp.arange(nr, dtype=jnp.int32)
    ip_use = jnp.where(use_same_r, ir_idx, jnp.full((nr,), jnp.int32(ip_next)))
    c_slice = c[iz, ir_idx, ip_use, :, :].transpose(1, 2, 0).reshape(nb_b * ny, nr)

    b_next_all = jnp.clip(resources - c_slice, b_min, b_max)
    b_next_all = b_next_all.reshape(nb_b, ny, nr)
    S_all = (G_mat[:, :, None] * b_next_all).sum(axis=(0, 1))
    best_ir = jnp.argmin(jnp.abs(S_all - B))
    return jnp.where(nr == 1, jnp.int32(0), best_ir)


def update_G_pi_direct(theta, G, iz, ir, ip_next, use_same_r, b_grid, y_grid, z_grid, r_grid, Ty, nb, ny, b_min, b_max, c_min_val):
    """G update with forward-looking policy slice."""
    G = _G_to_mat(G, nb, ny)
    c = theta_to_c(theta, c_min_val)
    ip_for_c = jnp.where(use_same_r, ir, ip_next)
    c_val = c[iz, ir, ip_for_c, :, :].ravel()
    b_next = (1 + r_grid[ir]) * jnp.repeat(b_grid, ny) + jnp.tile(y_grid, nb) * z_grid[iz] - c_val
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


def value_to_ir(p_val: float, r_grid_np: np.ndarray) -> int:
    """Map scalar price to grid index."""
    idx = np.clip(np.searchsorted(r_grid_np, p_val, side="right") - 1, 0, len(r_grid_np) - 1)
    return int(idx)


def make_p_trajectory_indices(r_realized: np.ndarray, r_grid_np: np.ndarray, T_traj: int) -> List[int]:
    """Convert realized r (length T_traj) to p_trajectory (length T_traj+1) as indices."""
    idxs = [value_to_ir(float(r), r_grid_np) for r in r_realized]
    idxs.append(idxs[-1] if idxs else 0)
    return idxs


def objective_one_trajectory(
    theta,
    z_trajectory: jnp.ndarray,  # (T,) int32
    p_trajectory: jnp.ndarray,  # (T+1,) int32 - indices into r_grid
    G0,
    b_grid, y_grid, z_grid, r_grid, Ty,
    nb, ny, nz_spg, nr_spg,
    beta, e_trunc, B, b_min, b_max, c_min_val, sigma,
    assume_pt1_equals_pt: bool,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Returns (L, r_realized) where r_realized is (T,) array of realized r values.
    Uses lax.scan for JIT compatibility; p_trajectory and assume_pt1 must be static for grad.
    """
    T = z_trajectory.shape[0]

    def body_fn(carry, t):
        G_cur, L_cur = carry
        iz = z_trajectory[t]
        is_last = (t == T - 1)
        use_same_r = assume_pt1_equals_pt or is_last
        ip_next = p_trajectory[t + 1]

        best_ir = P_star_best_ir(
            theta, G_cur, iz, b_grid, y_grid, z_grid, r_grid, ny,
            B, b_min, b_max, c_min_val, ip_next, use_same_r,
        )
        best_ir = lax.stop_gradient(best_ir)
        r_t = r_grid[best_ir]
        ip_for_c = jnp.where(use_same_r, best_ir, ip_next)

        c = theta_to_c(theta, c_min_val)
        c_t = c[iz, best_ir, ip_for_c, :, :]
        weight = (beta ** t) * jnp.float32((beta ** t) >= e_trunc)
        term = weight * (lax.stop_gradient(G_cur) * u_jax(c_t, sigma, c_min_val)).sum()
        L_new = L_cur + term

        G_new = update_G_pi_direct(
            theta, G_cur, iz, best_ir, ip_next, use_same_r,
            b_grid, y_grid, z_grid, r_grid, Ty, nb, ny,
            b_min, b_max, c_min_val,
        )
        G_new = lax.stop_gradient(G_new)
        return (G_new, L_new), r_t

    G = _G_to_mat(G0, nb, ny)
    L_init = jnp.float32(0.0)
    (G_final, L_n), r_vals = lax.scan(
        body_fn,
        (G, L_init),
        jnp.arange(T),
    )
    return L_n, r_vals


def generate_z_trajectory(T: int, nz_spg: int, Tz_np: np.ndarray) -> np.ndarray:
    """Sample z path. Returns (T,) int32 array of iz."""
    iz = np.random.randint(0, nz_spg)
    path = []
    for _ in range(T):
        path.append(iz)
        iz = np.random.choice(nz_spg, p=Tz_np[iz, :])
    return np.array(path, dtype=np.int32)


def run_inner_convergence(theta_flat, z_traj, N_p, r_grid_np, G0_steady, b_grid, y_grid, z_grid, r_grid, Ty,
                          nb_spg, ny, nz_spg, nr_spg, beta, e_trunc, B, b_min, b_max, c_min, sigma):
    """Run p-trajectory fixed-point inner loop with fixed theta; return r paths and convergence distances."""
    p_traj = [nr_spg // 2] * (len(z_traj) + 1)
    r_paths = []
    p_paths = []
    for k in range(N_p):
        assume_pt1 = (k == 0)
        L_val, r_vals = objective_one_trajectory(
            theta_flat.reshape(nz_spg, nr_spg, nr_spg, nb_spg, ny),
            jnp.array(z_traj), jnp.array(p_traj, dtype=jnp.int32), G0_steady,
            b_grid, y_grid, z_grid, r_grid, Ty, nb_spg, ny, nz_spg, nr_spg,
            beta, e_trunc, B, b_min, b_max, c_min, sigma, assume_pt1,
        )
        _ = L_val
        r_np = np.asarray(r_vals)
        r_paths.append(r_np)
        p_paths.append(np.asarray(p_traj[:-1], dtype=int))
        p_traj = make_p_trajectory_indices(r_np, r_grid_np, len(z_traj))
    r_paths = np.asarray(r_paths)  # (N_p, T)
    p_paths = np.asarray(p_paths)  # (N_p, T)
    if N_p >= 2:
        delta = np.max(np.abs(r_paths[1:] - r_paths[:-1]), axis=1)  # (N_p-1,)
    else:
        delta = np.zeros((0,), dtype=float)
    return r_paths, p_paths, delta


def main():
    parser = argparse.ArgumentParser(description="Forward-Looking Huggett in JAX")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--out_dir", type=str, default="forward_looking_hugget_output")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_sample", type=int, default=None, help="Number of z trajectories per epoch")
    parser.add_argument("--n_p", type=int, default=None, help="Inner iterations per trajectory")
    args = parser.parse_args()

    cal = get_calibration(quick=args.quick)
    if args.epochs is not None:
        cal["N_epoch_outer"] = args.epochs
    if args.n_sample is not None:
        cal["N_sample"] = args.n_sample
    if args.n_p is not None:
        cal["N_p"] = args.n_p

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
    T_traj = cal["T_traj"]
    N_z_test = cal["N_z_test"]
    lr_ini = cal["lr_ini"]

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

    G0_steady = jnp.ones(nb_spg * ny) / (nb_spg * ny)
    theta = init_theta(b_grid, y_grid, z_grid, r_grid, save_frac=0.2, c_min_val=c_min)
    theta = theta.reshape(-1)  # flatten for Adam

    def loss_fn(theta_flat, z_traj, p_traj, assume_pt1):
        theta_mat = theta_flat.reshape(nz_spg, nr_spg, nr_spg, nb_spg, ny)
        L, _ = objective_one_trajectory(
            theta_mat, z_traj, p_traj, G0_steady,
            b_grid, y_grid, z_grid, r_grid, Ty,
            nb_spg, ny, nz_spg, nr_spg,
            beta, e_trunc, B, b_min, b_max, c_min, sigma,
            assume_pt1,
        )
        return -L

    def adam_step(theta_flat, m, v, g, step, lr, beta1=0.9, beta2=0.999, eps=1e-8):
        m_new = beta1 * m + (1 - beta1) * g
        v_new = beta2 * v + (1 - beta2) * (g ** 2)
        m_hat = m_new / (1 - beta1 ** (step + 1))
        v_hat = v_new / (1 - beta2 ** (step + 1))
        theta_new = theta_flat + lr * m_hat / (jnp.sqrt(v_hat) + eps)
        return theta_new, m_new, v_new

    grad_fn = jax.jit(jax.grad(loss_fn, argnums=0), static_argnums=(3,))

    m = jnp.zeros_like(theta)
    v = jnp.zeros_like(theta)
    loss_hist = []
    p_trajectories_by_epoch = []

    print("Forward-Looking Huggett JAX: N_sample=%d, N_p=%d, N_epoch=%d, T_traj=%d" % (N_sample, N_p, N_epoch_outer, T_traj))
    print("JAX device:", jax.default_backend())

    start = time.perf_counter()
    global_step = 0
    for epoch in range(N_epoch_outer):
        epoch_losses = []
        epoch_pt = []
        for n in range(N_sample):
            # Same z path across inner loop for this sample.
            z_trajectory = generate_z_trajectory(T_traj, nz_spg, Tz_sub)
            p_trajectory = [nr_spg // 2] * (T_traj + 1)
            for k in range(N_p):
                assume_pt1 = (k == 0)
                theta_mat = theta.reshape(nz_spg, nr_spg, nr_spg, nb_spg, ny)
                L_val, r_vals = objective_one_trajectory(
                    theta_mat,
                    jnp.array(z_trajectory), jnp.array(p_trajectory, dtype=jnp.int32), G0_steady,
                    b_grid, y_grid, z_grid, r_grid, Ty,
                    nb_spg, ny, nz_spg, nr_spg,
                    beta, e_trunc, B, b_min, b_max, c_min, sigma,
                    assume_pt1,
                )
                g = grad_fn(theta, jnp.array(z_trajectory), jnp.array(p_trajectory, dtype=jnp.int32), assume_pt1)
                theta, m, v = adam_step(theta, m, v, g, global_step, lr_ini)
                global_step += 1
                epoch_losses.append(float(-L_val))
                epoch_pt.append(list(p_trajectory))
                r_realized_np = np.asarray(r_vals)
                p_trajectory = make_p_trajectory_indices(r_realized_np, r_grid_spg_np, T_traj)
        loss_hist.append(np.mean(epoch_losses))
        p_trajectories_by_epoch.append(epoch_pt[-N_p:] if len(epoch_pt) >= N_p else epoch_pt)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print("Epoch %d, mean L = %.4f" % (epoch + 1, loss_hist[-1]))

    elapsed = time.perf_counter() - start
    print("Training done in %.2f s" % elapsed)

    # Test on fixed trajectories
    test_z_trajectories = [generate_z_trajectory(T_traj, nz_spg, Tz_sub) for _ in range(N_z_test)]
    theta_test = theta.copy()
    m_t, v_t = jnp.zeros_like(theta_test), jnp.zeros_like(theta_test)
    test_results = []
    for traj_idx, z_traj in enumerate(test_z_trajectories):
        p_traj = [nr_spg // 2] * (T_traj + 1)
        losses_inner = []
        r_paths_inner = []
        for k in range(N_p):
            assume_pt1 = (k == 0)
            L_val, r_vals = objective_one_trajectory(
                theta_test.reshape(nz_spg, nr_spg, nr_spg, nb_spg, ny),
                jnp.array(z_traj), jnp.array(p_traj, dtype=jnp.int32), G0_steady,
                b_grid, y_grid, z_grid, r_grid, Ty,
                nb_spg, ny, nz_spg, nr_spg,
                beta, e_trunc, B, b_min, b_max, c_min, sigma,
                assume_pt1,
            )
            g = grad_fn(theta_test, jnp.array(z_traj), jnp.array(p_traj, dtype=jnp.int32), assume_pt1)
            theta_test, m_t, v_t = adam_step(theta_test, m_t, v_t, g, k, lr_ini)
            losses_inner.append(float(-L_val))
            r_paths_inner.append(list(np.asarray(r_vals)))
            p_traj = make_p_trajectory_indices(np.asarray(r_vals), r_grid_spg_np, T_traj)
        test_results.append({"losses": losses_inner, "r_paths": r_paths_inner})
        print("Test trajectory %d: L first=%.4f, last=%.4f" % (traj_idx + 1, losses_inner[0], losses_inner[-1]))

    # Convergence diagnostics with fixed theta (no gradient update in inner loop)
    conv_results = []
    for traj_idx, z_traj in enumerate(test_z_trajectories):
        r_paths_conv, p_paths_conv, delta_conv = run_inner_convergence(
            theta, z_traj, N_p, r_grid_spg_np, G0_steady, b_grid, y_grid, z_grid, r_grid, Ty,
            nb_spg, ny, nz_spg, nr_spg, beta, e_trunc, B, b_min, b_max, c_min, sigma,
        )
        conv_results.append({
            "z_traj": z_traj,
            "r_paths": r_paths_conv,
            "p_paths": p_paths_conv,
            "delta": delta_conv,
        })

    # Save outputs
    os.makedirs(args.out_dir, exist_ok=True)
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.unicode_minus"] = False

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    for traj_idx, res in enumerate(test_results):
        axs[0, 0].plot(range(N_p), res["losses"], label="z_traj %d" % (traj_idx + 1))
    axs[0, 0].set_xlabel("Inner iteration k")
    axs[0, 0].set_ylabel("L(θ)")
    axs[0, 0].set_title("Loss over inner loop (different z-trajectories)")
    axs[0, 0].legend()
    axs[0, 0].grid(alpha=0.3)

    t_axis = np.arange(T_traj)
    for traj_idx, res in enumerate(test_results):
        for k in [0, N_p - 1]:
            axs[0, 1].plot(t_axis, res["r_paths"][k], alpha=0.7, label="z_traj%d, k=%d" % (traj_idx + 1, k))
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

    res0 = test_results[0]
    for k in range(N_p):
        axs[1, 1].plot(t_axis, res0["r_paths"][k], alpha=0.5 + 0.5 * (k / max(N_p, 1)), label="k=%d" % k)
    axs[1, 1].set_xlabel("Time t")
    axs[1, 1].set_ylabel("r_t")
    axs[1, 1].set_title("p_t trajectory evolution (z_traj 1, inner k=0..N_p-1)")
    axs[1, 1].legend(ncol=2, fontsize=8)
    axs[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "forward_looking_results.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Plot all inner r trajectories for each fixed z path (convergence visibility)
    for traj_idx, cres in enumerate(conv_results):
        r_paths = cres["r_paths"]  # (N_p, T)
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

    with open(os.path.join(args.out_dir, "loss_hist.txt"), "w") as f:
        f.write("\n".join(map(str, loss_hist)))
    np.save(os.path.join(args.out_dir, "theta_final.npy"), np.array(theta.reshape(nz_spg, nr_spg, nr_spg, nb_spg, ny)))
    print("Saved %s" % args.out_dir)


if __name__ == "__main__":
    main()
