#!/usr/bin/env python3
"""
Krusell–Smith (1998) SRL/SPG — JAX implementation (baseline port).

This script mirrors SRL/run_krusell_smith_cluster.py at a functional level:
- policy: theta(b,y,z,r,w) -> c
- prices: (r,w) from aggregate K and z
- objective: E[sum beta^t u(c_t)] with stop-gradient through prices
- distribution update: direct d update with Gaussian lottery in b

Usage:
  python krusell_smith_jax.py [--epochs N] [--quick] [--out_dir DIR]
"""
from __future__ import annotations

import argparse
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

import jax
import jax.numpy as jnp
import optax


def get_calibration(quick: bool = False) -> dict:
    beta, sigma = 0.95, 3.0
    alpha, delta = 0.36, 0.08
    rho_y, nu_y = 0.6, 0.2
    rho_z, nu_z = 0.9, 0.03
    ny = 3
    b_min, b_max = 0.0, 100.0
    nr, nw, nz = 30, 50, 30
    r_min, r_max = 0.02, 0.07
    w_min, w_max = 0.9, 1.5
    c_min = 1e-3
    T_trunc = 50
    if quick:
        N_epoch, N_warmup = 20, 5
        N_sample, e_converge = 8, 1e-3
    else:
        N_epoch, N_warmup = 200, 25
        N_sample, e_converge = 32, 3e-4
    return {
        "beta": beta, "sigma": sigma, "alpha": alpha, "delta": delta,
        "rho_y": rho_y, "nu_y": nu_y, "rho_z": rho_z, "nu_z": nu_z,
        "ny": ny, "b_min": b_min, "b_max": b_max,
        "nr": nr, "nw": nw, "nz": nz,
        "r_min": r_min, "r_max": r_max, "w_min": w_min, "w_max": w_max,
        "c_min": c_min, "T_trunc": T_trunc,
        "N_epoch": N_epoch, "N_warmup": N_warmup,
        "lr_ini": 5e-4, "lr_decay": 0.5,
        "N_sample": N_sample, "e_converge": e_converge,
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


def theta_to_c(theta: jnp.ndarray, c_min: float) -> jnp.ndarray:
    return jax.nn.softplus(theta) + c_min


def u_jax(c: jnp.ndarray, sigma: float, c_min: float) -> jnp.ndarray:
    c = jnp.maximum(c, c_min)
    if abs(sigma - 1.0) < 1e-8:
        return jnp.log(c)
    return (c ** (1 - sigma)) / (1 - sigma)


def K_to_prices(K: jnp.ndarray, z: jnp.ndarray, alpha: float, delta: float):
    K = jnp.maximum(K, 1e-8)
    rK = alpha * z * (K ** (alpha - 1))
    w = (1 - alpha) * z * (K ** alpha)
    return rK - delta, w


def update_d_pi_direct_ks(
    theta: jnp.ndarray,
    d: jnp.ndarray,
    iz: int,
    ir: int,
    iw: int,
    b_grid: jnp.ndarray,
    y_grid: jnp.ndarray,
    z_grid: jnp.ndarray,
    r_grid: jnp.ndarray,
    w_grid: jnp.ndarray,
    Ty: jnp.ndarray,
    nb_spg: int,
    ny: int,
    c_min: float,
    b_min: float,
    b_max: float,
    sigma_b: float = 0.5,
) -> jnp.ndarray:
    J = nb_spg * ny
    d_mat = d.reshape(nb_spg, ny)
    z_val = z_grid[iz]
    r_val = r_grid[ir]
    w_val = w_grid[iw]
    c = theta_to_c(theta, c_min)[:, iz, ir, iw]  # (J,)

    b_flat = jnp.repeat(b_grid, ny)
    y_flat = jnp.tile(y_grid, nb_spg)
    b_next = (1 + r_val) * b_flat + y_flat * w_val * z_val - c
    b_next = jnp.clip(b_next, b_min, b_max)

    dist = b_next[:, None] - b_grid[None, :]
    w_b = jnp.exp(-(dist ** 2) / (2 * sigma_b**2))
    w_b = w_b / (w_b.sum(axis=1, keepdims=True) + 1e-8)

    M = jnp.transpose(w_b.reshape(nb_spg, ny, nb_spg), (2, 0, 1))
    Q = (M * d_mat[None, :, :]).sum(axis=1)
    d_new = (Q @ Ty).reshape(J)
    return d_new / (d_new.sum() + 1e-20)


def steady_state_d0(
    theta: jnp.ndarray,
    b_grid: jnp.ndarray,
    y_grid: jnp.ndarray,
    z_grid: jnp.ndarray,
    r_grid: jnp.ndarray,
    w_grid: jnp.ndarray,
    Ty: jnp.ndarray,
    nb_spg: int,
    ny: int,
    c_min: float,
    b_min: float,
    b_max: float,
    n_iter: int = 120,
) -> jnp.ndarray:
    J = nb_spg * ny
    d = jnp.ones((J,)) / J
    iz_mid, ir_mid, iw_mid = z_grid.shape[0] // 2, r_grid.shape[0] // 2, w_grid.shape[0] // 2
    for _ in range(n_iter):
        d = update_d_pi_direct_ks(theta, d, iz_mid, ir_mid, iw_mid, b_grid, y_grid, z_grid, r_grid, w_grid,
                                  Ty, nb_spg, ny, c_min, b_min, b_max)
    return d


def spg_objective_ks(
    theta: jnp.ndarray,
    key: jnp.ndarray,
    N_traj: int,
    T_horizon: int,
    d0: jnp.ndarray,
    b_grid: jnp.ndarray,
    y_grid: jnp.ndarray,
    z_grid: jnp.ndarray,
    r_grid: jnp.ndarray,
    w_grid: jnp.ndarray,
    Ty: jnp.ndarray,
    Tz: jnp.ndarray,
    nb_spg: int,
    ny: int,
    beta: float,
    alpha: float,
    delta: float,
    c_min: float,
    b_min: float,
    b_max: float,
    warm_up: bool,
) -> jnp.ndarray:
    J = nb_spg * ny
    b_flat = jnp.repeat(b_grid, ny)

    def single_traj(k):
        k, k0 = jax.random.split(k)
        iz = jax.random.randint(k0, (), 0, z_grid.shape[0])
        d = d0
        L_n = jnp.float32(0.0)
        for t in range(T_horizon):
            K = (d * b_flat).sum()
            r_t, w_t = K_to_prices(K, z_grid[iz], alpha, delta)
            ir = jnp.clip(jnp.searchsorted(r_grid, r_t, side="right") - 1, 0, r_grid.shape[0] - 1)
            iw = jnp.clip(jnp.searchsorted(w_grid, w_t, side="right") - 1, 0, w_grid.shape[0] - 1)

            c = theta_to_c(theta, c_min)
            c_t = c[:, iz, ir, iw]
            L_n = L_n + (beta ** t) * (d * u_jax(c_t, 3.0, c_min)).sum()

            if not warm_up:
                d = update_d_pi_direct_ks(theta, d, iz, ir, iw, b_grid, y_grid, z_grid, r_grid, w_grid,
                                          Ty, nb_spg, ny, c_min, b_min, b_max)
                d = jax.lax.stop_gradient(d)

            k, kz = jax.random.split(k)
            iz = jax.random.choice(kz, z_grid.shape[0], p=Tz[iz, :])
        return L_n

    keys = jax.random.split(key, N_traj)
    return jnp.mean(jax.vmap(single_traj)(keys))


def policy_from_grid_ks(b, y, r, w, z, c_grid, b_grid, y_grid, z_grid, r_grid, w_grid, ny, c_min=1e-3):
    b = np.atleast_1d(np.asarray(b, dtype=float))
    y = np.atleast_1d(np.asarray(y, dtype=float))
    r = np.atleast_1d(np.asarray(r, dtype=float))
    w = np.atleast_1d(np.asarray(w, dtype=float))
    z = np.atleast_1d(np.asarray(z, dtype=float))

    ib = np.clip(np.searchsorted(b_grid, b, side="right") - 1, 0, len(b_grid) - 1)
    iy = np.clip(np.searchsorted(y_grid, y, side="right") - 1, 0, len(y_grid) - 1)
    iz = np.clip(np.searchsorted(z_grid, z, side="right") - 1, 0, len(z_grid) - 1)
    ir = np.clip(np.searchsorted(r_grid, r, side="right") - 1, 0, len(r_grid) - 1)
    iw = np.clip(np.searchsorted(w_grid, w, side="right") - 1, 0, len(w_grid) - 1)

    j = ib * ny + iy
    c = np.maximum(c_grid[j, iz, ir, iw], c_min)
    cash = (1 + r) * b + w * y
    b_next = np.clip(cash - c, b_grid[0], b_grid[-1])
    c = np.maximum(cash - b_next, c_min)

    if c.size == 1:
        return float(c.ravel()[0]), 1.0, float(b_next.ravel()[0])
    return c, np.ones_like(c), b_next


def main():
    p = argparse.ArgumentParser(description="Krusell-Smith SRL/SPG in JAX")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--out_dir", type=str, default="krusell_smith_output")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    cal = get_calibration(quick=args.quick)
    if args.epochs is not None:
        cal["N_epoch"] = args.epochs

    np.random.seed(args.seed)

    ny = cal["ny"]
    b_min, b_max = cal["b_min"], cal["b_max"]
    c_min = cal["c_min"]

    y_grid_np, Ty_np = tauchen_ar1(cal["rho_y"], cal["nu_y"], ny, m=3, mean=1.0)
    inv_y = np.linalg.matrix_power(Ty_np.T, 200)[:, 0]
    y_grid_np = y_grid_np / (y_grid_np @ inv_y)

    log_z, Tz_np = tauchen_ar1(cal["rho_z"], cal["nu_z"], cal["nz"])
    z_grid_np = np.exp(log_z)
    inv_z = np.linalg.matrix_power(Tz_np.T, 200)[:, 0]
    z_grid_np = z_grid_np / (z_grid_np @ inv_z)

    nb_spg, nr_spg, nw_spg, nz_spg = 50, 15, 25, 10
    iz_spg = np.linspace(0, cal["nz"] - 1, nz_spg, dtype=int)

    b_grid = jnp.array(np.linspace(b_min, b_max, nb_spg))
    r_grid = jnp.array(np.linspace(cal["r_min"], cal["r_max"], nr_spg))
    w_grid = jnp.array(np.linspace(cal["w_min"], cal["w_max"], nw_spg))
    y_grid = jnp.array(y_grid_np)
    z_grid = jnp.array(z_grid_np[iz_spg])
    Ty = jnp.array(Ty_np)
    Tz_sub = Tz_np[np.ix_(iz_spg, iz_spg)]
    Tz_sub = Tz_sub / Tz_sub.sum(axis=1, keepdims=True)
    Tz = jnp.array(Tz_sub)

    key = jax.random.PRNGKey(args.seed)

    J = nb_spg * ny
    theta = jnp.zeros((J, nz_spg, nr_spg, nw_spg), dtype=jnp.float32)
    d0_steady = steady_state_d0(theta, b_grid, y_grid, z_grid, r_grid, w_grid, Ty, nb_spg, ny, c_min, b_min, b_max)

    T_horizon = min(cal["T_trunc"], 40)

    def loss_fn(theta, key, d0, warm_up):
        return -spg_objective_ks(
            theta, key, cal["N_sample"], T_horizon, d0,
            b_grid, y_grid, z_grid, r_grid, w_grid, Ty, Tz,
            nb_spg, ny, cal["beta"], cal["alpha"], cal["delta"], c_min, b_min, b_max, warm_up,
        )

    grad_fn = jax.grad(loss_fn, argnums=0)

    def lr_schedule(step):
        denom = max(cal["N_epoch"] - cal["N_warmup"], 1)
        t0 = jnp.maximum(step - cal["N_warmup"], 0) / denom
        return cal["lr_ini"] * (cal["lr_decay"] ** t0)

    opt = optax.adam(learning_rate=lr_schedule)
    opt_state = opt.init(theta)
    loss_hist = []

    print("KS JAX: nb=%d ny=%d nz=%d nr=%d nw=%d epochs=%d" % (nb_spg, ny, nz_spg, nr_spg, nw_spg, cal["N_epoch"]))
    print("JAX device:", jax.default_backend())

    start = time.perf_counter()
    for epoch in range(cal["N_epoch"]):
        warm_up = epoch < cal["N_warmup"]
        key, ke = jax.random.split(key)
        d0_phase = d0_steady if warm_up else (jnp.ones((J,)) / J)
        L_val = loss_fn(theta, ke, d0_phase, warm_up)
        g = grad_fn(theta, ke, d0_phase, warm_up)
        updates, opt_state = opt.update(g, opt_state)
        theta = optax.apply_updates(theta, updates)

        loss_hist.append(float(-L_val))
        gmax = float(jnp.abs(g).max())
        if (epoch + 1) <= 5 or (epoch + 1) % 20 == 0 or (epoch + 1) == cal["N_warmup"]:
            phase = "warm-up (d fixed)" if warm_up else "d evolves"
            print("Epoch %d, L=%.6f, |grad|=%.2e, %s" % (epoch + 1, -L_val, gmax, phase))
        if gmax < cal["e_converge"]:
            print("Converged at epoch %d" % (epoch + 1))
            break

    elapsed = time.perf_counter() - start
    print("Training done in %.2f s (%d epochs)" % (elapsed, len(loss_hist)))

    os.makedirs(args.out_dir, exist_ok=True)
    c_grid_np = np.array(theta_to_c(theta, c_min))
    b_grid_np = np.array(b_grid)
    y_grid_np = np.array(y_grid)
    z_grid_np = np.array(z_grid)
    r_grid_np = np.array(r_grid)
    w_grid_np = np.array(w_grid)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(loss_hist)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("L(theta)")
    ax.set_title("Krusell-Smith SPG loss (JAX)")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "loss_curve.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    z0 = z_grid_np[len(z_grid_np) // 2]
    w0 = w_grid_np[len(w_grid_np) // 2]
    iy_s, ib_s = 1, nb_spg // 2
    b_val, y_val = float(b_grid_np[ib_s]), float(y_grid_np[iy_s])
    c_r_curve = [policy_from_grid_ks(b_val, y_val, rv, w0, z0, c_grid_np, b_grid_np, y_grid_np, z_grid_np,
                                     r_grid_np, w_grid_np, ny, c_min=c_min)[0] for rv in r_grid_np]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(r_grid_np, c_r_curve, "-o")
    ax.set_xlabel("r")
    ax.set_ylabel("c")
    ax.set_title("Consumption vs r (JAX)")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "consumption_vs_r.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    np.save(os.path.join(args.out_dir, "c_grid.npy"), c_grid_np)
    np.save(os.path.join(args.out_dir, "b_grid.npy"), b_grid_np)
    np.save(os.path.join(args.out_dir, "y_grid.npy"), y_grid_np)
    np.save(os.path.join(args.out_dir, "z_grid.npy"), z_grid_np)
    np.save(os.path.join(args.out_dir, "r_grid.npy"), r_grid_np)
    np.save(os.path.join(args.out_dir, "w_grid.npy"), w_grid_np)
    with open(os.path.join(args.out_dir, "loss_hist.txt"), "w") as f:
        f.write("\n".join(map(str, loss_hist)))

    print("Saved outputs to", args.out_dir)


if __name__ == "__main__":
    main()
