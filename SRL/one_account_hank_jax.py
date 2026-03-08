#!/usr/bin/env python3
"""
One-account HANK (household block) — JAX implementation.

Baseline port of SRL/one_account_hank.ipynb:
- policy (c, n) on (b, y, r, w) grid (no z in policy)
- objective E[sum beta^t u(c_t, n_t)]
- taxes/dividends from simplified NK block identities
- optional (slow) market-clearing grid search for (r,w)

Usage:
  python one_account_hank_jax.py [--epochs N] [--quick] [--out_dir DIR] [--use_market_clearing]
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


def get_calibration(quick: bool = False) -> dict:
    beta = 0.975
    sigma = 1.0
    eta = 1.0
    B_supply = 10.0
    n_min = 1e-4

    epsilon = 6.0
    theta_rotemberg = 0.1

    rho_y, nu_y = 0.6, 0.2
    rho_z, nu_z = 0.9, 0.02

    ny = 3
    nz = 30
    b_min, b_max = -1.0, 50.0
    r_min, r_max = 0.005, 0.04
    w_min, w_max = 0.5, 2.0
    c_min = 1e-3

    if quick:
        N_epoch, N_sample, T_horizon = 20, 8, 20
    else:
        N_epoch, N_sample, T_horizon = 150, 24, 40

    return {
        "beta": beta, "sigma": sigma, "eta": eta,
        "B_supply": B_supply, "n_min": n_min,
        "epsilon": epsilon, "theta_rotemberg": theta_rotemberg,
        "rho_y": rho_y, "nu_y": nu_y, "rho_z": rho_z, "nu_z": nu_z,
        "ny": ny, "nz": nz,
        "b_min": b_min, "b_max": b_max,
        "r_min": r_min, "r_max": r_max, "w_min": w_min, "w_max": w_max,
        "c_min": c_min,
        "N_epoch": N_epoch, "N_sample": N_sample, "T_horizon": T_horizon,
        "lr_ini": 5e-4, "e_converge": 3e-4,
    }


def theta_to_cn(theta_c: jnp.ndarray, theta_n: jnp.ndarray, c_min: float, n_min: float):
    c = jax.nn.softplus(theta_c) + c_min
    n = jax.nn.sigmoid(theta_n) * (2.0 - n_min) + n_min
    return c, n


def u_jax(c: jnp.ndarray, n: jnp.ndarray, sigma: float, eta: float, c_min: float, n_min: float):
    c = jnp.maximum(c, c_min)
    n = jnp.clip(n, n_min, 10.0)
    if abs(sigma - 1.0) < 1e-8:
        uc = jnp.log(c)
    else:
        uc = (c ** (1 - sigma)) / (1 - sigma)
    un = -(n ** (1 + eta)) / (1 + eta)
    return uc + un


def Y_from_C_Pi(C_agg: jnp.ndarray, Pi_t: jnp.ndarray, theta_rot: float):
    denom = 1.0 - (theta_rot / 2.0) * (Pi_t ** 2)
    return C_agg / jnp.maximum(denom, 1e-8)


def dividend_agg(Y_t: jnp.ndarray, w_t: jnp.ndarray, z_t: jnp.ndarray, Pi_t: jnp.ndarray, theta_rot: float):
    return (1.0 - w_t / (z_t + 1e-20)) * Y_t - (theta_rot / 2.0) * (Pi_t ** 2) * Y_t


def update_d_hank(
    d: jnp.ndarray,
    c_t: jnp.ndarray,
    n_t: jnp.ndarray,
    iz: int,
    ir: int,
    iw: int,
    b_grid: jnp.ndarray,
    y_grid: jnp.ndarray,
    z_grid: jnp.ndarray,
    r_grid: jnp.ndarray,
    w_grid: jnp.ndarray,
    Ty: jnp.ndarray,
    nb: int,
    ny: int,
    B_supply: float,
    b_min: float,
    b_max: float,
    theta_rot: float,
    sigma_b: float = 0.5,
) -> jnp.ndarray:
    r_t = r_grid[ir]
    w_t = w_grid[iw]
    z_t = z_grid[iz]
    T_t = r_t * B_supply

    Pi_t = jnp.float32(0.0)
    C_agg = (d * c_t).sum()
    Y_t = Y_from_C_Pi(C_agg, Pi_t, theta_rot)
    d_t = dividend_agg(Y_t, w_t, z_t, Pi_t, theta_rot)

    b_vals = jnp.repeat(b_grid, ny)
    y_vals = jnp.tile(y_grid, nb)

    b_next = (1 + r_t) * b_vals + w_t * y_vals * n_t + d_t - T_t - c_t
    b_next = jnp.clip(b_next, b_min, b_max)

    dist = b_next[:, None] - b_grid[None, :]
    w_b = jnp.exp(-(dist ** 2) / (2 * sigma_b**2))
    w_b = w_b / (w_b.sum(axis=1, keepdims=True) + 1e-8)

    M = jnp.transpose(w_b.reshape(nb, ny, nb), (2, 0, 1))
    d_mat = d.reshape(nb, ny)
    Q = (M * d_mat[None, :, :]).sum(axis=1)
    d_new = (Q @ Ty).reshape(nb * ny)
    return d_new / (d_new.sum() + 1e-20)


def market_clearing_residual(
    ir: int,
    iw: int,
    d: jnp.ndarray,
    iz: int,
    c_pol: jnp.ndarray,
    n_pol: jnp.ndarray,
    b_grid: jnp.ndarray,
    y_grid: jnp.ndarray,
    z_grid: jnp.ndarray,
    r_grid: jnp.ndarray,
    w_grid: jnp.ndarray,
    Ty: jnp.ndarray,
    nb: int,
    ny: int,
    B_supply: float,
    b_min: float,
    b_max: float,
    theta_rot: float,
):
    c_t = c_pol[:, ir, iw]
    n_t = n_pol[:, ir, iw]
    d_next = update_d_hank(d, c_t, n_t, iz, ir, iw, b_grid, y_grid, z_grid, r_grid, w_grid,
                           Ty, nb, ny, B_supply, b_min, b_max, theta_rot)

    b_vals = jnp.repeat(b_grid, ny)
    B_next = (d_next * b_vals).sum()

    r_t = r_grid[ir]
    w_t = w_grid[iw]
    z_t = z_grid[iz]
    Pi_t = jnp.float32(0.0)
    C_agg = (d * c_t).sum()
    Y_t = Y_from_C_Pi(C_agg, Pi_t, theta_rot)

    N_supply = (d * n_t).sum()
    N_demand = Y_t / (z_t + 1e-20)

    return (B_next - B_supply) ** 2 + (N_supply - N_demand) ** 2


def spg_objective_hank(
    theta_c: jnp.ndarray,
    theta_n: jnp.ndarray,
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
    beta: float,
    sigma: float,
    eta: float,
    c_min: float,
    n_min: float,
    B_supply: float,
    b_min: float,
    b_max: float,
    theta_rot: float,
    use_market_clearing: bool,
) -> jnp.ndarray:
    nb = b_grid.shape[0]
    ny = y_grid.shape[0]

    def single_traj(k):
        k, kz = jax.random.split(k)
        iz = jax.random.randint(kz, (), 0, z_grid.shape[0])
        d = d0
        L_n = jnp.float32(0.0)

        for t in range(T_horizon):
            c_pol, n_pol = theta_to_cn(theta_c, theta_n, c_min, n_min)

            if use_market_clearing:
                best = jnp.inf
                ir_best, iw_best = 0, 0
                for ir in range(r_grid.shape[0]):
                    for iw in range(w_grid.shape[0]):
                        res = market_clearing_residual(ir, iw, d, iz, c_pol, n_pol, b_grid, y_grid, z_grid,
                                                       r_grid, w_grid, Ty, nb, ny, B_supply, b_min, b_max, theta_rot)
                        cond = res < best
                        best = jnp.where(cond, res, best)
                        ir_best = jnp.where(cond, ir, ir_best)
                        iw_best = jnp.where(cond, iw, iw_best)
                ir, iw = ir_best, iw_best
            else:
                k, kr, kw = jax.random.split(k, 3)
                ir = jax.random.randint(kr, (), 0, r_grid.shape[0])
                iw = jax.random.randint(kw, (), 0, w_grid.shape[0])

            r_t = jax.lax.stop_gradient(r_grid[ir])
            w_t = jax.lax.stop_gradient(w_grid[iw])
            z_t = z_grid[iz]
            T_t = r_t * B_supply

            c_t = c_pol[:, ir, iw]
            n_t = n_pol[:, ir, iw]

            Pi_t = jnp.float32(0.0)
            C_agg = jax.lax.stop_gradient((d * c_t).sum())
            Y_t = Y_from_C_Pi(C_agg, Pi_t, theta_rot)
            d_t = dividend_agg(Y_t, w_t, z_t, Pi_t, theta_rot)

            L_n = L_n + (beta ** t) * (d * u_jax(c_t, n_t, sigma, eta, c_min, n_min)).sum()

            b_vals = jnp.repeat(b_grid, ny)
            y_vals = jnp.tile(y_grid, nb)
            b_next = (1 + r_t) * b_vals + w_t * y_vals * n_t + d_t - T_t - c_t
            b_next = jnp.clip(b_next, b_min, b_max)

            dist = b_next[:, None] - b_grid[None, :]
            w_b = jnp.exp(-(dist ** 2) / (2 * 0.5**2))
            w_b = w_b / (w_b.sum(axis=1, keepdims=True) + 1e-8)
            M = jnp.transpose(w_b.reshape(nb, ny, nb), (2, 0, 1))
            d_mat = d.reshape(nb, ny)
            Q = (M * d_mat[None, :, :]).sum(axis=1)
            d = (Q @ Ty).reshape(nb * ny)
            d = jax.lax.stop_gradient(d / (d.sum() + 1e-20))

            k, kz = jax.random.split(k)
            iz = jax.random.choice(kz, z_grid.shape[0], p=Tz[iz, :])
        return L_n

    keys = jax.random.split(key, N_traj)
    return jnp.mean(jax.vmap(single_traj)(keys))


def main():
    parser = argparse.ArgumentParser(description="One-account HANK household block (JAX)")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--out_dir", type=str, default="one_account_hank_output")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_market_clearing", action="store_true", help="Solve (r,w) by grid-search clearing each period")
    args = parser.parse_args()

    cal = get_calibration(quick=args.quick)
    if args.epochs is not None:
        cal["N_epoch"] = args.epochs

    np.random.seed(args.seed)

    y_grid_np, Ty_np = tauchen_ar1(cal["rho_y"], cal["nu_y"], cal["ny"], m=3, mean=1.0)
    inv_y = np.linalg.matrix_power(Ty_np.T, 200)[:, 0]
    y_grid_np = y_grid_np / (y_grid_np @ inv_y)

    log_z, Tz_np = tauchen_ar1(cal["rho_z"], cal["nu_z"], cal["nz"])
    z_grid_np = np.exp(log_z)
    inv_z = np.linalg.matrix_power(Tz_np.T, 200)[:, 0]
    z_grid_np = z_grid_np / (z_grid_np @ inv_z)

    nb_spg, nr_spg, nw_spg, nz_spg = 30, 10, 10, 8
    iz = np.linspace(0, cal["nz"] - 1, nz_spg, dtype=int)

    b_grid = jnp.array(np.linspace(cal["b_min"], cal["b_max"], nb_spg))
    r_grid = jnp.array(np.linspace(cal["r_min"], cal["r_max"], nr_spg))
    w_grid = jnp.array(np.linspace(cal["w_min"], cal["w_max"], nw_spg))
    y_grid = jnp.array(y_grid_np)
    z_grid = jnp.array(z_grid_np[iz])
    Ty = jnp.array(Ty_np)
    Tz_sub = Tz_np[np.ix_(iz, iz)]
    Tz_sub = Tz_sub / Tz_sub.sum(axis=1, keepdims=True)
    Tz = jnp.array(Tz_sub)

    ny = y_grid.shape[0]
    J = nb_spg * ny

    theta_c = jnp.zeros((J, nr_spg, nw_spg), dtype=jnp.float32)
    theta_n = jnp.zeros((J, nr_spg, nw_spg), dtype=jnp.float32)

    d0 = jnp.ones((J,)) / J

    key = jax.random.PRNGKey(args.seed)

    def loss_fn(theta_c, theta_n, key):
        return -spg_objective_hank(
            theta_c, theta_n, key,
            cal["N_sample"], cal["T_horizon"], d0,
            b_grid, y_grid, z_grid, r_grid, w_grid, Ty, Tz,
            cal["beta"], cal["sigma"], cal["eta"], cal["c_min"], cal["n_min"],
            cal["B_supply"], cal["b_min"], cal["b_max"], cal["theta_rotemberg"],
            args.use_market_clearing,
        )

    grad_fn = jax.grad(loss_fn, argnums=(0, 1))
    opt = optax.adam(cal["lr_ini"])
    opt_state = opt.init((theta_c, theta_n))

    loss_hist = []
    print("HANK JAX: nb=%d ny=%d nz=%d nr=%d nw=%d epochs=%d" % (nb_spg, ny, nz_spg, nr_spg, nw_spg, cal["N_epoch"]))
    print("JAX device:", jax.default_backend(), "market_clearing=", args.use_market_clearing)

    start = time.perf_counter()
    for epoch in range(cal["N_epoch"]):
        key, ke = jax.random.split(key)
        L_val = loss_fn(theta_c, theta_n, ke)
        g_c, g_n = grad_fn(theta_c, theta_n, ke)
        updates, opt_state = opt.update((g_c, g_n), opt_state, params=(theta_c, theta_n))
        theta_c, theta_n = optax.apply_updates((theta_c, theta_n), updates)

        loss_hist.append(float(-L_val))
        gmax = float(jnp.maximum(jnp.abs(g_c).max(), jnp.abs(g_n).max()))
        if (epoch + 1) <= 5 or (epoch + 1) % 20 == 0:
            print("Epoch %d, L=%.6f, |grad|=%.2e" % (epoch + 1, -L_val, gmax))
        if gmax < cal["e_converge"]:
            print("Converged at epoch %d" % (epoch + 1))
            break

    elapsed = time.perf_counter() - start
    print("Training done in %.2f s (%d epochs)" % (elapsed, len(loss_hist)))

    os.makedirs(args.out_dir, exist_ok=True)
    c_pol, n_pol = theta_to_cn(theta_c, theta_n, cal["c_min"], cal["n_min"])
    c_pol_np = np.array(c_pol)
    n_pol_np = np.array(n_pol)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(loss_hist)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("L(theta)")
    ax.set_title("One-account HANK SPG loss (JAX)")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "loss_curve.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    ib, iy, iw = nb_spg // 2, 1, nw_spg // 2
    j = ib * ny + iy
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.array(r_grid), c_pol_np[j, :, iw], label="c")
    ax.plot(np.array(r_grid), n_pol_np[j, :, iw], label="n")
    ax.set_xlabel("r")
    ax.set_title("Policy vs r (fixed b,y,w)")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "policy_vs_r.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    np.save(os.path.join(args.out_dir, "c_policy.npy"), c_pol_np)
    np.save(os.path.join(args.out_dir, "n_policy.npy"), n_pol_np)
    np.save(os.path.join(args.out_dir, "b_grid.npy"), np.array(b_grid))
    np.save(os.path.join(args.out_dir, "y_grid.npy"), np.array(y_grid))
    np.save(os.path.join(args.out_dir, "z_grid.npy"), np.array(z_grid))
    np.save(os.path.join(args.out_dir, "r_grid.npy"), np.array(r_grid))
    np.save(os.path.join(args.out_dir, "w_grid.npy"), np.array(w_grid))
    with open(os.path.join(args.out_dir, "loss_hist.txt"), "w") as f:
        f.write("\n".join(map(str, loss_hist)))

    print("Saved outputs to", args.out_dir)


if __name__ == "__main__":
    main()
