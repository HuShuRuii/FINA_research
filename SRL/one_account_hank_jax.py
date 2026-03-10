#!/usr/bin/env python3
"""
One-account HANK (household block) — JAX implementation.

Baseline port of SRL/one_account_hank.ipynb:
- policy (c, n) on (b, y, r, w) grid (no z in policy)
- objective E[sum beta^t u(c_t, n_t)]
- per period solve: given (d_t, z_t, r_t), clear bond market for w_t;
  then recover Y_t, Pi_t using:
    Y_t = z_t * N_t
    Y_t = C_t + 0.5 * Pi_t * Y_t^2
- r_{t+1} follows Taylor rule from period-t outcomes.

Usage:
  python one_account_hank_jax.py [--epochs N] [--quick] [--out_dir DIR]
      [--rho_r 0.8 --r_ss 0.038 --phi_pi 1.5 --phi_y 0.1 --sigma_r 0.0]
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
    rho_r = 0.8
    r_ss = 0.038
    phi_pi = 1.5
    phi_y = 0.1
    sigma_r = 0.0
    pi_target = 0.0
    y_target = 1.0

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
        "rho_r": rho_r, "r_ss": r_ss, "phi_pi": phi_pi, "phi_y": phi_y, "sigma_r": sigma_r,
        "pi_target": pi_target, "y_target": y_target,
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


def policy_at_rw(
    theta_c: jnp.ndarray,
    theta_n: jnp.ndarray,
    r_t: jnp.ndarray,
    w_t: jnp.ndarray,
    r_grid: jnp.ndarray,
    w_grid: jnp.ndarray,
    c_min: float,
    n_min: float,
):
    """Bilinear interpolation of policy on (r,w) grid."""
    c_pol, n_pol = theta_to_cn(theta_c, theta_n, c_min, n_min)  # (J,nr,nw)
    nr = r_grid.shape[0]
    nw = w_grid.shape[0]

    ir_hi = jnp.clip(jnp.searchsorted(r_grid, r_t, side="right"), 1, nr - 1)
    ir_lo = ir_hi - 1
    wr = (r_t - r_grid[ir_lo]) / jnp.maximum(r_grid[ir_hi] - r_grid[ir_lo], 1e-20)

    iw_hi = jnp.clip(jnp.searchsorted(w_grid, w_t, side="right"), 1, nw - 1)
    iw_lo = iw_hi - 1
    ww = (w_t - w_grid[iw_lo]) / jnp.maximum(w_grid[iw_hi] - w_grid[iw_lo], 1e-20)

    c00 = c_pol[:, ir_lo, iw_lo]
    c01 = c_pol[:, ir_lo, iw_hi]
    c10 = c_pol[:, ir_hi, iw_lo]
    c11 = c_pol[:, ir_hi, iw_hi]

    n00 = n_pol[:, ir_lo, iw_lo]
    n01 = n_pol[:, ir_lo, iw_hi]
    n10 = n_pol[:, ir_hi, iw_lo]
    n11 = n_pol[:, ir_hi, iw_hi]

    c_t = (1 - wr) * ((1 - ww) * c00 + ww * c01) + wr * ((1 - ww) * c10 + ww * c11)
    n_t = (1 - wr) * ((1 - ww) * n00 + ww * n01) + wr * ((1 - ww) * n10 + ww * n11)
    return c_t, n_t


def one_period_bnext_and_dist(
    d: jnp.ndarray,
    c_t: jnp.ndarray,
    n_t: jnp.ndarray,
    z_t: jnp.ndarray,
    r_t: jnp.ndarray,
    w_t: jnp.ndarray,
    pi_t: jnp.ndarray,
    y_t: jnp.ndarray,
    b_grid: jnp.ndarray,
    y_grid: jnp.ndarray,
    Ty: jnp.ndarray,
    nb: int,
    ny: int,
    B_supply: float,
    b_min: float,
    b_max: float,
    sigma_b: float = 0.5,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Given (r_t, w_t, pi_t, y_t), update distribution and return b_next."""
    T_t = r_t * B_supply
    div_t = (1.0 - w_t / (z_t + 1e-20)) * y_t - 0.5 * pi_t * (y_t ** 2)

    b_vals = jnp.repeat(b_grid, ny)
    y_vals = jnp.tile(y_grid, nb)

    b_next = (1 + r_t) * b_vals + w_t * y_vals * n_t + div_t - T_t - c_t
    b_next = jnp.clip(b_next, b_min, b_max)

    dist = b_next[:, None] - b_grid[None, :]
    w_b = jnp.exp(-(dist ** 2) / (2 * sigma_b**2))
    w_b = w_b / (w_b.sum(axis=1, keepdims=True) + 1e-8)

    M = jnp.transpose(w_b.reshape(nb, ny, nb), (2, 0, 1))
    d_mat = d.reshape(nb, ny)
    Q = (M * d_mat[None, :, :]).sum(axis=1)
    d_new = (Q @ Ty).reshape(nb * ny)
    d_new = d_new / (d_new.sum() + 1e-20)
    return b_next, d_new


def macro_block_from_w(
    theta_c: jnp.ndarray,
    theta_n: jnp.ndarray,
    d: jnp.ndarray,
    z_t: jnp.ndarray,
    r_t: jnp.ndarray,
    w_t: jnp.ndarray,
    b_grid: jnp.ndarray,
    y_grid: jnp.ndarray,
    r_grid: jnp.ndarray,
    w_grid: jnp.ndarray,
    Ty: jnp.ndarray,
    nb: int,
    ny: int,
    c_min: float,
    n_min: float,
    B_supply: float,
    b_min: float,
    b_max: float,
):
    """Solve (Pi_t, Y_t) given w_t and return bond residual for clearing."""
    c_t, n_t = policy_at_rw(theta_c, theta_n, r_t, w_t, r_grid, w_grid, c_min, n_min)
    c_agg = (d * c_t).sum()
    n_supply = (d * n_t).sum()
    y_t = z_t * n_supply
    pi_t = 2.0 * (y_t - c_agg) / jnp.maximum(y_t ** 2, 1e-8)  # user-specified goods equation

    b_next, d_next = one_period_bnext_and_dist(
        d, c_t, n_t, z_t, r_t, w_t, pi_t, y_t,
        b_grid, y_grid, Ty, nb, ny, B_supply, b_min, b_max,
    )
    b_vals = jnp.repeat(b_grid, ny)
    b_next_agg = (d_next * b_vals).sum()
    bond_res = b_next_agg - B_supply
    return bond_res, c_t, n_t, pi_t, y_t, d_next, b_next


def solve_w_given_rt(
    theta_c: jnp.ndarray,
    theta_n: jnp.ndarray,
    d: jnp.ndarray,
    z_t: jnp.ndarray,
    r_t: jnp.ndarray,
    b_grid: jnp.ndarray,
    y_grid: jnp.ndarray,
    r_grid: jnp.ndarray,
    w_grid: jnp.ndarray,
    Ty: jnp.ndarray,
    nb: int,
    ny: int,
    c_min: float,
    n_min: float,
    B_supply: float,
    b_min: float,
    b_max: float,
):
    """Find w_t to clear bond market at given (d_t, z_t, r_t) by bracket interpolation."""
    nw = w_grid.shape[0]
    res_vals = []
    out_vals = []
    for iw in range(nw):
        out = macro_block_from_w(
            theta_c, theta_n, d, z_t, r_t, w_grid[iw], b_grid, y_grid, r_grid, w_grid, Ty, nb, ny,
            c_min, n_min, B_supply, b_min, b_max,
        )
        res_vals.append(out[0])
        out_vals.append(out)
    res = jnp.stack(res_vals)
    ge = res >= 0.0
    has_ge = jnp.any(ge)
    first_ge = jnp.argmax(ge)
    iw_hi_raw = jnp.where(has_ge, first_ge, nw - 1)
    iw_hi = jnp.clip(iw_hi_raw, 1, nw - 1)
    iw_lo = iw_hi - 1
    r_lo = res[iw_lo]
    r_hi = res[iw_hi]
    w_mix = jnp.clip((0.0 - r_lo) / jnp.maximum(r_hi - r_lo, 1e-20), 0.0, 1.0)
    w_t = w_grid[iw_lo] + w_mix * (w_grid[iw_hi] - w_grid[iw_lo])

    # Re-evaluate at interpolated wage.
    out_star = macro_block_from_w(
        theta_c, theta_n, d, z_t, r_t, w_t, b_grid, y_grid, r_grid, w_grid, Ty, nb, ny,
        c_min, n_min, B_supply, b_min, b_max,
    )
    return w_t, out_star


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
    rho_r: float,
    r_ss: float,
    phi_pi: float,
    phi_y: float,
    sigma_r: float,
    pi_target: float,
    y_target: float,
) -> jnp.ndarray:
    nb = b_grid.shape[0]
    ny = y_grid.shape[0]

    def single_traj(k):
        k, kz = jax.random.split(k)
        iz = jax.random.randint(kz, (), 0, z_grid.shape[0])
        d = d0
        r_t = jnp.float32(r_ss)
        L_n = jnp.float32(0.0)

        for t in range(T_horizon):
            z_t = z_grid[iz]
            r_t = jnp.clip(r_t, r_grid[0], r_grid[-1])
            # Solve macro block off-graph (prices treated as given in policy gradient).
            w_t, out = solve_w_given_rt(
                jax.lax.stop_gradient(theta_c),
                jax.lax.stop_gradient(theta_n),
                jax.lax.stop_gradient(d),
                z_t, r_t, b_grid, y_grid, r_grid, w_grid, Ty, nb, ny,
                c_min, n_min, B_supply, b_min, b_max,
            )
            _bond_res, _c_macro, _n_macro, pi_t, y_t, d_next, _b_next = out
            # Stop gradients through macro block (SRL convention).
            w_t = jax.lax.stop_gradient(w_t)
            pi_t = jax.lax.stop_gradient(pi_t)
            y_t = jax.lax.stop_gradient(y_t)
            c_t, n_t = policy_at_rw(theta_c, theta_n, r_t, w_t, r_grid, w_grid, c_min, n_min)
            d_use = jax.lax.stop_gradient(d)
            L_n = L_n + (beta ** t) * (d_use * u_jax(c_t, n_t, sigma, eta, c_min, n_min)).sum()
            d = jax.lax.stop_gradient(d_next)

            k, kz = jax.random.split(k)
            iz = jax.random.choice(kz, z_grid.shape[0], p=Tz[iz, :])
            k, ke = jax.random.split(k)
            eps_r = jax.random.normal(ke, ())
            r_t = (1.0 - rho_r) * r_ss + rho_r * r_t + phi_pi * (pi_t - pi_target) + phi_y * (y_t - y_target) + sigma_r * eps_r
        return L_n

    keys = jax.random.split(key, N_traj)
    return jnp.mean(jax.vmap(single_traj)(keys))


def main():
    parser = argparse.ArgumentParser(description="One-account HANK household block (JAX)")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--out_dir", type=str, default="one_account_hank_output")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rho_r", type=float, default=None, help="Taylor rule persistence")
    parser.add_argument("--r_ss", type=float, default=None, help="Steady-state policy rate")
    parser.add_argument("--phi_pi", type=float, default=None, help="Taylor loading on inflation")
    parser.add_argument("--phi_y", type=float, default=None, help="Taylor loading on output gap")
    parser.add_argument("--sigma_r", type=float, default=None, help="Taylor shock std")
    args = parser.parse_args()

    cal = get_calibration(quick=args.quick)
    if args.epochs is not None:
        cal["N_epoch"] = args.epochs
    if args.rho_r is not None:
        cal["rho_r"] = args.rho_r
    if args.r_ss is not None:
        cal["r_ss"] = args.r_ss
    if args.phi_pi is not None:
        cal["phi_pi"] = args.phi_pi
    if args.phi_y is not None:
        cal["phi_y"] = args.phi_y
    if args.sigma_r is not None:
        cal["sigma_r"] = args.sigma_r

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
            cal["B_supply"], cal["b_min"], cal["b_max"],
            cal["rho_r"], cal["r_ss"], cal["phi_pi"], cal["phi_y"], cal["sigma_r"],
            cal["pi_target"], cal["y_target"],
        )

    grad_fn = jax.grad(loss_fn, argnums=(0, 1))
    opt = optax.adam(cal["lr_ini"])
    opt_state = opt.init((theta_c, theta_n))

    loss_hist = []
    print("HANK JAX: nb=%d ny=%d nz=%d nr=%d nw=%d epochs=%d" % (nb_spg, ny, nz_spg, nr_spg, nw_spg, cal["N_epoch"]))
    print("JAX device:", jax.default_backend())
    print("Taylor: rho_r=%.3f r_ss=%.3f phi_pi=%.3f phi_y=%.3f sigma_r=%.3f"
          % (cal["rho_r"], cal["r_ss"], cal["phi_pi"], cal["phi_y"], cal["sigma_r"]))

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
