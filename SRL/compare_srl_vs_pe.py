#!/usr/bin/env python3
"""
Benchmark a clean partial-equilibrium (PE) household problem against the saved
original Huggett general-equilibrium (GE) policy output.

PE setup:
  - Same household state space as GE: (b, y, z, r)
  - Same stochastic y and z processes as the GE calibration
  - Exogenous interest-rate process:
      r_{t+1} = (1-rho_r) * r_ss + rho_r * r_t + v_r * sqrt(max(0, r_t)) * eps_t
  - Borrowing constraint: b' >= -1

GE setup:
  - Loads the saved policy grid and simulation diagnostics from a valid Huggett GE run
  - Treats r_t as endogenous and already solved in the saved benchmark output
"""
from __future__ import annotations

import argparse
import os
from typing import Dict, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


def tauchen_ar1(
    rho: float,
    sigma_innov: float,
    n_states: int,
    m: float = 3.0,
    mean: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    std = sigma_innov / np.sqrt(1.0 - rho ** 2)
    if mean == 0.0:
        x_min, x_max = -m * std, m * std
    else:
        x_min = max(1e-6, mean - m * std)
        x_max = mean + m * std
    x_grid = np.linspace(x_min, x_max, n_states)
    step = (x_max - x_min) / (n_states - 1) if n_states > 1 else 1.0
    mu_i = (1.0 - rho) * mean + rho * x_grid
    z_lo = (x_grid - mu_i[:, None] + step / 2.0) / sigma_innov
    z_hi = (x_grid - mu_i[:, None] - step / 2.0) / sigma_innov
    trans = np.zeros((n_states, n_states), dtype=float)
    trans[:, 0] = norm.cdf(z_lo[:, 0])
    trans[:, -1] = 1.0 - norm.cdf(z_hi[:, -1])
    if n_states > 2:
        trans[:, 1:-1] = norm.cdf(z_lo[:, 1:-1]) - norm.cdf(z_hi[:, 1:-1])
    trans = trans / np.maximum(trans.sum(axis=1, keepdims=True), 1e-20)
    return x_grid, trans


def crra_u(c: np.ndarray, sigma: float, c_min: float) -> np.ndarray:
    c = np.maximum(c, c_min)
    if abs(sigma - 1.0) < 1e-8:
        return np.log(c)
    return (c ** (1.0 - sigma)) / (1.0 - sigma)


def interp_weights_1d(grid: np.ndarray, x: float) -> Tuple[int, int, float]:
    if x <= float(grid[0]):
        return 0, 0, 0.0
    if x >= float(grid[-1]):
        n = len(grid) - 1
        return n, n, 0.0
    hi = int(np.searchsorted(grid, x, side="right"))
    lo = hi - 1
    w = (x - grid[lo]) / max(grid[hi] - grid[lo], 1e-20)
    return lo, hi, float(w)


def build_r_transition(
    r_grid: np.ndarray,
    rho_r: float,
    r_ss: float,
    v_r: float,
    quad_n: int = 61,
) -> np.ndarray:
    eps = np.linspace(-4.0, 4.0, quad_n)
    w = np.exp(-0.5 * eps ** 2)
    w = w / np.maximum(w.sum(), 1e-20)

    nr = len(r_grid)
    trans = np.zeros((nr, nr), dtype=float)
    for i, r_val in enumerate(r_grid):
        r_next = (1.0 - rho_r) * r_ss + rho_r * r_val + v_r * np.sqrt(max(0.0, r_val)) * eps
        r_next = np.clip(r_next, r_grid[0], r_grid[-1])
        for rn, wn in zip(r_next, w):
            j_hi = int(np.searchsorted(r_grid, rn, side="right"))
            if j_hi <= 0:
                trans[i, 0] += wn
            elif j_hi >= nr:
                trans[i, nr - 1] += wn
            else:
                j_lo = j_hi - 1
                alpha = (rn - r_grid[j_lo]) / max(r_grid[j_hi] - r_grid[j_lo], 1e-20)
                trans[i, j_lo] += wn * (1.0 - alpha)
                trans[i, j_hi] += wn * alpha
    trans = trans / np.maximum(trans.sum(axis=1, keepdims=True), 1e-20)
    return trans


def policy_slice_at_zr(
    policy_grid: np.ndarray,
    z_val: float,
    r_val: float,
    z_grid: np.ndarray,
    r_grid: np.ndarray,
) -> np.ndarray:
    iz_lo, iz_hi, wz = interp_weights_1d(z_grid, z_val)
    ir_lo, ir_hi, wr = interp_weights_1d(r_grid, r_val)
    slice_lo_z = (1.0 - wr) * policy_grid[:, :, iz_lo, ir_lo] + wr * policy_grid[:, :, iz_lo, ir_hi]
    slice_hi_z = (1.0 - wr) * policy_grid[:, :, iz_hi, ir_lo] + wr * policy_grid[:, :, iz_hi, ir_hi]
    return (1.0 - wz) * slice_lo_z + wz * slice_hi_z


def eval_policy_curve(
    b_eval: np.ndarray,
    y_val: float,
    z_val: float,
    r_val: float,
    policy_grid: np.ndarray,
    b_grid: np.ndarray,
    y_grid: np.ndarray,
    z_grid: np.ndarray,
    r_grid: np.ndarray,
) -> np.ndarray:
    iy_lo, iy_hi, wy = interp_weights_1d(y_grid, y_val)
    slice_zr = policy_slice_at_zr(policy_grid, z_val, r_val, z_grid, r_grid)
    policy_by_b = (1.0 - wy) * slice_zr[:, iy_lo] + wy * slice_zr[:, iy_hi]
    return np.interp(b_eval, b_grid, policy_by_b)


def simulate_pe_r_path(
    T: int,
    r0: float,
    rho_r: float,
    r_ss: float,
    v_r: float,
    r_min: float,
    r_max: float,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    r_path = np.empty(T, dtype=float)
    r_path[0] = np.clip(r0, r_min, r_max)
    for t in range(T - 1):
        eps = rng.standard_normal()
        r_next = (1.0 - rho_r) * r_ss + rho_r * r_path[t] + v_r * np.sqrt(max(0.0, r_path[t])) * eps
        r_path[t + 1] = np.clip(r_next, r_min, r_max)
    return r_path


def simulate_markov_values(
    T: int,
    grid: np.ndarray,
    trans: np.ndarray,
    start_value: float,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = int(np.argmin(np.abs(grid - start_value)))
    out_idx = np.empty(T, dtype=int)
    for t in range(T):
        out_idx[t] = idx
        idx = int(rng.choice(len(grid), p=trans[idx]))
    return grid[out_idx]


def validate_ge_benchmark(ge_dir: str, r_grid: np.ndarray) -> Dict[str, float]:
    stats: Dict[str, float] = {}
    residual_path = os.path.join(ge_dir, "market_clearing_residual.npy")
    if os.path.exists(residual_path):
        residual = np.load(residual_path)
        stats["ge_max_abs_market_residual"] = float(np.max(np.abs(residual)))
        if stats["ge_max_abs_market_residual"] > 1e-4:
            raise ValueError(
                "Selected GE benchmark has large market-clearing residual %.3e; "
                "please point to a valid original Huggett run instead." % stats["ge_max_abs_market_residual"]
            )

    summary_path = os.path.join(ge_dir, "steady_state_summary.txt")
    if os.path.exists(summary_path):
        parsed: Dict[str, float] = {}
        with open(summary_path, "r", encoding="utf-8") as f:
            for line in f:
                if "=" not in line:
                    continue
                key, value = line.strip().split("=", 1)
                try:
                    parsed[key] = float(value)
                except ValueError:
                    continue
        if "market_residual" in parsed and abs(parsed["market_residual"]) > 1e-4:
            raise ValueError(
                "Selected GE benchmark steady state has residual %.3e and is not a clean benchmark."
                % parsed["market_residual"]
            )
        if "r_ss" in parsed and abs(parsed["r_ss"] - float(r_grid[-1])) < 1e-8:
            raise ValueError(
                "Selected GE benchmark pins the steady-state rate to the upper grid bound %.3f and is invalid."
                % float(r_grid[-1])
            )
        stats.update({k: float(v) for k, v in parsed.items()})
    return stats


def solve_pe_policy(
    b_grid: np.ndarray,
    y_grid: np.ndarray,
    z_grid: np.ndarray,
    r_grid: np.ndarray,
    Ty: np.ndarray,
    Tz: np.ndarray,
    Tr: np.ndarray,
    beta: float,
    sigma: float,
    c_min: float,
    borrow_min: float,
    max_iter: int = 300,
    tol: float = 1e-6,
    log_every: int = 10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    nb, ny, nz, nr = len(b_grid), len(y_grid), len(z_grid), len(r_grid)
    choice_idx = np.flatnonzero(b_grid >= borrow_min - 1e-12)
    if choice_idx.size == 0:
        raise ValueError("No feasible b' choices satisfy borrow_min=%.6f" % borrow_min)
    b_choices = b_grid[choice_idx]

    V = np.zeros((nb, ny, nz, nr), dtype=float)
    policy_idx = np.full((nb, ny, nz, nr), int(choice_idx[0]), dtype=int)
    bellman_err = []

    resources_base = (
        (1.0 + r_grid[None, None, None, :]) * b_grid[:, None, None, None]
        + (y_grid[None, :, None, None] * z_grid[None, None, :, None])
    )

    for it in range(max_iter):
        EV_r = np.einsum("byzr,kr->byzk", V, Tr, optimize=True)
        EV_rz = np.einsum("byzk,jz->byjk", EV_r, Tz, optimize=True)
        EV = np.einsum("byjk,iy->bijk", EV_rz, Ty, optimize=True)

        V_new = np.empty_like(V)
        policy_new = np.empty_like(policy_idx)
        for iz in range(nz):
            yz_income = y_grid[None, :] * z_grid[iz]
            for ir in range(nr):
                resources = (1.0 + r_grid[ir]) * b_grid[:, None] + yz_income
                c_all = resources[:, :, None] - b_choices[None, None, :]
                util = np.where(c_all > c_min, crra_u(c_all, sigma, c_min), -1e12)
                continuation = beta * EV[choice_idx, :, iz, ir].T
                rhs = util + continuation[None, :, :]
                best_local = np.argmax(rhs, axis=2)
                V_new[:, :, iz, ir] = np.take_along_axis(rhs, best_local[:, :, None], axis=2)[:, :, 0]
                policy_new[:, :, iz, ir] = choice_idx[best_local]

        err = float(np.max(np.abs(V_new - V)))
        bellman_err.append(err)
        V = V_new
        policy_idx = policy_new
        if (it + 1) % log_every == 0 or it == 0:
            print("PE VFI iter %d, sup-norm diff = %.3e" % (it + 1, err), flush=True)
        if err < tol:
            break

    b_next = b_grid[policy_idx]
    c_policy = np.maximum(resources_base - b_next, c_min)
    return c_policy, b_next, np.asarray(bellman_err, dtype=float)


def update_distribution(
    G: np.ndarray,
    b_next: np.ndarray,
    b_grid: np.ndarray,
    Ty: np.ndarray,
) -> np.ndarray:
    nb, ny = G.shape
    b_next = np.clip(b_next, b_grid[0], b_grid[-1])
    idx_hi = np.searchsorted(b_grid, b_next, side="right").clip(1, nb - 1)
    idx_lo = idx_hi - 1
    denom = np.maximum(b_grid[idx_hi] - b_grid[idx_lo], 1e-20)
    w_hi = (b_next - b_grid[idx_lo]) / denom
    w_lo = 1.0 - w_hi

    y_idx = np.broadcast_to(np.arange(ny, dtype=int), (nb, ny))
    dest_lo = (idx_lo * ny + y_idx).ravel()
    dest_hi = (idx_hi * ny + y_idx).ravel()
    mass = G.ravel()

    G_next_flat = np.zeros(nb * ny, dtype=float)
    np.add.at(G_next_flat, dest_lo, w_lo.ravel() * mass)
    np.add.at(G_next_flat, dest_hi, w_hi.ravel() * mass)
    G_next = G_next_flat.reshape(nb, ny) @ Ty
    G_next = G_next / np.maximum(G_next.sum(), 1e-20)
    return G_next


def summarize_distribution(
    G: np.ndarray,
    b_grid: np.ndarray,
    borrow_min: float,
) -> Dict[str, float]:
    b_mass = G.sum(axis=1)
    cdf = np.cumsum(b_mass)

    def q_b(q: float) -> float:
        idx = int(np.searchsorted(cdf, q, side="left"))
        idx = min(idx, len(b_grid) - 1)
        return float(b_grid[idx])

    return {
        "mean_b": float(np.sum(b_mass * b_grid)),
        "borrower_share": float(np.sum(b_mass[b_grid < 0.0])),
        "constraint_share": float(np.sum(b_mass[b_grid <= borrow_min + 1e-10])),
        "p50_b": q_b(0.50),
        "p90_b": q_b(0.90),
    }


def simulate_distribution_path(
    policy_grid: np.ndarray,
    b_grid: np.ndarray,
    y_grid: np.ndarray,
    z_grid: np.ndarray,
    r_grid: np.ndarray,
    Ty: np.ndarray,
    z_path: np.ndarray,
    r_path: np.ndarray,
    G0: np.ndarray,
    borrow_min: float,
) -> Dict[str, np.ndarray]:
    T = min(len(z_path), len(r_path))
    G = G0.copy()
    mean_b = np.zeros(T, dtype=float)
    borrower_share = np.zeros(T, dtype=float)
    constraint_share = np.zeros(T, dtype=float)
    p50_b = np.zeros(T, dtype=float)
    p90_b = np.zeros(T, dtype=float)

    for t in range(T):
        c_slice = policy_slice_at_zr(policy_grid, float(z_path[t]), float(r_path[t]), z_grid, r_grid)
        resources = (1.0 + float(r_path[t])) * b_grid[:, None] + y_grid[None, :] * float(z_path[t])
        b_next = np.clip(resources - c_slice, borrow_min, b_grid[-1])
        G = update_distribution(G, b_next, b_grid, Ty)
        stats = summarize_distribution(G, b_grid, borrow_min)
        mean_b[t] = stats["mean_b"]
        borrower_share[t] = stats["borrower_share"]
        constraint_share[t] = stats["constraint_share"]
        p50_b[t] = stats["p50_b"]
        p90_b[t] = stats["p90_b"]

    return {
        "mean_b": mean_b,
        "borrower_share": borrower_share,
        "constraint_share": constraint_share,
        "p50_b": p50_b,
        "p90_b": p90_b,
    }


def save_line_plot(
    x: np.ndarray,
    ys: Tuple[np.ndarray, np.ndarray],
    labels: Tuple[str, str],
    xlabel: str,
    ylabel: str,
    title: str,
    path: str,
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(x, ys[0], label=labels[0], linewidth=2)
    ax.plot(x, ys[1], label=labels[1], linewidth=2, linestyle="--")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark PE vs original Huggett GE")
    parser.add_argument("--ge_dir", type=str, default="hugget_output_g0mix")
    parser.add_argument("--srl_dir", type=str, default=None, help="Backward-compatible alias for --ge_dir")
    parser.add_argument("--out_dir", type=str, default="pe_vs_ge_benchmark")
    parser.add_argument("--y", type=float, default=0.55)
    parser.add_argument("--z", type=float, default=0.94)
    parser.add_argument("--r", type=float, default=0.019)
    parser.add_argument("--beta", type=float, default=0.96)
    parser.add_argument("--sigma", type=float, default=2.0)
    parser.add_argument("--c_min", type=float, default=1e-3)
    parser.add_argument("--rho_r", type=float, default=0.8)
    parser.add_argument("--r_ss", type=float, default=0.038)
    parser.add_argument("--v_r", type=float, default=0.02)
    parser.add_argument("--borrow_min", type=float, default=-1.0)
    parser.add_argument("--max_iter", type=int, default=200)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--curve_points", type=int, default=400)
    parser.add_argument("--sim_seed", type=int, default=42)
    parser.add_argument("--sim_horizon", type=int, default=None)
    args = parser.parse_args()

    if args.srl_dir is not None:
        args.ge_dir = args.srl_dir

    os.makedirs(args.out_dir, exist_ok=True)

    ge_c_grid = np.load(os.path.join(args.ge_dir, "c_grid.npy"))
    b_grid = np.load(os.path.join(args.ge_dir, "b_grid.npy"))
    y_grid = np.load(os.path.join(args.ge_dir, "y_grid.npy"))
    z_grid = np.load(os.path.join(args.ge_dir, "z_grid.npy"))
    r_grid = np.load(os.path.join(args.ge_dir, "r_grid.npy"))
    if abs(float(b_grid[0]) - args.borrow_min) > 1e-10:
        raise ValueError(
            "Borrowing constraint mismatch: GE grid minimum is %.6f but borrow_min is %.6f. "
            "Set borrow_min to the GE grid minimum so PE and GE stay aligned."
            % (float(b_grid[0]), args.borrow_min)
        )
    ge_stats = validate_ge_benchmark(args.ge_dir, r_grid)

    _, Ty = tauchen_ar1(0.6, 0.2, len(y_grid), m=3.0, mean=1.0)
    log_z_grid, Tz = tauchen_ar1(0.9, 0.02, len(z_grid), m=3.0, mean=0.0)
    z_grid_check = np.exp(log_z_grid)
    invariant_z = np.linalg.matrix_power(Tz.T, 200)[:, 0]
    z_grid_check = z_grid_check / (z_grid_check @ invariant_z)
    if np.max(np.abs(z_grid_check - z_grid)) > 1e-6:
        raise ValueError("Loaded GE z grid does not match the expected Huggett calibration grid.")
    Tr = build_r_transition(r_grid, args.rho_r, args.r_ss, args.v_r)

    print("Solving PE household problem on (%d, %d, %d, %d) grid..." % (
        len(b_grid), len(y_grid), len(z_grid), len(r_grid)
    ), flush=True)
    pe_c_grid, pe_bnext_grid, bellman_err = solve_pe_policy(
        b_grid=b_grid,
        y_grid=y_grid,
        z_grid=z_grid,
        r_grid=r_grid,
        Ty=Ty,
        Tz=Tz,
        Tr=Tr,
        beta=args.beta,
        sigma=args.sigma,
        c_min=args.c_min,
        borrow_min=args.borrow_min,
        max_iter=args.max_iter,
        tol=args.tol,
    )

    ge_bnext_grid = np.clip(
        (1.0 + r_grid[None, None, None, :]) * b_grid[:, None, None, None]
        + y_grid[None, :, None, None] * z_grid[None, None, :, None]
        - ge_c_grid,
        args.borrow_min,
        b_grid[-1],
    )

    b_eval = np.linspace(float(b_grid[0]), float(b_grid[-1]), args.curve_points)
    c_ge = eval_policy_curve(
        b_eval=b_eval,
        y_val=args.y,
        z_val=args.z,
        r_val=args.r,
        policy_grid=ge_c_grid,
        b_grid=b_grid,
        y_grid=y_grid,
        z_grid=z_grid,
        r_grid=r_grid,
    )
    c_pe = eval_policy_curve(
        b_eval=b_eval,
        y_val=args.y,
        z_val=args.z,
        r_val=args.r,
        policy_grid=pe_c_grid,
        b_grid=b_grid,
        y_grid=y_grid,
        z_grid=z_grid,
        r_grid=r_grid,
    )
    bnext_ge = np.clip((1.0 + args.r) * b_eval + args.y * args.z - c_ge, args.borrow_min, float(b_grid[-1]))
    bnext_pe = np.clip((1.0 + args.r) * b_eval + args.y * args.z - c_pe, args.borrow_min, float(b_grid[-1]))

    ge_r_path = np.load(os.path.join(args.ge_dir, "r_path.npy"))
    if args.sim_horizon is None:
        sim_horizon = int(len(ge_r_path))
    else:
        sim_horizon = int(args.sim_horizon)
    ge_r_path = ge_r_path[:sim_horizon]

    ge_z_path_file = os.path.join(args.ge_dir, "z_path.npy")
    if os.path.exists(ge_z_path_file):
        common_z_path = np.load(ge_z_path_file)[:sim_horizon]
    else:
        common_z_path = simulate_markov_values(
            T=sim_horizon,
            grid=z_grid,
            trans=Tz,
            start_value=args.z,
            seed=args.sim_seed + 17,
        )

    pe_r_path = simulate_pe_r_path(
        T=sim_horizon,
        r0=args.r,
        rho_r=args.rho_r,
        r_ss=args.r_ss,
        v_r=args.v_r,
        r_min=float(r_grid[0]),
        r_max=float(r_grid[-1]),
        seed=args.sim_seed,
    )

    invariant_y = np.linalg.matrix_power(Ty.T, 200)[:, 0]
    invariant_y = invariant_y / np.maximum(invariant_y.sum(), 1e-20)
    G0 = np.ones((len(b_grid), 1), dtype=float) / len(b_grid)
    G0 = G0 * invariant_y[None, :]
    G0 = G0 / np.maximum(G0.sum(), 1e-20)

    ge_dist = simulate_distribution_path(
        policy_grid=ge_c_grid,
        b_grid=b_grid,
        y_grid=y_grid,
        z_grid=z_grid,
        r_grid=r_grid,
        Ty=Ty,
        z_path=common_z_path,
        r_path=ge_r_path,
        G0=G0,
        borrow_min=args.borrow_min,
    )
    pe_dist = simulate_distribution_path(
        policy_grid=pe_c_grid,
        b_grid=b_grid,
        y_grid=y_grid,
        z_grid=z_grid,
        r_grid=r_grid,
        Ty=Ty,
        z_path=common_z_path,
        r_path=pe_r_path,
        G0=G0,
        borrow_min=args.borrow_min,
    )

    save_line_plot(
        x=b_eval,
        ys=(c_ge, c_pe),
        labels=("GE benchmark", "PE benchmark"),
        xlabel="b",
        ylabel="c(b)",
        title="Consumption policy at y=%.3f, z=%.3f, r=%.3f" % (args.y, args.z, args.r),
        path=os.path.join(args.out_dir, "c_vs_b_ge_vs_pe.png"),
    )
    save_line_plot(
        x=b_eval,
        ys=(bnext_ge, bnext_pe),
        labels=("GE benchmark", "PE benchmark"),
        xlabel="b",
        ylabel="b'(b)",
        title="Savings policy at y=%.3f, z=%.3f, r=%.3f" % (args.y, args.z, args.r),
        path=os.path.join(args.out_dir, "bnext_vs_b_ge_vs_pe.png"),
    )
    save_line_plot(
        x=np.arange(sim_horizon),
        ys=(ge_r_path, pe_r_path),
        labels=("GE realized r_t", "PE exogenous r_t"),
        xlabel="t",
        ylabel="r_t",
        title="Interest-rate paths: GE endogenous vs PE exogenous",
        path=os.path.join(args.out_dir, "r_paths_ge_vs_pe.png"),
    )
    save_line_plot(
        x=np.arange(sim_horizon),
        ys=(ge_dist["mean_b"], pe_dist["mean_b"]),
        labels=("GE distribution", "PE distribution"),
        xlabel="t",
        ylabel="mean b",
        title="Mean assets over time from common initial distribution",
        path=os.path.join(args.out_dir, "mean_assets_ge_vs_pe.png"),
    )
    save_line_plot(
        x=np.arange(sim_horizon),
        ys=(ge_dist["borrower_share"], pe_dist["borrower_share"]),
        labels=("GE distribution", "PE distribution"),
        xlabel="t",
        ylabel="share with b < 0",
        title="Borrower share over time from common initial distribution",
        path=os.path.join(args.out_dir, "borrower_share_ge_vs_pe.png"),
    )
    save_line_plot(
        x=np.arange(sim_horizon),
        ys=(ge_dist["constraint_share"], pe_dist["constraint_share"]),
        labels=("GE distribution", "PE distribution"),
        xlabel="t",
        ylabel="share at borrowing constraint",
        title="Constraint mass over time from common initial distribution",
        path=os.path.join(args.out_dir, "constraint_share_ge_vs_pe.png"),
    )

    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    ax.plot(bellman_err, color="tab:blue")
    ax.set_xlabel("VFI iteration")
    ax.set_ylabel("sup-norm error")
    ax.set_title("PE Bellman iteration error")
    ax.set_yscale("log")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "pe_bellman_error.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)

    np.save(os.path.join(args.out_dir, "pe_c_grid.npy"), pe_c_grid)
    np.save(os.path.join(args.out_dir, "pe_bnext_grid.npy"), pe_bnext_grid)
    np.save(os.path.join(args.out_dir, "pe_r_transition.npy"), Tr)
    np.save(os.path.join(args.out_dir, "pe_r_path.npy"), pe_r_path)
    np.save(os.path.join(args.out_dir, "ge_r_path_used.npy"), ge_r_path)
    np.save(os.path.join(args.out_dir, "z_path_used.npy"), common_z_path)
    np.savez(
        os.path.join(args.out_dir, "distribution_paths.npz"),
        ge_mean_b=ge_dist["mean_b"],
        pe_mean_b=pe_dist["mean_b"],
        ge_borrower_share=ge_dist["borrower_share"],
        pe_borrower_share=pe_dist["borrower_share"],
        ge_constraint_share=ge_dist["constraint_share"],
        pe_constraint_share=pe_dist["constraint_share"],
        ge_p50_b=ge_dist["p50_b"],
        pe_p50_b=pe_dist["p50_b"],
        ge_p90_b=ge_dist["p90_b"],
        pe_p90_b=pe_dist["p90_b"],
    )

    summary_path = os.path.join(args.out_dir, "comparison_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Focused comparison state: y=%.6f z=%.6f r=%.6f\n" % (args.y, args.z, args.r))
        f.write("Borrowing constraint in both models: b' >= %.6f\n" % args.borrow_min)
        f.write("\n")
        f.write("GE equations and flow:\n")
        f.write("  - Household policy loaded from saved GE benchmark c_GE(b,y,z,r).\n")
        f.write("  - Interest rate is endogenous in GE and comes from period-by-period bond-market clearing.\n")
        f.write("  - This benchmark uses the saved GE simulation path rather than altering the GE solver.\n")
        f.write("\n")
        f.write("PE equations and flow:\n")
        f.write("  - Household policy solved on the same (b,y,z,r) grid as GE.\n")
        f.write("  - y_t and z_t follow the same Markov processes as the GE calibration.\n")
        f.write("  - r_{t+1} = (1-rho_r) * r_ss + rho_r * r_t + v_r * sqrt(max(0,r_t)) * eps_t.\n")
        f.write("  - The PE household takes the exogenous r transition matrix as given.\n")
        f.write("\n")
        f.write("Conceptual mismatch fixed in this script:\n")
        f.write("  - The previous PE comparison held y and z fixed in the Bellman problem, so it was not benchmarking PE and GE on the same state space.\n")
        f.write("  - The updated PE code now keeps stochastic y_t and z_t and aligns the borrowing constraint exactly with GE.\n")
        f.write("\n")
        f.write("Numerical comparison at the requested fixed state:\n")
        f.write("  - Mean |c_GE - c_PE| over b grid: %.6e\n" % float(np.mean(np.abs(c_ge - c_pe))))
        f.write("  - Max  |c_GE - c_PE| over b grid: %.6e\n" % float(np.max(np.abs(c_ge - c_pe))))
        f.write("  - Mean |b'_GE - b'_PE| over b grid: %.6e\n" % float(np.mean(np.abs(bnext_ge - bnext_pe))))
        f.write("  - Max  |b'_GE - b'_PE| over b grid: %.6e\n" % float(np.max(np.abs(bnext_ge - bnext_pe))))
        f.write("\n")
        f.write("Distribution simulation setup:\n")
        f.write("  - Common initial distribution: uniform over b and invariant over y.\n")
        f.write("  - Common z path source: %s\n" % ("GE saved z_path.npy" if os.path.exists(ge_z_path_file) else "simulated from GE-style Tz"))
        f.write("  - GE uses saved endogenous r path; PE uses exogenous r path starting from r0=%.6f.\n" % args.r)
        f.write("\n")
        f.write("Distribution summaries at final simulation date:\n")
        f.write("  - GE mean b: %.6f\n" % ge_dist["mean_b"][-1])
        f.write("  - PE mean b: %.6f\n" % pe_dist["mean_b"][-1])
        f.write("  - GE borrower share: %.6f\n" % ge_dist["borrower_share"][-1])
        f.write("  - PE borrower share: %.6f\n" % pe_dist["borrower_share"][-1])
        f.write("  - GE constraint share: %.6f\n" % ge_dist["constraint_share"][-1])
        f.write("  - PE constraint share: %.6f\n" % pe_dist["constraint_share"][-1])
        f.write("\n")
        f.write("Interest-rate path summaries:\n")
        f.write("  - GE mean/std r_t: %.6f / %.6f\n" % (float(np.mean(ge_r_path)), float(np.std(ge_r_path))))
        f.write("  - PE mean/std r_t: %.6f / %.6f\n" % (float(np.mean(pe_r_path)), float(np.std(pe_r_path))))
        f.write("  - PE path min/max r_t: %.6f / %.6f\n" % (float(np.min(pe_r_path)), float(np.max(pe_r_path))))
        if "ge_max_abs_market_residual" in ge_stats:
            f.write("  - GE max abs market-clearing residual in benchmark: %.6e\n" % ge_stats["ge_max_abs_market_residual"])
        f.write("\n")
        f.write("Economic interpretation in this implementation:\n")
        f.write("  - In PE, households face exogenous mean-reverting rate risk and do not move r_t through their aggregate savings.\n")
        f.write("  - In GE, the same household states map into a policy that is evaluated at endogenous rates coming from market clearing.\n")
        f.write("  - Differences in c(b) and b'(b) therefore reflect the fact that PE prices are taken as given, while GE prices feed back from the cross-sectional distribution and bond demand.\n")
        if bellman_err.size > 0:
            f.write("  - Final PE Bellman sup-norm error: %.6e after %d iterations.\n" % (float(bellman_err[-1]), int(bellman_err.size)))

    print("Saved benchmark comparison to %s" % args.out_dir, flush=True)
    print("Focused mean |c_GE-c_PE| = %.3e" % float(np.mean(np.abs(c_ge - c_pe))), flush=True)


if __name__ == "__main__":
    main()
