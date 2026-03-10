#!/usr/bin/env python3
"""
Compare SRL policy c(b|y,z,r) from saved grid with a partial-equilibrium (PE) policy.

PE rate process:
  r_{t+1} = (1-rho_r) * r_ss + rho_r * r_t + v_r * sqrt(max(0, r_t)) * eps_t
  eps_t ~ N(0,1)
"""
from __future__ import annotations

import argparse
import os
from typing import Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


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


def eval_srl_c_b_curve(
    b_eval: np.ndarray,
    y_val: float,
    z_val: float,
    r_val: float,
    c_grid: np.ndarray,   # (nb, ny, nz, nr)
    b_grid: np.ndarray,
    y_grid: np.ndarray,
    z_grid: np.ndarray,
    r_grid: np.ndarray,
) -> np.ndarray:
    iy_lo, iy_hi, wy = interp_weights_1d(y_grid, y_val)
    iz_lo, iz_hi, wz = interp_weights_1d(z_grid, z_val)
    ir_lo, ir_hi, wr = interp_weights_1d(r_grid, r_val)

    c_by_state = np.zeros_like(b_eval, dtype=float)
    for k, b in enumerate(b_eval):
        ib_lo, ib_hi, wb = interp_weights_1d(b_grid, float(b))

        def c_at(iy: int, iz: int, ir: int) -> float:
            return (1.0 - wb) * c_grid[ib_lo, iy, iz, ir] + wb * c_grid[ib_hi, iy, iz, ir]

        c_yzr = 0.0
        for iy, wy_i in ((iy_lo, 1.0 - wy), (iy_hi, wy)):
            for iz, wz_i in ((iz_lo, 1.0 - wz), (iz_hi, wz)):
                c_r = (1.0 - wr) * c_at(iy, iz, ir_lo) + wr * c_at(iy, iz, ir_hi)
                c_yzr += wy_i * wz_i * c_r
        c_by_state[k] = c_yzr
    return c_by_state


def eval_srl_bnext_b_curve(
    b_eval: np.ndarray,
    y_val: float,
    z_val: float,
    r_val: float,
    c_curve: np.ndarray,
    b_min: float,
    b_max: float,
) -> np.ndarray:
    resources = (1.0 + r_val) * b_eval + y_val * z_val
    return np.clip(resources - c_curve, b_min, b_max)


def build_r_transition(
    r_grid: np.ndarray,
    rho_r: float,
    r_ss: float,
    v_r: float,
    quad_n: int = 41,
) -> np.ndarray:
    eps = np.linspace(-4.0, 4.0, quad_n)
    w = np.exp(-0.5 * eps**2)
    w = w / w.sum()

    nr = len(r_grid)
    P = np.zeros((nr, nr), dtype=float)
    for i, r in enumerate(r_grid):
        r_next = (1.0 - rho_r) * r_ss + rho_r * r + v_r * np.sqrt(max(0.0, r)) * eps
        r_next = np.clip(r_next, r_grid[0], r_grid[-1])
        for rn, wn in zip(r_next, w):
            j_hi = int(np.searchsorted(r_grid, rn, side="right"))
            if j_hi <= 0:
                P[i, 0] += wn
            elif j_hi >= nr:
                P[i, nr - 1] += wn
            else:
                j_lo = j_hi - 1
                alpha = (rn - r_grid[j_lo]) / max(r_grid[j_hi] - r_grid[j_lo], 1e-20)
                P[i, j_lo] += wn * (1.0 - alpha)
                P[i, j_hi] += wn * alpha
    P = P / np.maximum(P.sum(axis=1, keepdims=True), 1e-20)
    return P


def solve_pe_policy(
    b_grid: np.ndarray,
    r_grid: np.ndarray,
    P_r: np.ndarray,
    y_val: float,
    z_val: float,
    beta: float,
    sigma: float,
    c_min: float,
    borrow_min: float,
    max_iter: int = 400,
    tol: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    nb, nr = len(b_grid), len(r_grid)
    V = np.zeros((nb, nr), dtype=float)
    policy_idx = np.zeros((nb, nr), dtype=int)

    # Enforce borrowing constraint b' >= borrow_min explicitly.
    feasible_next = np.maximum(b_grid, borrow_min)
    b_next_candidates = feasible_next[None, None, :]  # (1,1,nb)
    b_cur = b_grid[:, None, None]              # (nb,1,1)
    r_cur = r_grid[None, :, None]              # (1,nr,1)

    resources = (1.0 + r_cur) * b_cur + (y_val * z_val)
    c_all = resources - b_next_candidates
    util = np.where(c_all > c_min, crra_u(c_all, sigma, c_min), -1e12)

    for _ in range(max_iter):
        # EV_by_r_choice[j, k] = E[V(b'_k, r') | r_t = r_j]
        EV_by_r_choice = P_r @ V.T  # (nr, nb_choice)
        rhs = util + beta * EV_by_r_choice[None, :, :]  # (nb, nr, nb_choice)
        new_policy = np.argmax(rhs, axis=2)
        V_new = np.take_along_axis(rhs, new_policy[:, :, None], axis=2)[:, :, 0]
        err = float(np.max(np.abs(V_new - V)))
        V = V_new
        policy_idx = new_policy
        if err < tol:
            break

    b_next = feasible_next[policy_idx]
    c_policy = np.maximum((1.0 + r_grid[None, :]) * b_grid[:, None] + y_val * z_val - b_next, c_min)
    return c_policy, policy_idx


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare SRL c(b) with PE c(b)")
    parser.add_argument("--srl_dir", type=str, default="hugget_output_hpc_full_v3")
    parser.add_argument("--out_dir", type=str, default="hugget_compare")
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
    parser.add_argument("--nr_pe", type=int, default=41)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    c_grid = np.load(os.path.join(args.srl_dir, "c_grid.npy"))
    b_grid = np.load(os.path.join(args.srl_dir, "b_grid.npy"))
    y_grid = np.load(os.path.join(args.srl_dir, "y_grid.npy"))
    z_grid = np.load(os.path.join(args.srl_dir, "z_grid.npy"))
    r_grid = np.load(os.path.join(args.srl_dir, "r_grid.npy"))

    b_eval = np.linspace(float(b_grid[0]), float(b_grid[-1]), 400)
    ge_borrow_min = float(b_grid[0])
    c_srl = eval_srl_c_b_curve(
        b_eval=b_eval,
        y_val=args.y,
        z_val=args.z,
        r_val=args.r,
        c_grid=c_grid,
        b_grid=b_grid,
        y_grid=y_grid,
        z_grid=z_grid,
        r_grid=r_grid,
    )

    r_lo = max(1e-4, args.r_ss - 0.02)
    r_hi = args.r_ss + 0.02
    r_grid_pe = np.linspace(r_lo, r_hi, args.nr_pe)
    P_r = build_r_transition(r_grid_pe, args.rho_r, args.r_ss, args.v_r)
    c_pe_grid, _ = solve_pe_policy(
        b_grid=b_grid,
        r_grid=r_grid_pe,
        P_r=P_r,
        y_val=args.y,
        z_val=args.z,
        beta=args.beta,
        sigma=args.sigma,
        c_min=args.c_min,
        borrow_min=args.borrow_min,
    )

    ir_lo, ir_hi, wr = interp_weights_1d(r_grid_pe, args.r)
    c_pe_at_r = (1.0 - wr) * c_pe_grid[:, ir_lo] + wr * c_pe_grid[:, ir_hi]
    c_pe = np.interp(b_eval, b_grid, c_pe_at_r)
    bnext_srl = eval_srl_bnext_b_curve(
        b_eval=b_eval,
        y_val=args.y,
        z_val=args.z,
        r_val=args.r,
        c_curve=c_srl,
        b_min=ge_borrow_min,
        b_max=float(b_grid[-1]),
    )
    bnext_pe = np.clip((1.0 + args.r) * b_eval + args.y * args.z - c_pe, args.borrow_min, float(b_grid[-1]))

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(b_eval, c_srl, label="SRL", linewidth=2)
    ax.plot(b_eval, c_pe, label="PE (exogenous r process)", linewidth=2, linestyle="--")
    ax.set_xlabel("b")
    ax.set_ylabel("c(b)")
    ax.set_title(
        "c(b) at y=%.3f, z=%.3f, r=%.3f" % (args.y, args.z, args.r)
    )
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    out_png = os.path.join(args.out_dir, "c_vs_b_srl_vs_pe.png")
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)

    out_txt = os.path.join(args.out_dir, "compare_summary.txt")
    with open(out_txt, "w") as f:
        f.write("Requested state: y=%.6f z=%.6f r=%.6f\n" % (args.y, args.z, args.r))
        f.write("SRL grids range: y=[%.6f, %.6f], z=[%.6f, %.6f], r=[%.6f, %.6f]\n" %
                (y_grid.min(), y_grid.max(), z_grid.min(), z_grid.max(), r_grid.min(), r_grid.max()))
        f.write("PE rate process: r'=(1-rho_r)r_ss+rho_r*r+v_r*sqrt(max(0,r))*eps\n")
        f.write("rho_r=%.6f r_ss=%.6f v_r=%.6f\n" % (args.rho_r, args.r_ss, args.v_r))
        f.write("Borrowing constraint in GE: b' >= %.6f (from SRL grid minimum)\n" % ge_borrow_min)
        f.write("Borrowing constraint in PE: b' >= %.6f\n" % args.borrow_min)
        if abs(ge_borrow_min - args.borrow_min) > 1e-10:
            f.write("WARNING: GE and PE borrow constraints differ.\n")
        else:
            f.write("GE and PE borrow constraints are aligned.\n")
        f.write("GE note: SRL/GE clears market each period via aggregate bond condition.\n")
        f.write("Check min b'_GE on plotted curve: %.6f\n" % float(np.min(bnext_srl)))
        f.write("Check min b'_PE on plotted curve: %.6f\n" % float(np.min(bnext_pe)))
        f.write("Mean |c_srl-c_pe| on eval grid: %.6e\n" % float(np.mean(np.abs(c_srl - c_pe))))
        f.write("Max  |c_srl-c_pe| on eval grid: %.6e\n" % float(np.max(np.abs(c_srl - c_pe))))
    print("Saved:", out_png)
    print("Saved:", out_txt)


if __name__ == "__main__":
    main()
