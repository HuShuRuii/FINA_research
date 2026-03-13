#!/usr/bin/env python3
"""
Forward-Looking Huggett (PyTorch version).

Policy is truly forward-looking:
  c_t = pi(b, y, r_t, p_{t+1}, z_t)
with policy tensor shape (nz, nr_cur, nr_next, nb, ny).

Inner-loop rule:
  - k=0: assume p_{t+1}=p_t
  - t=T-1: also use p_{t+1}=p_t
  - otherwise use p_trajectory[t+1]

Important:
  Within each outer epoch, all inner iterations share the same z trajectory.
"""
from __future__ import annotations

import argparse
import os
import time
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import torch


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


def get_calibration(quick: bool = False):
    cal = {
        "beta": 0.96, "sigma": 2.0, "rho_y": 0.6, "nu_y": 0.2, "rho_z": 0.9, "nu_z": 0.02,
        "B": 0.0, "b_min": -1.0, "b_max": 50.0, "nb": 200, "ny": 3, "nr": 20, "nz": 30,
        "r_min": 0.01, "r_max": 0.06, "c_min": 1e-3, "e_trunc": 1e-3, "lr_ini": 1e-3,
        "p_damping": 0.1, "p_init": 0.038,
    }
    if quick:
        cal.update({"N_epoch_outer": 10, "N_p": 3, "N_theta_per_p": 2, "T_traj": 20, "N_sample": 4, "N_z_test": 2})
    else:
        cal.update({"N_epoch_outer": 120, "N_p": 10, "N_theta_per_p": 5, "T_traj": 170, "N_sample": 20, "N_z_test": 4})
    return cal


def theta_to_c(theta: torch.Tensor, c_min: float) -> torch.Tensor:
    return torch.clamp(theta, min=c_min)


def init_theta(b_grid, y_grid, z_grid, r_grid, save_frac=0.2, c_min=1e-3):
    nb, ny = b_grid.numel(), y_grid.numel()
    nz, nr = z_grid.numel(), r_grid.numel()
    b_flat = b_grid.repeat_interleave(ny)
    y_flat = y_grid.repeat(nb)
    cash = b_flat[None, None, :] * (1 + r_grid[None, :, None]) + y_flat[None, None, :] * z_grid[:, None, None]
    c_base = torch.clamp(cash.reshape(nz, nr, nb, ny) * (1 - save_frac), min=c_min)
    return c_base[:, :, None, :, :].expand(nz, nr, nr, nb, ny).contiguous()


def u(c: torch.Tensor, sigma: float, c_min: float):
    c = torch.clamp(c, min=c_min)
    if abs(sigma - 1.0) < 1e-8:
        return torch.log(c)
    return (c ** (1 - sigma)) / (1 - sigma)


def interp_weights_1d(grid: torch.Tensor, x: torch.Tensor):
    n = grid.numel()
    if n == 1:
        return 0, 0, torch.zeros((), device=grid.device, dtype=grid.dtype)
    x_clip = torch.clamp(x, min=grid[0], max=grid[-1])
    hi = int(torch.searchsorted(grid, x_clip, right=True).clamp(1, n - 1).item())
    lo = hi - 1
    denom = torch.clamp(grid[hi] - grid[lo], min=1e-20)
    w = (x_clip - grid[lo]) / denom
    return lo, hi, w


def interpolate_c_at_prices(theta: torch.Tensor, iz: int, r_cur_val: torch.Tensor, p_next_val: torch.Tensor,
                            r_grid: torch.Tensor, c_min: float) -> torch.Tensor:
    c = theta_to_c(theta, c_min)
    ir_lo, ir_hi, w_r = interp_weights_1d(r_grid, r_cur_val)
    ip_lo, ip_hi, w_p = interp_weights_1d(r_grid, p_next_val)
    c00 = c[iz, ir_lo, ip_lo, :, :]
    c01 = c[iz, ir_lo, ip_hi, :, :]
    c10 = c[iz, ir_hi, ip_lo, :, :]
    c11 = c[iz, ir_hi, ip_hi, :, :]
    c_lo = (1.0 - w_p) * c00 + w_p * c01
    c_hi = (1.0 - w_p) * c10 + w_p * c11
    return (1.0 - w_r) * c_lo + w_r * c_hi


def update_G_from_c_and_r(c_t, r_star, G, iz, b_grid, y_grid, z_grid, Ty, b_min, b_max):
    nb, ny = b_grid.numel(), y_grid.numel()
    b_next = (1 + r_star) * b_grid.repeat_interleave(ny) + y_grid.repeat(nb) * z_grid[iz] - c_t.reshape(-1)
    b_next = torch.clamp(b_next, b_min, b_max)

    idx_hi = torch.searchsorted(b_grid, b_next).clamp(1, nb - 1)
    idx_lo = idx_hi - 1
    denom = torch.clamp(b_grid[idx_hi] - b_grid[idx_lo], min=1e-20)
    w_hi = (b_next - b_grid[idx_lo]) / denom
    w_lo = 1.0 - w_hi

    G_flat = G.reshape(-1)
    dest_lo = idx_lo * ny + torch.arange(ny, device=G.device).repeat(nb)
    dest_hi = idx_hi * ny + torch.arange(ny, device=G.device).repeat(nb)
    G_new_flat = torch.zeros_like(G_flat)
    G_new_flat.scatter_add_(0, dest_lo, w_lo * G_flat)
    G_new_flat.scatter_add_(0, dest_hi, w_hi * G_flat)
    G_new = G_new_flat.reshape(nb, ny) @ Ty
    return G_new / (G_new.sum() + 1e-20)


def market_clearing_stats(theta, G, iz, p_next_val, use_same_r, b_grid, y_grid, z_grid, r_grid, B, b_min, b_max, c_min):
    nb, ny = b_grid.numel(), y_grid.numel()
    nr = r_grid.numel()
    c = theta_to_c(theta, c_min)
    b_flat = b_grid.repeat_interleave(ny)
    y_flat = y_grid.repeat(nb)
    resources = b_flat[:, None] * (1 + r_grid[None, :]) + (y_flat * z_grid[iz])[:, None]
    ir_idx = torch.arange(nr, device=theta.device)
    if use_same_r:
        c_slice = c[iz, ir_idx, ir_idx, :, :]
    else:
        ip_lo, ip_hi, w_p = interp_weights_1d(r_grid, p_next_val)
        c_lo = c[iz, ir_idx, ip_lo, :, :]
        c_hi = c[iz, ir_idx, ip_hi, :, :]
        c_slice = (1.0 - w_p) * c_lo + w_p * c_hi
    c_slice = c_slice.permute(1, 2, 0).reshape(nb * ny, nr)
    b_next_all = torch.clamp(resources - c_slice, b_min, b_max).reshape(nb, ny, nr)
    S_all = (G[:, :, None] * b_next_all).sum(dim=(0, 1))

    ge = S_all >= B
    if bool(ge.any().item()):
        first_ge = int(torch.argmax(ge.to(torch.int32)).item())
    else:
        first_ge = nr - 1
    ir_hi = min(max(first_ge, 1), nr - 1)
    ir_lo = max(ir_hi - 1, 0)
    S_lo = S_all[ir_lo]
    S_hi = S_all[ir_hi]
    w_r = torch.clamp((B - S_lo) / torch.clamp(S_hi - S_lo, min=1e-20), 0.0, 1.0)
    r_star = r_grid[ir_lo] + w_r * (r_grid[ir_hi] - r_grid[ir_lo])
    S_star = S_lo + w_r * (S_hi - S_lo)
    residual = S_star - B
    return r_star, ir_lo, ir_hi, w_r, residual


def steady_state_G0(theta, b_grid, y_grid, z_grid, r_grid, Ty, b_min, b_max, c_min, n_iter: int = 150, tol: float = 1e-6):
    nb, ny = b_grid.numel(), y_grid.numel()
    iz_mid = z_grid.numel() // 2
    ir_mid = r_grid.numel() // 2 if r_grid.numel() > 0 else 0
    G = torch.ones(nb, ny, dtype=theta.dtype, device=theta.device) / (nb * ny)
    c_mid = theta_to_c(theta, c_min)[iz_mid, ir_mid, ir_mid, :, :]
    r_mid = r_grid[ir_mid]
    for _ in range(n_iter):
        G_new = update_G_from_c_and_r(c_mid, r_mid, G, iz_mid, b_grid, y_grid, z_grid, Ty, b_min, b_max)
        if float(torch.max(torch.abs(G_new - G)).item()) < tol:
            G = G_new
            break
        G = G_new
    return G


def init_p_path_values(T_traj: int, p_init_val: float) -> np.ndarray:
    return np.full((T_traj + 1,), p_init_val, dtype=np.float32)


def damped_update_p_path(p_old: np.ndarray, r_realized: np.ndarray, alpha: float) -> np.ndarray:
    p_target = np.array(p_old, copy=True)
    p_target[:-1] = r_realized
    p_target[-1] = r_realized[-1]
    return ((1.0 - alpha) * p_old + alpha * p_target).astype(np.float32)


def generate_z_trajectory(T: int, nz: int, Tz_np: np.ndarray) -> np.ndarray:
    iz = np.random.randint(0, nz)
    out = []
    for _ in range(T):
        out.append(iz)
        iz = np.random.choice(nz, p=Tz_np[iz, :])
    return np.asarray(out, dtype=np.int64)


def objective_one_trajectory(theta, z_path, p_path_vals, G0, b_grid, y_grid, z_grid, r_grid, Ty, cal):
    beta, sigma = cal["beta"], cal["sigma"]
    e_trunc, B = cal["e_trunc"], cal["B"]
    b_min, b_max, c_min = cal["b_min"], cal["b_max"], cal["c_min"]
    T = len(z_path)
    G = G0.clone()
    L = torch.tensor(0.0, device=theta.device, dtype=theta.dtype)
    r_realized = []
    residuals = []
    boundary_hits = []
    for t in range(T):
        iz = int(z_path[t])
        is_last = (t == T - 1)
        p_next_val = torch.tensor(float(p_path_vals[t + 1]), dtype=theta.dtype, device=theta.device)
        r_t, ir_lo, ir_hi, w_r, residual = market_clearing_stats(
            theta, G.detach(), iz, p_next_val, bool(is_last), b_grid, y_grid, z_grid, r_grid, B, b_min, b_max, c_min
        )
        p_eval = r_t.detach() if is_last else p_next_val
        c_t = interpolate_c_at_prices(theta, iz, r_t.detach(), p_eval, r_grid, c_min)
        w = (beta ** t) * (1.0 if (beta ** t) >= e_trunc else 0.0)
        L = L + w * (G.detach() * u(c_t, sigma, c_min)).sum()
        G = update_G_from_c_and_r(c_t, r_t.detach(), G.detach(), iz, b_grid, y_grid, z_grid, Ty, b_min, b_max).detach()
        r_realized.append(float(r_t.detach().item()))
        residuals.append(float(residual.detach().item()))
        boundary_hit = ((r_t.detach() <= r_grid[0] + 1e-12) | (r_t.detach() >= r_grid[-1] - 1e-12)).item()
        boundary_hits.append(float(boundary_hit))
    return L, np.asarray(r_realized), np.asarray(residuals), np.asarray(boundary_hits)


def run_inner_convergence(theta, z_path, N_p, r_grid_np, G0, b_grid, y_grid, z_grid, r_grid, Ty, cal):
    p_path = init_p_path_values(len(z_path), cal["p_init"])
    losses = []
    r_paths = []
    residual_paths = []
    boundary_paths = []
    p_paths = []
    for k in range(N_p):
        L, r_realized, residuals, boundary_hits = objective_one_trajectory(
            theta, z_path, p_path, G0, b_grid, y_grid, z_grid, r_grid, Ty, cal
        )
        losses.append(float(L.detach().cpu()))
        r_paths.append(r_realized)
        residual_paths.append(residuals)
        boundary_paths.append(boundary_hits)
        p_paths.append(np.asarray(p_path[:-1], dtype=float))
        p_path = damped_update_p_path(p_path, r_realized, cal["p_damping"])
    losses = np.asarray(losses)
    r_paths = np.asarray(r_paths)
    residual_paths = np.asarray(residual_paths)
    boundary_paths = np.asarray(boundary_paths)
    p_paths = np.asarray(p_paths)
    if r_paths.shape[0] >= 2:
        delta = np.max(np.abs(r_paths[1:] - r_paths[:-1]), axis=1)
    else:
        delta = np.zeros((0,), dtype=float)
    return losses, r_paths, residual_paths, boundary_paths, p_paths, delta


def main():
    p = argparse.ArgumentParser(description="Forward-looking Huggett (PyTorch)")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--out_dir", type=str, default="forward_looking_hugget_output_py")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_sample", type=int, default=None)
    p.add_argument("--n_p", type=int, default=None)
    p.add_argument("--theta_steps_per_p", type=int, default=None)
    p.add_argument("--p_damping", type=float, default=None)
    p.add_argument("--p_init", type=float, default=None)
    args = p.parse_args()

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

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    y_grid_np, Ty_np = tauchen_ar1(cal["rho_y"], cal["nu_y"], cal["ny"], m=3, mean=1.0)
    y_grid_np = y_grid_np / (y_grid_np @ np.linalg.matrix_power(Ty_np.T, 200)[:, 0])
    log_z, Tz_np = tauchen_ar1(cal["rho_z"], cal["nu_z"], cal["nz"])
    z_grid_np = np.exp(log_z)
    z_grid_np = z_grid_np / (z_grid_np @ np.linalg.matrix_power(Tz_np.T, 200)[:, 0])

    nb_spg, nr_spg, nz_spg = cal["nb"], cal["nr"], cal["nz"]
    z_sub = z_grid_np
    Tz_sub = Tz_np

    b_grid = torch.tensor(np.linspace(cal["b_min"], cal["b_max"], nb_spg), dtype=torch.float32, device=device)
    r_grid = torch.tensor(np.linspace(cal["r_min"], cal["r_max"], nr_spg), dtype=torch.float32, device=device)
    y_grid = torch.tensor(y_grid_np, dtype=torch.float32, device=device)
    z_grid = torch.tensor(z_sub, dtype=torch.float32, device=device)
    Ty = torch.tensor(Ty_np, dtype=torch.float32, device=device)
    theta = init_theta(b_grid, y_grid, z_grid, r_grid, save_frac=0.2, c_min=cal["c_min"]).to(device)
    theta = torch.nn.Parameter(theta)
    G0 = steady_state_G0(theta.detach(), b_grid, y_grid, z_grid, r_grid, Ty, cal["b_min"], cal["b_max"], cal["c_min"])
    opt = torch.optim.Adam([theta], lr=cal["lr_ini"])

    r_grid_np = np.asarray(r_grid.detach().cpu())
    loss_hist = []
    mean_abs_resid_hist = []
    boundary_share_hist = []

    print(
        f"Forward-Looking Huggett PY: device={device}, N_sample={cal['N_sample']}, "
        f"N_p={cal['N_p']}, N_theta_per_p={cal['N_theta_per_p']}, "
        f"N_epoch={cal['N_epoch_outer']}, T={cal['T_traj']}"
    )
    print("G0 initialized from mid-(z,r) invariant distribution")
    print(f"Damped p update: alpha={cal['p_damping']:.3f}, p_init={cal['p_init']:.4f}")
    t0 = time.perf_counter()
    for ep in range(cal["N_epoch_outer"]):
        ep_losses = []
        ep_abs_resid = []
        ep_boundary_share = []
        for _n in range(cal["N_sample"]):
            # One z trajectory per sample; reused by all inner k.
            z_path = generate_z_trajectory(cal["T_traj"], nz_spg, Tz_sub)
            p_path = init_p_path_values(cal["T_traj"], cal["p_init"])
            for k in range(cal["N_p"]):
                for _ in range(cal["N_theta_per_p"]):
                    opt.zero_grad(set_to_none=True)
                    L, _, _, _ = objective_one_trajectory(
                        theta, z_path, p_path, G0, b_grid, y_grid, z_grid, r_grid, Ty, cal
                    )
                    (-L).backward()
                    opt.step()
                L, r_realized, residuals, boundary_hits = objective_one_trajectory(
                    theta, z_path, p_path, G0, b_grid, y_grid, z_grid, r_grid, Ty, cal
                )
                ep_losses.append(float(L.detach().cpu()))
                ep_abs_resid.append(float(np.mean(np.abs(residuals))))
                ep_boundary_share.append(float(np.mean(boundary_hits)))
                p_path = damped_update_p_path(p_path, r_realized, cal["p_damping"])
        loss_hist.append(float(np.mean(ep_losses)))
        mean_abs_resid_hist.append(float(np.mean(ep_abs_resid)))
        boundary_share_hist.append(float(np.mean(ep_boundary_share)))
        if ep == 0 or (ep + 1) % 10 == 0:
            print(
                f"Epoch {ep+1}, mean L = {loss_hist[-1]:.4f}, "
                f"mean |mc residual| = {mean_abs_resid_hist[-1]:.4e}, "
                f"boundary share = {boundary_share_hist[-1]:.3f}"
            )
    print(f"Training done in {time.perf_counter()-t0:.2f}s")

    os.makedirs(args.out_dir, exist_ok=True)
    test_z = [generate_z_trajectory(cal["T_traj"], nz_spg, Tz_sub) for _ in range(cal["N_z_test"])]
    conv_results = []
    for i, z_path in enumerate(test_z):
        losses, r_paths, residual_paths, boundary_paths, p_paths, delta = run_inner_convergence(
            theta, z_path, cal["N_p"], r_grid_np, G0, b_grid, y_grid, z_grid, r_grid, Ty, cal
        )
        conv_results.append({
            "losses": losses,
            "r_paths": r_paths,
            "residual_paths": residual_paths,
            "boundary_paths": boundary_paths,
            "p_paths": p_paths,
            "delta": delta,
        })
        print(
            f"Test trajectory {i+1}: L first={losses[0]:.4f}, last={losses[-1]:.4f}, "
            f"last |mc residual|={np.mean(np.abs(residual_paths[-1])):.4e}, "
            f"last boundary share={np.mean(boundary_paths[-1]):.3f}"
        )
        t_axis = np.arange(r_paths.shape[1])
        fig, axs = plt.subplots(2, 1, figsize=(10, 8))
        for k in range(r_paths.shape[0]):
            alpha = 0.25 + 0.75 * (k + 1) / max(r_paths.shape[0], 1)
            axs[0].plot(t_axis, r_paths[k], alpha=alpha, linewidth=1.2, label=f"k={k}")
        axs[0].set_title(f"All inner r trajectories (fixed z path {i+1})")
        axs[0].set_xlabel("t")
        axs[0].set_ylabel("r_t")
        axs[0].grid(alpha=0.3)
        axs[0].legend(ncol=4, fontsize=8)

        if delta.size > 0:
            axs[1].plot(np.arange(1, r_paths.shape[0]), delta, "-o")
        axs[1].set_title("Inner-loop trajectory convergence distance")
        axs[1].set_xlabel("inner iteration k")
        axs[1].set_ylabel("max_t |r^(k)-r^(k-1)|")
        axs[1].grid(alpha=0.3)
        plt.tight_layout()
        fig.savefig(os.path.join(args.out_dir, f"trajectory_convergence_traj{i+1:02d}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        np.save(os.path.join(args.out_dir, f"conv_r_paths_traj{i+1:02d}.npy"), r_paths)
        np.save(os.path.join(args.out_dir, f"conv_residual_paths_traj{i+1:02d}.npy"), residual_paths)
        np.save(os.path.join(args.out_dir, f"conv_boundary_paths_traj{i+1:02d}.npy"), boundary_paths)
        np.save(os.path.join(args.out_dir, f"conv_p_paths_traj{i+1:02d}.npy"), p_paths)
        np.save(os.path.join(args.out_dir, f"conv_delta_traj{i+1:02d}.npy"), delta)

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    for traj_idx, cres in enumerate(conv_results):
        axs[0, 0].plot(np.arange(cal["N_p"]), cres["losses"], label=f"z_traj {traj_idx+1}")
    axs[0, 0].set_xlabel("Inner iteration k")
    axs[0, 0].set_ylabel("L(theta)")
    axs[0, 0].set_title("Fixed-theta loss over inner loop")
    axs[0, 0].grid(alpha=0.3)
    axs[0, 0].legend(fontsize=8)

    t_axis = np.arange(cal["T_traj"])
    for traj_idx, cres in enumerate(conv_results):
        axs[0, 1].plot(t_axis, cres["r_paths"][0], alpha=0.7, label=f"traj{traj_idx+1}, k=0")
        axs[0, 1].plot(t_axis, cres["r_paths"][-1], alpha=0.7, linestyle="--", label=f"traj{traj_idx+1}, k={cal['N_p']-1}")
    axs[0, 1].set_xlabel("Time t")
    axs[0, 1].set_ylabel("r_t")
    axs[0, 1].set_title("First vs last inner trajectory")
    axs[0, 1].grid(alpha=0.3)
    axs[0, 1].legend(ncol=2, fontsize=8)

    axs[1, 0].plot(loss_hist, color="tab:blue")
    axs[1, 0].set_xlabel("Epoch")
    axs[1, 0].set_ylabel("Mean L")
    axs[1, 0].set_title("Training: mean L per epoch")
    axs[1, 0].grid(alpha=0.3)

    for traj_idx, cres in enumerate(conv_results):
        if cres["delta"].size > 0:
            axs[1, 1].plot(np.arange(1, cal["N_p"]), cres["delta"], "-o", label=f"traj {traj_idx+1}")
    axs[1, 1].set_xlabel("inner iteration k")
    axs[1, 1].set_ylabel("max_t |r^(k)-r^(k-1)|")
    axs[1, 1].set_title("Inner-loop trajectory convergence distance")
    axs[1, 1].grid(alpha=0.3)
    axs[1, 1].legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "forward_looking_results.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    ax.plot(loss_hist)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean L")
    ax.set_title("Forward-looking Huggett (PyTorch)")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "loss_curve.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    for traj_idx, cres in enumerate(conv_results):
        ax.plot(t_axis, cres["r_paths"][0], alpha=0.7, label=f"traj{traj_idx+1}, first")
        ax.plot(t_axis, cres["r_paths"][-1], alpha=0.7, linestyle="--", label=f"traj{traj_idx+1}, last")
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
        axs[0].plot(np.abs(cres["residual_paths"][-1]), label=f"traj {traj_idx+1}")
        axs[1].plot(np.arange(1, cal["N_p"] + 1), cres["boundary_paths"].mean(axis=1), "-o", label=f"traj {traj_idx+1}")
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
    np.save(os.path.join(args.out_dir, "theta_final.npy"), theta.detach().cpu().numpy())
    print("Saved", args.out_dir)


if __name__ == "__main__":
    main()
