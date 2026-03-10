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
    }
    if quick:
        cal.update({"N_epoch_outer": 10, "N_p": 3, "T_traj": 20, "N_sample": 4, "N_z_test": 2})
    else:
        cal.update({"N_epoch_outer": 120, "N_p": 10, "T_traj": 170, "N_sample": 20, "N_z_test": 4})
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


def update_G(theta, G, iz, ir, ip_next, use_same_r, b_grid, y_grid, z_grid, r_grid, Ty, b_min, b_max, c_min):
    nb, ny = b_grid.numel(), y_grid.numel()
    c = theta_to_c(theta, c_min)
    ip_for_c = ir if use_same_r else ip_next
    c_val = c[iz, ir, ip_for_c, :, :].reshape(-1)
    b_next = (1 + r_grid[ir]) * b_grid.repeat_interleave(ny) + y_grid.repeat(nb) * z_grid[iz] - c_val
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


def P_star_best_ir(theta, G, iz, ip_next, use_same_r, b_grid, y_grid, z_grid, r_grid, B, b_min, b_max, c_min):
    nb, ny = b_grid.numel(), y_grid.numel()
    nr = r_grid.numel()
    c = theta_to_c(theta, c_min)
    b_flat = b_grid.repeat_interleave(ny)
    y_flat = y_grid.repeat(nb)
    S_all = []
    for ir in range(nr):
        ip_for_c = ir if use_same_r else ip_next
        c_val = c[iz, ir, ip_for_c, :, :].reshape(-1)
        b_next = (1 + r_grid[ir]) * b_flat + y_flat * z_grid[iz] - c_val
        b_next = torch.clamp(b_next, b_min, b_max).reshape(nb, ny)
        S_all.append((G * b_next).sum())
    S = torch.stack(S_all)
    return torch.argmin(torch.abs(S - B))


def value_to_ir(p_val: float, r_grid_np: np.ndarray) -> int:
    return int(np.clip(np.searchsorted(r_grid_np, p_val, side="right") - 1, 0, len(r_grid_np) - 1))


def make_p_trajectory_indices(r_realized: np.ndarray, r_grid_np: np.ndarray, T_traj: int) -> List[int]:
    idx = [value_to_ir(float(rv), r_grid_np) for rv in r_realized]
    idx.append(idx[-1] if idx else 0)
    return idx


def generate_z_trajectory(T: int, nz: int, Tz_np: np.ndarray) -> np.ndarray:
    iz = np.random.randint(0, nz)
    out = []
    for _ in range(T):
        out.append(iz)
        iz = np.random.choice(nz, p=Tz_np[iz, :])
    return np.asarray(out, dtype=np.int64)


def objective_one_trajectory(theta, z_path, p_path, G0, b_grid, y_grid, z_grid, r_grid, Ty, cal, assume_pt1_equals_pt):
    beta, sigma = cal["beta"], cal["sigma"]
    e_trunc, B = cal["e_trunc"], cal["B"]
    b_min, b_max, c_min = cal["b_min"], cal["b_max"], cal["c_min"]
    T = len(z_path)
    G = G0.clone()
    L = torch.tensor(0.0, device=theta.device, dtype=theta.dtype)
    r_realized = []
    for t in range(T):
        iz = int(z_path[t])
        is_last = (t == T - 1)
        use_same_r = bool(assume_pt1_equals_pt or is_last)
        ip_next = int(p_path[t + 1])
        ir = P_star_best_ir(theta, G.detach(), iz, ip_next, use_same_r, b_grid, y_grid, z_grid, r_grid, B, b_min, b_max, c_min)
        ir = int(ir.item())
        ip_for_c = ir if use_same_r else ip_next
        c_t = theta_to_c(theta, c_min)[iz, ir, ip_for_c, :, :]
        w = (beta ** t) * (1.0 if (beta ** t) >= e_trunc else 0.0)
        L = L + w * (G.detach() * u(c_t, sigma, c_min)).sum()
        G = update_G(theta, G.detach(), iz, ir, ip_next, use_same_r, b_grid, y_grid, z_grid, r_grid, Ty, b_min, b_max, c_min).detach()
        r_realized.append(float(r_grid[ir].item()))
    return L, np.asarray(r_realized)


def run_inner_convergence(theta, z_path, N_p, r_grid_np, G0, b_grid, y_grid, z_grid, r_grid, Ty, cal):
    p_path = [r_grid.numel() // 2] * (len(z_path) + 1)
    r_paths = []
    p_paths = []
    for k in range(N_p):
        assume_pt1 = (k == 0)
        _, r_realized = objective_one_trajectory(theta, z_path, p_path, G0, b_grid, y_grid, z_grid, r_grid, Ty, cal, assume_pt1)
        r_paths.append(r_realized)
        p_paths.append(np.asarray(p_path[:-1], dtype=int))
        p_path = make_p_trajectory_indices(r_realized, r_grid_np, len(z_path))
    r_paths = np.asarray(r_paths)
    p_paths = np.asarray(p_paths)
    if r_paths.shape[0] >= 2:
        delta = np.max(np.abs(r_paths[1:] - r_paths[:-1]), axis=1)
    else:
        delta = np.zeros((0,), dtype=float)
    return r_paths, p_paths, delta


def main():
    p = argparse.ArgumentParser(description="Forward-looking Huggett (PyTorch)")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--out_dir", type=str, default="forward_looking_hugget_output_py")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_sample", type=int, default=None)
    p.add_argument("--n_p", type=int, default=None)
    args = p.parse_args()

    cal = get_calibration(quick=args.quick)
    if args.epochs is not None:
        cal["N_epoch_outer"] = args.epochs
    if args.n_sample is not None:
        cal["N_sample"] = args.n_sample
    if args.n_p is not None:
        cal["N_p"] = args.n_p

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
    G0 = torch.ones(nb_spg, cal["ny"], dtype=torch.float32, device=device) / (nb_spg * cal["ny"])

    theta = init_theta(b_grid, y_grid, z_grid, r_grid, save_frac=0.2, c_min=cal["c_min"]).to(device)
    theta = torch.nn.Parameter(theta)
    opt = torch.optim.Adam([theta], lr=cal["lr_ini"])

    r_grid_np = np.asarray(r_grid.detach().cpu())
    loss_hist = []

    print(
        f"Forward-Looking Huggett PY: device={device}, N_sample={cal['N_sample']}, "
        f"N_p={cal['N_p']}, N_epoch={cal['N_epoch_outer']}, T={cal['T_traj']}"
    )
    t0 = time.perf_counter()
    for ep in range(cal["N_epoch_outer"]):
        ep_losses = []
        for _n in range(cal["N_sample"]):
            # One z trajectory per sample; reused by all inner k.
            z_path = generate_z_trajectory(cal["T_traj"], nz_spg, Tz_sub)
            p_path = [nr_spg // 2] * (cal["T_traj"] + 1)
            for k in range(cal["N_p"]):
                assume_pt1 = (k == 0)
                opt.zero_grad(set_to_none=True)
                L, r_realized = objective_one_trajectory(theta, z_path, p_path, G0, b_grid, y_grid, z_grid, r_grid, Ty, cal, assume_pt1)
                (-L).backward()
                opt.step()
                ep_losses.append(float(L.detach().cpu()))
                p_path = make_p_trajectory_indices(r_realized, r_grid_np, cal["T_traj"])
        loss_hist.append(float(np.mean(ep_losses)))
        if ep == 0 or (ep + 1) % 10 == 0:
            print(f"Epoch {ep+1}, mean L = {loss_hist[-1]:.4f}")
    print(f"Training done in {time.perf_counter()-t0:.2f}s")

    os.makedirs(args.out_dir, exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    ax.plot(loss_hist)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean L")
    ax.set_title("Forward-looking Huggett (PyTorch)")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "loss_curve.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Post-training trajectory convergence visualization
    test_z = [generate_z_trajectory(cal["T_traj"], nz_spg, Tz_sub) for _ in range(cal["N_z_test"])]
    for i, z_path in enumerate(test_z):
        r_paths, p_paths, delta = run_inner_convergence(theta, z_path, cal["N_p"], r_grid_np, G0, b_grid, y_grid, z_grid, r_grid, Ty, cal)
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
        np.save(os.path.join(args.out_dir, f"conv_p_paths_traj{i+1:02d}.npy"), p_paths)
        np.save(os.path.join(args.out_dir, f"conv_delta_traj{i+1:02d}.npy"), delta)
    with open(os.path.join(args.out_dir, "loss_hist.txt"), "w") as f:
        f.write("\n".join(map(str, loss_hist)))
    np.save(os.path.join(args.out_dir, "theta_final.npy"), theta.detach().cpu().numpy())
    print("Saved", args.out_dir)


if __name__ == "__main__":
    main()
