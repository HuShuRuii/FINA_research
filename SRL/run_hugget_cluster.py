#!/usr/bin/env python3
"""
Huggett (1993) SRL/SPG — cluster-run script.
Runs calibration, SPG training, then saves visualizations to an output directory.

Dependencies: numpy, matplotlib, scipy, torch (install on cluster as needed).

Usage:
  python run_hugget_cluster.py [--out_dir OUTPUT_DIR] [--seed SEED] [--epochs N] [--quick]
"""
import argparse
import os
import sys
import time
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless for cluster
import matplotlib.pyplot as plt
from scipy.stats import norm

import torch

# ---------- Logger ----------
import builtins
_original_print = builtins.print

def log(msg):
    """Print with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _original_print(f"[{timestamp}] {msg}")
    sys.stdout.flush()

# Override print to always flush
def _print(*args, **kwargs):
    kwargs['flush'] = True
    _original_print(*args, **kwargs)
builtins.print = _print

# ---------- Parse args ----------
def parse_args():
    p = argparse.ArgumentParser(description="Run Huggett SRL on cluster, save figures.")
    p.add_argument("--out_dir", type=str, default="hugget_output", help="Directory for figures and logs")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--epochs", type=int, default=None, help="Max SPG epochs (default: from calibration)")
    p.add_argument("--quick", action="store_true", help="Short run: 20 epochs, small sample")
    return p.parse_args()

# ---------- Calibration (SRL Table 2 & 3) ----------
def get_calibration(quick=False):
    beta, sigma = 0.96, 2.0
    rho_y, nu_y = 0.6, 0.2
    rho_z, nu_z = 0.9, 0.02
    B, b_min = 0.0, -1.0
    nb, b_max = 40, 50.0
    ny, nr, nz = 3, 20, 30
    r_min, r_max = 0.01, 0.06
    c_min = 1e-3
    T_trunc = 170
    e_trunc = 1e-3  # stop trajectory when beta**t < e_trunc (remaining utility negligible)
    N_epoch, N_warmup = (20, 5) if quick else (400, 50)
    lr_ini, lr_decay = 1e-3, 0.5
    N_sample, e_converge = (64, 1e-3) if quick else (512, 3e-4)
    return {
        "beta": beta, "sigma": sigma, "rho_y": rho_y, "nu_y": nu_y,
        "rho_z": rho_z, "nu_z": nu_z, "B": B, "b_min": b_min,
        "nb": nb, "b_max": b_max, "ny": ny, "nr": nr, "nz": nz,
        "r_min": r_min, "r_max": r_max, "c_min": c_min, "T_trunc": T_trunc, "e_trunc": e_trunc,
        "N_epoch": N_epoch, "N_warmup": N_warmup, "lr_ini": lr_ini, "lr_decay": lr_decay,
        "N_sample": N_sample, "e_converge": e_converge,
    }

# ---------- Tauchen ----------
def tauchen_ar1(rho, sigma_innov, n_states, m=3, mean=0.0):
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

# ---------- Policy from grid: b continuous (lottery); iy/iz/ir are integer indices ----------
def policy_from_grid(b, iy, iz, ir, c_grid, b_grid, y_grid, z_grid, r_grid, c_min_val=1e-3):
    """b: continuous (lottery on b_grid); iy, iz, ir: integer indices. Returns (c, b_next)."""
    b = np.atleast_1d(np.asarray(b, dtype=float))
    nb = len(b_grid)
    b_c = np.clip(b, b_grid[0], b_grid[-1])
    j_hi = np.clip(np.searchsorted(b_grid, b_c), 1, nb - 1)
    j_lo = j_hi - 1
    w = (b_c - b_grid[j_lo]) / np.maximum(b_grid[j_hi] - b_grid[j_lo], 1e-20)
    c = (1 - w) * c_grid[j_lo, iy, iz, ir] + w * c_grid[j_hi, iy, iz, ir]
    c = np.maximum(c, c_min_val)
    c_total = (1 + r_grid[ir]) * b + y_grid[iy] * z_grid[iz]
    b_next = np.clip(c_total - c, b_grid[0], b_grid[-1])
    c = np.maximum(c_total - b_next, c_min_val)
    if c.size == 1:
        return float(c.ravel()[0]), float(b_next.ravel()[0])
    return c, b_next

# ---------- Main ----------
def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    log(f"=== Starting Huggett Model Training ===")
    log(f"Output directory: {args.out_dir}")
    log(f"Epochs: {args.epochs if args.epochs else 'default'}")
    log(f"Quick mode: {args.quick}")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    log(f"Setting up calibration...")
    cal = get_calibration(quick=args.quick)
    if args.epochs is not None:
        cal["N_epoch"] = args.epochs

    beta = cal["beta"]
    sigma = cal["sigma"]
    c_min = cal["c_min"]
    b_min = cal["b_min"]
    b_max = cal["b_max"]
    ny = cal["ny"]
    nb, nr, nz = cal["nb"], cal["nr"], cal["nz"]
    r_min, r_max = cal["r_min"], cal["r_max"]

    # Grids (numpy)
    b_grid = np.linspace(b_min, b_max, nb)
    r_grid = np.linspace(r_min, r_max, nr)
    y_grid, Ty = tauchen_ar1(cal["rho_y"], cal["nu_y"], ny, m=3, mean=1.0)
    invariant_y = np.linalg.matrix_power(Ty.T, 200)[:, 0]
    y_grid = y_grid / (y_grid @ invariant_y)
    log_z_grid, Tz = tauchen_ar1(cal["rho_z"], cal["nu_z"], nz)
    z_grid = np.exp(log_z_grid)
    invariant_z = np.linalg.matrix_power(Tz.T, 200)[:, 0]
    z_grid = z_grid / (z_grid @ invariant_z)

    # Device & SPG grids
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    nb_spg, nr_spg, nz_spg = 50, 10, 10
    b_grid_spg = torch.tensor(np.linspace(b_min, b_max, nb_spg), dtype=dtype, device=device)
    iz_spg = np.linspace(0, nz - 1, nz_spg, dtype=int)
    z_grid_spg_np = z_grid[iz_spg]
    z_grid_t = torch.tensor(z_grid_spg_np, dtype=dtype, device=device)
    r_grid_t = torch.tensor(np.linspace(r_min, r_max, nr_spg), dtype=dtype, device=device)
    y_grid_t = torch.tensor(y_grid, dtype=dtype, device=device)
    Ty_t = torch.tensor(Ty, dtype=dtype, device=device)
    Tz_sub = Tz[np.ix_(iz_spg, iz_spg)]
    Tz_sub = Tz_sub / Tz_sub.sum(axis=1, keepdims=True)
    Tz_t = torch.tensor(Tz_sub, dtype=dtype, device=device)
    nz_spg = Tz_t.shape[0]
    nr_spg = len(r_grid_t)

    def theta_to_consumption_grid(theta, *_, c_min_val=1e-3):
        return torch.clamp(theta, min=c_min_val)

    def init_theta(b_grid_t, y_grid_t, z_grid_t, r_grid_t, save_frac=0.2, c_min_val=1e-3):
        nb_t, ny_t = len(b_grid_t), len(y_grid_t)
        J, nz_t, nr_t = nb_t * ny_t, len(z_grid_t), len(r_grid_t)
        b_flat = b_grid_t.repeat_interleave(ny_t)
        y_flat = y_grid_t.repeat(nb_t)
        cash = b_flat.view(J, 1, 1) * (1 + r_grid_t).view(1, 1, nr_t) + y_flat.view(J, 1, 1) * z_grid_t.view(1, nz_t, 1)
        c_grid = torch.clamp((1 - save_frac) * cash, min=c_min_val)
        return c_grid.view(nb_t, ny_t, nz_t, nr_t)

    def _G_to_mat_spg(G, nb_spg, ny):
        return G.view(nb_spg, ny) if G.dim() == 1 else G

    def update_G_pi_direct(theta, G, iz, ir, b_grid_t, y_grid_t, z_grid_t, r_grid_t, Ty_t, nb_spg, ny):
        G = _G_to_mat_spg(G, nb_spg, ny)
        c = theta_to_consumption_grid(theta, b_grid_t, y_grid_t, z_grid_t, r_grid_t)
        c_val = c[:, :, iz, ir].ravel()
        b_next = (1 + r_grid_t[ir]) * b_grid_t.repeat_interleave(ny) + y_grid_t.repeat(nb_spg) * z_grid_t[iz] - c_val
        b_next = torch.clamp(b_next, b_grid_t[0], b_grid_t[-1])
        idx_hi = torch.searchsorted(b_grid_t, b_next).clamp(1, nb_spg - 1)
        idx_lo = idx_hi - 1
        w_hi = (b_next - b_grid_t[idx_lo]) / (b_grid_t[idx_hi] - b_grid_t[idx_lo]).clamp(min=1e-20)
        w_lo = 1.0 - w_hi
        eye = torch.eye(nb_spg, device=b_grid_t.device, dtype=b_grid_t.dtype)
        w_b = w_lo.unsqueeze(1) * eye[idx_lo] + w_hi.unsqueeze(1) * eye[idx_hi]
        M = w_b.view(nb_spg, ny, nb_spg).permute(2, 0, 1)
        Q = (M * G.unsqueeze(0)).sum(dim=1)
        G_new = Q @ Ty_t
        G_new = G_new / (G_new.sum() + 1e-20)
        return G_new

    def P_star_detach(theta, G, iz, b_grid_t, y_grid_t, z_grid_t, r_grid_t, ny, B=0.0):
        nr = len(r_grid_t)
        if nr == 1:
            return r_grid_t[0].detach()
        nb_b = len(b_grid_t)
        G_mat = _G_to_mat_spg(G, nb_b, ny)
        z_val = z_grid_t[iz]
        c_all = theta_to_consumption_grid(theta, b_grid_t, y_grid_t, z_grid_t, r_grid_t)[:, :, iz, :].reshape(nb_b * ny, nr)
        b_flat = b_grid_t.repeat_interleave(ny)
        y_flat = y_grid_t.repeat(nb_b)
        resources = b_flat.unsqueeze(1) * (1 + r_grid_t).unsqueeze(0) + (y_flat * z_val).unsqueeze(1)
        b_next_all = (resources - c_all).clamp(b_min, b_max)
        b_next_all = b_next_all.view(nb_b, ny, nr)
        S_all = (G_mat.unsqueeze(2) * b_next_all).sum(dim=(0, 1))
        best_ir = (S_all - B).abs().argmin().item()
        return r_grid_t[best_ir].detach()

    def r_to_ir(r_val, r_grid_t):
        r = r_val.item() if torch.is_tensor(r_val) else float(r_val)
        grid = r_grid_t.cpu().numpy()
        idx = np.searchsorted(grid, r, side='right') - 1
        return int(np.clip(idx, 0, len(grid) - 1))

    def u_torch(c_vec, sig=sigma):
        c_vec = torch.clamp(c_vec, min=c_min)
        if abs(sig - 1.0) < 1e-8:
            return torch.log(c_vec)
        return (c_vec ** (1 - sig)) / (1 - sig)

    def steady_state_G0(theta, b_grid_t, y_grid_t, z_grid_t, r_grid_t, Ty_t, nb_spg, ny, nz_spg, nr_spg, n_iter=150, tol=1e-6):
        iz_mid = nz_spg // 2
        ir_mid = nr_spg // 2 if nr_spg > 0 else 0
        with torch.no_grad():
            G = torch.ones(nb_spg, ny, device=device, dtype=dtype) / (nb_spg * ny)
            for _ in range(n_iter):
                G_new = update_G_pi_direct(theta, G, iz_mid, ir_mid, b_grid_t, y_grid_t, z_grid_t, r_grid_t, Ty_t, nb_spg, ny)
                if (G_new - G).abs().max() < tol:
                    return G_new
                G = G_new
        return G

    def spg_objective(theta, N_traj, T_horizon, b_grid_t, y_grid_t, z_grid_t, r_grid_t, Ty_t, Tz_t,
                     nb_spg, ny, nz_spg, nr_spg, beta_t, G0=None, warm_up=False):
        if G0 is None:
            G0 = torch.ones(nb_spg, ny, device=device, dtype=dtype) / (nb_spg * ny)
        else:
            G0 = _G_to_mat_spg(G0, nb_spg, ny)
        L_list = []
        e_trunc = cal.get("e_trunc", 1e-3)
        for n in range(N_traj):
            iz = np.random.randint(0, nz_spg)
            G = G0.clone()
            L_n = torch.tensor(0.0, device=device, dtype=dtype)
            for t in range(T_horizon):
                if (beta_t ** t) < e_trunc:
                    break
                r_t = P_star_detach(theta, G, iz, b_grid_t, y_grid_t, z_grid_t, r_grid_t, ny)
                ir = r_to_ir(r_t, r_grid_t)
                c = theta_to_consumption_grid(theta, b_grid_t, y_grid_t, z_grid_t, r_grid_t)
                c_t = c[:, :, iz, ir]
                L_n = L_n + (beta_t ** t) * (G * u_torch(c_t)).sum()
                if not warm_up:
                    G = update_G_pi_direct(theta, G, iz, ir, b_grid_t, y_grid_t, z_grid_t, r_grid_t, Ty_t, nb_spg, ny).detach()
                iz = torch.multinomial(Tz_t[iz, :], 1).squeeze().item()
            L_list.append(L_n)
        return torch.stack(L_list).mean()

    def save_visualizations(out_dir, theta_cur, loss_cur, suffix=""):
        """Save loss curve + consumption policy + c vs r to out_dir. suffix e.g. '_ep050' for checkpoints."""
        c_grid_np = theta_to_consumption_grid(theta_cur.detach(), c_min_val=c_min).cpu().numpy()
        if c_grid_np.ndim == 3:
            c_grid_np = c_grid_np.reshape(nb_spg, ny, nz_spg, nr_spg)
        b_grid_np = b_grid_spg.cpu().numpy()
        y_grid_np = y_grid_t.cpu().numpy()
        z_grid_np = z_grid_t.cpu().numpy()
        r_grid_np = r_grid_t.cpu().numpy()

        def policy_cur(b, iy, iz, ir):
            return policy_from_grid(b, iy, iz, ir, c_grid_np, b_grid_np, y_grid_np, z_grid_np, r_grid_np,
                                   c_min_val=c_min)

        plt.rcParams["font.size"] = 10
        plt.rcParams["axes.unicode_minus"] = False

        fig, ax = plt.subplots(1, 1, figsize=(7, 4))
        ax.plot(loss_cur, color="tab:blue")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("L(θ)")
        ax.set_title("SPG training loss")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, f"loss_curve{suffix}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

        fig, axs = plt.subplots(3, 3, figsize=(18, 9))
        ir_indices = [0, nr_spg // 2, nr_spg - 1] if nr_spg >= 3 else list(range(nr_spg))
        iz_indices = [0, len(z_grid_np) // 2, len(z_grid_np) - 1]
        for row, ir_v in enumerate(ir_indices[:3]):
            for col, iz_v in enumerate(iz_indices):
                ax = axs[row, col]
                iy_v = 1
                b_lin = np.linspace(b_min, b_max, 100)
                c_ge, _ = policy_cur(b_lin, iy_v, iz_v, ir_v)
                label = f"y={y_grid_np[iy_v]:.3f}, r={r_grid_np[ir_v]:.3f}, z={z_grid_np[iz_v]:.2f}"
                ax.plot(b_lin, c_ge, label=label)
                ax.set_xlabel("Bond holdings b")
                ax.set_ylabel("Consumption c_GE(b,y,r,z)")
                ax.grid(alpha=0.3)
                ax.legend()
        fig.suptitle("SRL/GE: Consumption policy c_GE(b, y, r, z)")
        plt.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(os.path.join(out_dir, f"consumption_policy_grid{suffix}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

        iz_mid = len(z_grid_np) // 2
        iy_s, ib_s = 1, nb_spg // 2
        b_val = float(b_grid_np[ib_s])
        c_r_curve = [policy_cur(b_val, iy_s, iz_mid, ir_v)[0] for ir_v in range(nr_spg)]
        fig, ax = plt.subplots(1, 1, figsize=(7, 4))
        ax.plot(r_grid_np, c_r_curve, "-o")
        ax.set_xlabel("r")
        ax.set_ylabel("c(b,y,r,z)")
        ax.set_title(f"Consumption vs r (b={b_val:.2f}, y={y_grid_np[iy_s]:.2f}, z={z_grid_np[iz_mid]:.2f})")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, f"consumption_vs_r{suffix}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    # ---------- Train SPG ----------
    log(f"Initializing SPG training...")
    log(f"Device: {device}")
    log(f"Grid sizes: b={nb_spg}, y={ny}, z={nz_spg}, r={nr_spg}")
    log(f"Exporting visualizations every 50 epochs to {args.out_dir}/")

    theta = init_theta(b_grid_spg, y_grid_t, z_grid_t, r_grid_t)
    theta = theta.requires_grad_(True)
    optimizer = torch.optim.Adam([theta], lr=cal["lr_ini"])
    log(f"Computing initial steady state G0...")
    G0_steady = steady_state_G0(theta, b_grid_spg, y_grid_t, z_grid_t, r_grid_t, Ty_t, nb_spg, ny, nz_spg, nr_spg)
    T_horizon = min(cal["T_trunc"], 50)
    loss_hist = []
    log(f"Starting training loop ({cal['N_epoch']} epochs)...")

    start_time = time.time()
    for epoch in range(cal["N_epoch"]):
        warm_up = epoch < cal["N_warmup"]
        t0 = max(epoch - cal["N_warmup"], 0) / max(cal["N_epoch"] - cal["N_warmup"], 1)
        lr_t = cal["lr_ini"] * (cal["lr_decay"] ** t0)
        for g in optimizer.param_groups:
            g["lr"] = lr_t
        theta_old = theta.detach().clone()
        optimizer.zero_grad()
        G0_phase = G0_steady.detach() if warm_up else None
        L = spg_objective(theta, cal["N_sample"], T_horizon, b_grid_spg, y_grid_t, z_grid_t, r_grid_t, Ty_t, Tz_t,
                         nb_spg, ny, nz_spg, nr_spg, beta, G0=G0_phase, warm_up=warm_up)
        loss_hist.append(L.item())
        (-L).backward()
        optimizer.step()
        param_change = (theta.detach() - theta_old).abs().max().item()

        # Export visualizations every 50 epochs (HPC4: you can download from out_dir)
        if (epoch + 1) % 50 == 0:
            suffix = f"_ep{epoch+1:04d}"
            log(f"Exporting checkpoint at epoch {epoch+1} -> {args.out_dir}/loss_curve{suffix}.png etc.")
            save_visualizations(args.out_dir, theta, loss_hist, suffix=suffix)

        if param_change < cal["e_converge"]:
            break
        if (epoch + 1) % 10 == 0 or (epoch + 1) <= 5 or (epoch + 1) == cal["N_warmup"]:
            phase = "warm-up (G fixed)" if warm_up else "G evolves"
            elapsed = time.time() - start_time
            log(f"Epoch {epoch+1}/{cal['N_epoch']}, L(θ) = {L.item():.6f}, lr = {lr_t:.2e}, |Δθ| = {param_change:.2e}, {phase}, elapsed={elapsed:.1f}s")

    # ---------- Final export (no suffix = latest) ----------
    log(f"Exporting final visualizations -> {args.out_dir}/loss_curve.png etc.")
    save_visualizations(args.out_dir, theta, loss_hist, suffix="")

    # Save loss history as text
    with open(os.path.join(args.out_dir, "loss_hist.txt"), "w") as f:
        f.write("\n".join(map(str, loss_hist)))

    log(f"Done! Figures and loss_hist.txt saved to {args.out_dir}/")
    total_time = time.time() - start_time
    log(f"Total training time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")

if __name__ == "__main__":
    main()
