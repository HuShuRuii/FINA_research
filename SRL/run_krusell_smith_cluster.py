#!/usr/bin/env python3
"""
Krusell–Smith (1998) SRL/SPG — cluster-run script.
Runs calibration, SPG training, then saves visualizations to an output directory.

Dependencies: numpy, matplotlib, scipy, torch (install on cluster as needed).

Usage:
  python run_krusell_smith_cluster.py [--out_dir OUTPUT_DIR] [--seed SEED] [--epochs N] [--quick]
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

def _print(*args, **kwargs):
    kwargs['flush'] = True
    _original_print(*args, **kwargs)
builtins.print = _print

# ---------- Parse args ----------
def parse_args():
    p = argparse.ArgumentParser(description="Run Krusell-Smith SRL on cluster, save figures.")
    p.add_argument("--out_dir", type=str, default="krusell_smith_output", help="Directory for figures and logs")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--epochs", type=int, default=None, help="Max SPG epochs (default: from calibration)")
    p.add_argument("--quick", action="store_true", help="Short run: 20 epochs, small sample")
    return p.parse_args()

# ---------- Calibration (SRL 4.2, App A.2 Table 4 & 5) ----------
def get_calibration(quick=False):
    beta, sigma = 0.95, 3.0
    alpha, delta = 0.36, 0.08
    rho_y, nu_y = 0.6, 0.2
    rho_z, nu_z = 0.9, 0.03
    ny = 3
    b_min, b_max = 0.0, 100.0
    nb = 200
    nr, nw, nz = 30, 50, 30
    r_min, r_max = 0.02, 0.07
    w_min, w_max = 0.9, 1.5
    c_min = 1e-3
    T_trunc = 90
    N_epoch, N_warmup = (20, 5) if quick else (200, 25)
    lr_ini, lr_decay = 5e-4, 0.5
    N_sample, e_converge = (16, 1e-3) if quick else (32, 3e-4)
    return {
        "beta": beta, "sigma": sigma, "alpha": alpha, "delta": delta,
        "rho_y": rho_y, "nu_y": nu_y, "rho_z": rho_z, "nu_z": nu_z,
        "ny": ny, "nb": nb, "b_min": b_min, "b_max": b_max,
        "nr": nr, "nw": nw, "nz": nz,
        "r_min": r_min, "r_max": r_max, "w_min": w_min, "w_max": w_max,
        "c_min": c_min, "T_trunc": T_trunc,
        "N_epoch": N_epoch, "N_warmup": N_warmup, "lr_ini": lr_ini, "lr_decay": lr_decay,
        "N_sample": N_sample, "e_converge": e_converge,
    }

# ---------- K_to_prices ----------
def K_to_prices(K, z, alpha, delta):
    """(K, z) -> (r_net, w). L=1 fixed."""
    K = np.maximum(K, 1e-8)
    rK = alpha * z * (K ** (alpha - 1))
    w = (1 - alpha) * z * (K ** alpha)
    r_net = rK - delta
    return r_net, w

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

# ---------- Policy from grid (K-S: c grid J,nz,nr,nw) ----------
def policy_from_grid_ks(b, y, r, w, z, theta_grid, b_grid, y_grid, z_grid, r_grid, w_grid, ny,
                        c_min_val=1e-3, n_labour=1.0, b_min=None, b_max=None):
    b_min = b_min if b_min is not None else b_grid[0]
    b_max = b_max if b_max is not None else b_grid[-1]
    ib = np.atleast_1d(np.clip(np.searchsorted(b_grid, b, side="right") - 1, 0, len(b_grid) - 1))
    iy = np.atleast_1d(np.clip(np.searchsorted(y_grid, y, side="right") - 1, 0, len(y_grid) - 1))
    iz = np.atleast_1d(np.clip(np.searchsorted(z_grid, z, side="right") - 1, 0, len(z_grid) - 1))
    ir = np.atleast_1d(np.clip(np.searchsorted(r_grid, r, side="right") - 1, 0, len(r_grid) - 1))
    iw = np.atleast_1d(np.clip(np.searchsorted(w_grid, w, side="right") - 1, 0, len(w_grid) - 1))
    b = np.atleast_1d(np.asarray(b, dtype=float))
    y = np.atleast_1d(np.asarray(y, dtype=float))
    r = np.atleast_1d(np.asarray(r, dtype=float))
    w = np.atleast_1d(np.asarray(w, dtype=float))
    z = np.atleast_1d(np.asarray(z, dtype=float))
    j = ib * ny + iy
    c = np.maximum(theta_grid[j, iz, ir, iw], c_min_val)
    cash = (1 + r) * b + w * n_labour * y
    b_next = np.clip(cash - c, b_min, b_max)
    c = np.maximum(cash - b_next, c_min_val)
    n = np.full_like(b, n_labour)
    if c.size == 1:
        return c.ravel()[0], n.ravel()[0], b_next.ravel()[0]
    return c, n, b_next

# ---------- Main ----------
def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    log(f"=== Starting Krusell-Smith Model Training ===")
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
    alpha = cal["alpha"]
    delta = cal["delta"]
    c_min = cal["c_min"]
    b_min = cal["b_min"]
    b_max = cal["b_max"]
    ny = cal["ny"]
    nb, nr, nw, nz = cal["nb"], cal["nr"], cal["nw"], cal["nz"]
    r_min, r_max = cal["r_min"], cal["r_max"]
    w_min, w_max = cal["w_min"], cal["w_max"]

    # Grids (numpy)
    b_grid = np.linspace(b_min, b_max, nb)
    r_grid = np.linspace(r_min, r_max, nr)
    w_grid = np.linspace(w_min, w_max, nw)
    y_grid, Ty = tauchen_ar1(cal["rho_y"], cal["nu_y"], ny, m=3, mean=1.0)
    invariant_y = np.linalg.matrix_power(Ty.T, 200)[:, 0]
    y_grid = y_grid / (y_grid @ invariant_y)
    log_z_grid, Tz = tauchen_ar1(cal["rho_z"], cal["nu_z"], nz)
    z_grid = np.exp(log_z_grid)
    invariant_z = np.linalg.matrix_power(Tz.T, 200)[:, 0]
    z_grid = z_grid / (z_grid @ invariant_z)

    # K_to_prices (numpy)
    def K_to_prices_np(K, z):
        return K_to_prices(K, z, alpha, delta)

    # Device & SPG grids
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    nb_spg, nr_spg, nw_spg, nz_spg = 50, 15, 25, 10
    ny_spg = ny
    J = nb_spg * ny_spg

    b_grid_t = torch.tensor(np.linspace(b_min, b_max, nb_spg), dtype=dtype, device=device)
    r_grid_t = torch.tensor(np.linspace(r_min, r_max, nr_spg), dtype=dtype, device=device)
    w_grid_t = torch.tensor(np.linspace(w_min, w_max, nw_spg), dtype=dtype, device=device)
    iz_spg = np.linspace(0, nz - 1, nz_spg, dtype=int)
    z_grid_t = torch.tensor(z_grid[iz_spg], dtype=dtype, device=device)
    y_grid_t = torch.tensor(y_grid, dtype=dtype, device=device)
    Ty_t = torch.tensor(Ty, dtype=dtype, device=device)
    Tz_sub = Tz[np.ix_(iz_spg, iz_spg)]
    Tz_sub = Tz_sub / Tz_sub.sum(axis=1, keepdims=True)
    Tz_t = torch.tensor(Tz_sub, dtype=dtype, device=device)
    nz_spg = Tz_t.shape[0]

    def theta_to_c_grid(theta, *_, c_min_val=1e-3):
        return torch.nn.functional.softplus(theta) + c_min_val

    def init_theta_ks(b_grid_t, y_grid_t, z_grid_t, r_grid_t, w_grid_t, save_frac=0.25):
        nz, nr, nw = len(z_grid_t), len(r_grid_t), len(w_grid_t)
        c_grid = torch.zeros(J, nz, nr, nw, dtype=dtype, device=device)
        for ib in range(nb_spg):
            for iy in range(ny_spg):
                j = ib * ny_spg + iy
                b = b_grid_t[ib].item()
                y = y_grid_t[iy].item()
                for iz in range(nz):
                    z = z_grid_t[iz].item()
                    for ir in range(nr):
                        r = r_grid_t[ir].item()
                        for iw in range(nw):
                            w = w_grid_t[iw].item()
                            cash = (1 + r) * b + w * y
                            c_grid[j, iz, ir, iw] = max((1 - save_frac) * cash, c_min)
        x = c_grid - c_min
        return torch.log(torch.clamp(torch.exp(torch.clamp(x, max=20)) - 1, min=1e-8))

    def K_from_d(d, b_grid_t, ny_spg):
        b_vals = b_grid_t.repeat_interleave(ny_spg)
        return (d * b_vals).sum()

    def P_star_detach_ks(theta, d, iz, b_grid_t, y_grid_t, z_grid_t, r_grid_t, w_grid_t, ny_spg):
        K = K_from_d(d, b_grid_t, ny_spg).item()
        z_val = z_grid_t[iz].item()
        r_net, w_val = K_to_prices_np(K, z_val)
        r_t = torch.tensor(r_net, device=device, dtype=dtype)
        w_t = torch.tensor(w_val, device=device, dtype=dtype)
        return r_t.detach(), w_t.detach()

    # Map scalar r,w to grid indices (same idea as Huggett r_to_ir: single value -> NumPy searchsorted).
    def rw_to_ir_iw(r_val, w_val, r_grid_t, w_grid_t):
        r = r_val.item() if torch.is_tensor(r_val) else float(r_val)
        w = w_val.item() if torch.is_tensor(w_val) else float(w_val)
        rn, wn = r_grid_t.cpu().numpy(), w_grid_t.cpu().numpy()
        ir = int(np.clip(np.searchsorted(rn, r), 0, len(rn) - 1))
        iw = int(np.clip(np.searchsorted(wn, w), 0, len(wn) - 1))
        return ir, iw

    def update_d_pi_direct_ks(theta, d, iz, ir, iw, b_grid_t, y_grid_t, z_grid_t, r_grid_t, w_grid_t,
                              Ty_t, nb_spg, ny_spg, sigma_b=0.5):
        d_mat = d.view(nb_spg, ny_spg) if d.dim() == 1 else d
        z_val = z_grid_t[iz]
        r_val = r_grid_t[ir]
        w_val = w_grid_t[iw]
        c = theta_to_c_grid(theta)[:, iz, ir, iw]
        b_next = (1 + r_val) * b_grid_t.repeat_interleave(ny_spg) + y_grid_t.repeat(nb_spg) * w_val - c
        b_next = torch.clamp(b_next, b_min, b_max)
        dist = b_next.unsqueeze(1) - b_grid_t.unsqueeze(0)
        w_b = torch.exp(-dist.pow(2) / (2 * sigma_b**2))
        w_b = w_b / (w_b.sum(dim=1, keepdim=True) + 1e-8)
        M = w_b.view(nb_spg, ny_spg, nb_spg).permute(2, 0, 1)
        Q = (M * d_mat.unsqueeze(0)).sum(dim=1)
        d_new = (Q @ Ty_t).reshape(-1)
        return d_new / (d_new.sum() + 1e-20)

    def u_torch(c_vec, sig=sigma):
        c_vec = torch.clamp(c_vec, min=c_min)
        if abs(sig - 1.0) < 1e-8:
            return torch.log(c_vec)
        return (c_vec ** (1 - sig)) / (1 - sig)

    def steady_state_d0_ks(theta, b_grid_t, y_grid_t, z_grid_t, r_grid_t, w_grid_t, Ty_t,
                           nb_spg, ny_spg, nz_spg, nr_spg, nw_spg, n_iter=150):
        iz_mid, ir_mid, iw_mid = nz_spg // 2, nr_spg // 2, nw_spg // 2
        with torch.no_grad():
            d = torch.ones(J, device=device, dtype=dtype) / J
            for _ in range(n_iter):
                d = update_d_pi_direct_ks(theta, d, iz_mid, ir_mid, iw_mid, b_grid_t, y_grid_t, z_grid_t,
                                          r_grid_t, w_grid_t, Ty_t, nb_spg, ny_spg)
        return d

    def spg_objective_ks(theta, N_traj, T_horizon, b_grid_t, y_grid_t, z_grid_t, r_grid_t, w_grid_t,
                         Ty_t, Tz_t, nb_spg, ny_spg, nz_spg, nr_spg, nw_spg, beta_t, d0=None, warm_up=False):
        if d0 is None:
            d0 = torch.ones(J, device=device, dtype=dtype) / J
        L_list = []
        for n in range(N_traj):
            iz = np.random.randint(0, nz_spg)
            d = d0.clone()
            L_n = torch.tensor(0.0, device=device, dtype=dtype)
            for t in range(T_horizon):
                r_t, w_t = P_star_detach_ks(theta, d, iz, b_grid_t, y_grid_t, z_grid_t, r_grid_t, w_grid_t, ny_spg)
                ir, iw = rw_to_ir_iw(r_t.item(), w_t.item(), r_grid_t, w_grid_t)
                c = theta_to_c_grid(theta)
                c_t = c[:, iz, ir, iw]
                L_n = L_n + (beta_t ** t) * (d @ u_torch(c_t))
                if not warm_up:
                    d = update_d_pi_direct_ks(theta, d, iz, ir, iw, b_grid_t, y_grid_t, z_grid_t,
                                              r_grid_t, w_grid_t, Ty_t, nb_spg, ny_spg).detach()
                iz = torch.multinomial(Tz_t[iz, :], 1).squeeze().item()
            L_list.append(L_n)
        return torch.stack(L_list).mean()

    def save_visualizations(out_dir, theta_cur, loss_cur, suffix=""):
        c_grid_np = theta_to_c_grid(theta_cur.detach(), c_min_val=c_min).cpu().numpy()
        b_grid_np = b_grid_t.cpu().numpy()
        y_grid_np = y_grid_t.cpu().numpy()
        z_grid_np = z_grid_t.cpu().numpy()
        r_grid_np = r_grid_t.cpu().numpy()
        w_grid_np = w_grid_t.cpu().numpy()

        def policy_cur(b, y, r, w, z):
            return policy_from_grid_ks(b, y, r, w, z, c_grid_np, b_grid_np, y_grid_np, z_grid_np,
                                      r_grid_np, w_grid_np, ny_spg, c_min_val=c_min,
                                      b_min=b_min, b_max=b_max)

        plt.rcParams["font.size"] = 10
        plt.rcParams["axes.unicode_minus"] = False

        fig, ax = plt.subplots(1, 1, figsize=(7, 4))
        ax.plot(loss_cur, color="tab:blue")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("L(θ)")
        ax.set_title("Krusell-Smith SPG training loss")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, f"loss_curve{suffix}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

        z0 = z_grid_np[len(z_grid_np)//2]
        w0 = w_grid_np[len(w_grid_np)//2]
        iy_s, ib_s = 1, nb_spg // 2
        b_val, y_val = float(b_grid_np[ib_s]), float(y_grid_np[iy_s])
        c_r_curve = [policy_cur(b_val, y_val, r_val, w0, z0)[0] for r_val in r_grid_np]
        fig, ax = plt.subplots(1, 1, figsize=(7, 4))
        ax.plot(r_grid_np, c_r_curve, "-o")
        ax.set_xlabel("r")
        ax.set_ylabel("c(b,y,r,w,z)")
        ax.set_title(f"Consumption vs r (b={b_val:.2f}, y={y_val:.2f}, w={w0:.2f}, z={z0:.2f})")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, f"consumption_vs_r{suffix}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    # ---------- Train SPG ----------
    log(f"Initializing SPG training...")
    log(f"Device: {device}")
    log(f"Grid sizes: b={nb_spg}, y={ny_spg}, z={nz_spg}, r={nr_spg}, w={nw_spg}")
    log(f"Exporting visualizations every 50 epochs to {args.out_dir}/")

    theta = init_theta_ks(b_grid_t, y_grid_t, z_grid_t, r_grid_t, w_grid_t)
    theta = theta.requires_grad_(True)
    optimizer = torch.optim.Adam([theta], lr=cal["lr_ini"])
    log(f"Computing initial steady state d0...")
    d0_steady = steady_state_d0_ks(theta, b_grid_t, y_grid_t, z_grid_t, r_grid_t, w_grid_t, Ty_t,
                                    nb_spg, ny_spg, nz_spg, nr_spg, nw_spg)
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
        d0_phase = d0_steady.detach() if warm_up else None
        L = spg_objective_ks(theta, cal["N_sample"], T_horizon, b_grid_t, y_grid_t, z_grid_t,
                             r_grid_t, w_grid_t, Ty_t, Tz_t, nb_spg, ny_spg, nz_spg, nr_spg, nw_spg,
                             beta, d0=d0_phase, warm_up=warm_up)
        loss_hist.append(L.item())
        (-L).backward()
        optimizer.step()
        param_change = (theta.detach() - theta_old).abs().max().item()

        if (epoch + 1) % 50 == 0:
            suffix = f"_ep{epoch+1:04d}"
            log(f"Exporting checkpoint at epoch {epoch+1} -> {args.out_dir}/loss_curve{suffix}.png etc.")
            save_visualizations(args.out_dir, theta, loss_hist, suffix=suffix)

        if param_change < cal["e_converge"]:
            break
        if (epoch + 1) % 10 == 0 or (epoch + 1) <= 5 or (epoch + 1) == cal["N_warmup"]:
            phase = "warm-up (d fixed)" if warm_up else "d evolves"
            elapsed = time.time() - start_time
            log(f"Epoch {epoch+1}/{cal['N_epoch']}, L(θ) = {L.item():.6f}, lr = {lr_t:.2e}, |Δθ| = {param_change:.2e}, {phase}, elapsed={elapsed:.1f}s")

    log(f"Exporting final visualizations -> {args.out_dir}/loss_curve.png etc.")
    save_visualizations(args.out_dir, theta, loss_hist, suffix="")

    with open(os.path.join(args.out_dir, "loss_hist.txt"), "w") as f:
        f.write("\n".join(map(str, loss_hist)))

    log(f"Done! Figures and loss_hist.txt saved to {args.out_dir}/")
    total_time = time.time() - start_time
    log(f"Total training time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")

if __name__ == "__main__":
    main()
