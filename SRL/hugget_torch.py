#!/usr/bin/env python3
"""
Huggett (1993) SRL/SPG — PyTorch reference implementation.
Huggett (1993) SRL/SPG —— PyTorch 参考实现。

This file mirrors the current JAX training logic closely enough for code reading:
1. simulate a full T-period path with prices detached,
2. sample Nupdate time steps from each path,
3. replay only a short g_grad_window around each sampled step.
这个文件尽量与当前 JAX 训练逻辑保持一致，便于阅读：
1. 先做完整 T 期前向模拟，并对价格更新截断梯度，
2. 从每条路径抽取 Nupdate 个时点，
3. 只对每个抽样时点附近的短窗口 g_grad_window 做分布梯度回放。

Dependencies: numpy, matplotlib, scipy, torch.
依赖：numpy、matplotlib、scipy、torch。
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
    p = argparse.ArgumentParser(description="Run Huggett SRL in PyTorch and save diagnostics.")
    p.add_argument("--out_dir", type=str, default="hugget_output", help="Directory for figures and logs")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--epochs", type=int, default=None, help="Max SPG epochs (default: from calibration)")
    p.add_argument("--quick", action="store_true", help="Short run: 20 epochs, small sample")
    p.add_argument("--device", type=str, default=None, choices=("cpu", "cuda", "mps"), help="Force device (default: cuda if available, else mps on Mac, else cpu)")
    p.add_argument("--n_sample", type=int, default=None, help="Override trajectories per epoch / 覆盖每个 epoch 的轨迹数")
    p.add_argument("--n_warmup", type=int, default=None, help="Override warm-up epochs / 覆盖 warm-up 轮数")
    p.add_argument("--lr_ini", type=float, default=None, help="Override initial lr / 覆盖初始学习率")
    p.add_argument("--lr_decay", type=float, default=None, help="Override lr decay / 覆盖学习率衰减")
    p.add_argument("--log_every", type=int, default=10, help="Progress print frequency / 日志频率")
    p.add_argument("--n_update", type=int, default=16, help="Sampled time steps per path used for gradients / 每条路径用于梯度的抽样时点数")
    p.add_argument("--g_grad_window", type=int, default=10, help="Truncated replay window for G gradients / 分布梯度的截断窗口长度")
    return p.parse_args()

# ---------- Calibration (SRL Table 2 & 3) ----------
def get_calibration(quick=False):
    beta, sigma = 0.96, 2.0
    rho_y, nu_y = 0.6, 0.2
    rho_z, nu_z = 0.9, 0.02
    B, b_min = 0.0, -1.0
    nb, b_max = 200, 50.0
    ny, nr, nz = 3, 20, 30
    r_min, r_max = 0.01, 0.06
    c_min = 1e-3
    T_trunc = 170
    e_trunc = 1e-3  # stop trajectory when beta**t < e_trunc (remaining utility negligible)
    N_epoch, N_warmup = (30, 5) if quick else (1000, 50)
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

# ---------- Policy from grid: b continuous (lottery); r can be index or value (linear interp if value) ----------
def policy_from_grid(b, iy, iz, ir_or_r, c_grid, b_grid, y_grid, z_grid, r_grid, c_min_val=1e-3):
    """b: continuous (lottery). ir_or_r: int = grid index; float = r value (linear interp in r for c). Returns (c, b_next)."""
    b = np.atleast_1d(np.asarray(b, dtype=float))
    nb, nr = len(b_grid), len(r_grid)
    if isinstance(ir_or_r, (int, np.integer)):
        ir_lo = int(np.clip(ir_or_r, 0, nr - 1))
        ir_hi = ir_lo
        w_r = 0.0
        r_use = float(r_grid[ir_lo])
    else:
        r_val = float(ir_or_r)
        r_use = r_val
        ir_lo = int(np.clip(np.searchsorted(r_grid, r_val, side="right") - 1, 0, nr - 2))
        ir_hi = ir_lo + 1
        w_r = (r_val - r_grid[ir_lo]) / max(r_grid[ir_hi] - r_grid[ir_lo], 1e-20)
    b_c = np.clip(b, b_grid[0], b_grid[-1])
    j_hi = np.clip(np.searchsorted(b_grid, b_c), 1, nb - 1)
    j_lo = j_hi - 1
    w_b = (b_c - b_grid[j_lo]) / np.maximum(b_grid[j_hi] - b_grid[j_lo], 1e-20)
    c = (1 - w_r) * ((1 - w_b) * c_grid[j_lo, iy, iz, ir_lo] + w_b * c_grid[j_hi, iy, iz, ir_lo])
    c += w_r * ((1 - w_b) * c_grid[j_lo, iy, iz, ir_hi] + w_b * c_grid[j_hi, iy, iz, ir_hi])
    c = np.maximum(c, c_min_val)
    c_total = (1 + r_use) * b + y_grid[iy] * z_grid[iz]
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
    if args.n_sample is not None:
        cal["N_sample"] = args.n_sample
    if args.n_warmup is not None:
        cal["N_warmup"] = args.n_warmup
    if args.lr_ini is not None:
        cal["lr_ini"] = args.lr_ini
    if args.lr_decay is not None:
        cal["lr_decay"] = args.lr_decay

    beta = cal["beta"]
    sigma = cal["sigma"]
    c_min = cal["c_min"]
    b_min = cal["b_min"]
    b_max = cal["b_max"]
    ny = cal["ny"]
    nb, nr, nz = cal["nb"], cal["nr"], cal["nz"]
    r_min, r_max = cal["r_min"], cal["r_max"]

    # Grids (numpy) / 网格（先用 NumPy 构造，再转 Torch）
    b_grid = np.linspace(b_min, b_max, nb)
    r_grid = np.linspace(r_min, r_max, nr)
    y_grid, Ty = tauchen_ar1(cal["rho_y"], cal["nu_y"], ny, m=3, mean=1.0)
    invariant_y = np.linalg.matrix_power(Ty.T, 200)[:, 0]
    y_grid = y_grid / (y_grid @ invariant_y)
    log_z_grid, Tz = tauchen_ar1(cal["rho_z"], cal["nu_z"], nz)
    z_grid = np.exp(log_z_grid)
    invariant_z = np.linalg.matrix_power(Tz.T, 200)[:, 0]
    z_grid = z_grid / (z_grid @ invariant_z)

    # Device: prefer CUDA, then MPS (MacBook Pro Apple Silicon), then CPU
    if args.device is not None:
        device = torch.device(args.device)
        device_name = str(device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = "cuda"
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available() and getattr(torch.backends.mps, "is_built", lambda: True)():
        device = torch.device("mps")
        device_name = "mps (Apple Silicon)"
    else:
        device = torch.device("cpu")
        device_name = "cpu"
    log(f"Device: {device_name}")
    dtype = torch.float32
    # Use the paper/full grid by default so the PyTorch code matches the JAX logic.
    # 默认使用论文 full grid，这样 PyTorch 代码和 JAX 逻辑一致。
    nb_spg, nr_spg, nz_spg = nb, nr, nz
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
    # M3 UMA: .cpu() copies are free (shared DRAM), but every .item() flushes the Metal queue.
    # Precompute all constants once so the hot path is free of device syncs.
    r_grid_np      = r_grid_t.cpu().numpy()
    eye_nb         = torch.eye(nb_spg, device=device, dtype=dtype)
    b_flat_precomp = b_grid_spg.repeat_interleave(ny)   # (J,)  J = nb*ny
    y_flat_precomp = y_grid_t.repeat(nb_spg)             # (J,)
    Tz_np          = Tz_t.cpu().numpy()
    Tz_cdf_np      = Tz_np.cumsum(axis=1)                # (nz,nz) for vectorised CPU z-sampling

    def theta_to_consumption_grid(theta, *_, c_min_val=1e-3):
        return torch.clamp(theta, min=c_min_val)

    def init_theta(b_grid_t, y_grid_t, z_grid_t, r_grid_t, save_frac=0.2, c_min_val=1e-3):
        nb_t, ny_t = len(b_grid_t), len(y_grid_t)
        nz_t, nr_t = len(z_grid_t), len(r_grid_t)
        b_flat = b_grid_t.repeat_interleave(ny_t)
        y_flat = y_grid_t.repeat(nb_t)
        cash = (b_flat.view(1, 1, -1) * (1 + r_grid_t).view(1, nr_t, 1)
                + y_flat.view(1, 1, -1) * z_grid_t.view(nz_t, 1, 1))
        cash = cash.view(nz_t, nr_t, nb_t, ny_t)
        return torch.clamp((1 - save_frac) * cash, min=c_min_val)

    def _G_to_mat_spg(G, nb_spg, ny):
        return G.view(nb_spg, ny) if G.dim() == 1 else G

    def u_torch(c_vec, sig=sigma):
        c_vec = torch.clamp(c_vec, min=c_min)
        if abs(sig - 1.0) < 1e-8:
            return torch.log(c_vec)
        return (c_vec ** (1 - sig)) / (1 - sig)

    iy_flat_precomp = torch.arange(ny, device=device, dtype=torch.long).repeat(nb_spg)  # (J,)

    # ── P_star: all on device, returns bracket for reuse ─────────────────────
    def P_star_bracket(c_iz_flat, G_mat, z_val, b_flat, y_flat, b_grid_t, r_grid_t, B=0.0):
        """Returns (r_star, ir_lo, ir_hi, w_r) -- no .item()."""
        nr         = len(r_grid_t)
        G_flat     = G_mat.reshape(-1)                                    # (J,)
        resources  = b_flat.unsqueeze(1) * (1 + r_grid_t) + (y_flat * z_val).unsqueeze(1)  # (J, nr)
        b_next_all = (resources - c_iz_flat).clamp(b_min, b_max)         # (J, nr)
        S_all      = G_flat @ b_next_all                                  # (nr,) -- one matmul
        ge         = (S_all >= B).to(r_grid_t.dtype)
        ir_hi      = (ge.cumsum(0) >= 1).to(r_grid_t.dtype).argmax(0).clamp(1, nr - 1)
        ir_lo      = (ir_hi - 1).clamp(0, nr - 2)
        S_lo, S_hi = S_all[ir_lo], S_all[ir_hi]
        w_r        = ((B - S_lo) / (S_hi - S_lo).clamp(1e-20)).clamp(0.0, 1.0)
        r_star     = (r_grid_t[ir_lo] + w_r * (r_grid_t[ir_hi] - r_grid_t[ir_lo])).detach()
        return r_star, ir_lo, ir_hi, w_r

    def P_star_detach(theta, G, iz, b_grid_t, y_grid_t, z_grid_t, r_grid_t, ny, B=0.0):
        if len(r_grid_t) == 1:
            return r_grid_t[0].detach()
        G_mat     = _G_to_mat_spg(G, len(b_grid_t), ny)
        c         = theta_to_consumption_grid(theta)
        c_iz_flat = c[iz].permute(1, 2, 0).reshape(len(b_grid_t) * ny, len(r_grid_t))
        r_star, *_ = P_star_bracket(c_iz_flat, G_mat, z_grid_t[iz],
                                     b_flat_precomp, y_flat_precomp, b_grid_t, r_grid_t, B)
        return r_star

    # ── G update: scatter_add replaces dense (J x nb) eye_nb lottery matrix ───
    # Old: eye_nb[idx] builds (150, 50) matrices x 2  => 60 KB/call
    # New: scatter_add on (J,) vectors                =>  2 KB/call
    def _update_G_from_ct(c_t_flat, r_star, z_val, G_mat, b_grid_t, Ty_t, nb_spg, ny):
        b_next     = ((1 + r_star) * b_flat_precomp
                      + y_flat_precomp * z_val
                      - c_t_flat).clamp(b_grid_t[0], b_grid_t[-1])
        idx_hi     = torch.searchsorted(b_grid_t, b_next).clamp(1, nb_spg - 1)
        idx_lo     = idx_hi - 1
        w_hi       = (b_next - b_grid_t[idx_lo]) / (b_grid_t[idx_hi] - b_grid_t[idx_lo]).clamp(1e-20)
        G_flat     = G_mat.reshape(-1)
        G_new_flat = torch.zeros_like(G_flat)
        dest_lo    = idx_lo * ny + iy_flat_precomp     # (J,)
        dest_hi    = idx_hi * ny + iy_flat_precomp
        G_new_flat.scatter_add_(0, dest_lo, (1.0 - w_hi) * G_flat)
        G_new_flat.scatter_add_(0, dest_hi, w_hi * G_flat)
        G_new = G_new_flat.view(nb_spg, ny) @ Ty_t
        return G_new / (G_new.sum() + 1e-20)

    # Legacy wrappers kept for any external callers
    def consumption_at_r_continuous(theta, iz, r_val, b_grid_t, y_grid_t, z_grid_t, r_grid_t, c_min_val=1e-3, r_grid_np=None):
        c    = theta_to_consumption_grid(theta)
        _nr  = len(r_grid_t)
        ge_s = (r_grid_t <= r_val).to(r_grid_t.dtype)
        irl  = (ge_s.sum() - 1).clamp(0, _nr - 2).long()
        irh  = (irl + 1).clamp(0, _nr - 1)
        wr   = ((r_val - r_grid_t[irl]) / (r_grid_t[irh] - r_grid_t[irl]).clamp(1e-20)).clamp(0, 1)
        return (1 - wr) * c[iz, irl] + wr * c[iz, irh]

    def update_G_pi_direct(theta, G, iz, ir, b_grid_t, y_grid_t, z_grid_t, r_grid_t, Ty_t, nb_spg, ny, r_val=None, eye_nb=None):
        G_mat = _G_to_mat_spg(G, nb_spg, ny)
        if r_val is not None:
            c_t   = consumption_at_r_continuous(theta, iz, r_val, b_grid_t, y_grid_t, z_grid_t, r_grid_t, c_min)
            r_use = r_val
        else:
            c_t   = theta_to_consumption_grid(theta)[iz, ir]
            r_use = r_grid_t[ir]
        return _update_G_from_ct(c_t.reshape(-1), r_use, z_grid_t[iz], G_mat, b_grid_t, Ty_t, nb_spg, ny)

    # ── Steady state — fixed iterations; removes 150 .abs().max().item() syncs ─
    def steady_state_G0(theta, b_grid_t, y_grid_t, z_grid_t, r_grid_t, Ty_t, nb_spg, ny, nz_spg, nr_spg, n_iter=200, tol=1e-6):
        iz_mid = nz_spg // 2
        ir_m   = nr_spg // 2
        with torch.no_grad():
            G     = torch.ones(nb_spg, ny, device=device, dtype=dtype) / (nb_spg * ny)
            c     = theta_to_consumption_grid(theta)
            c_t   = c[iz_mid, ir_m]                  # (nb, ny) — fixed policy slice
            r_use = r_grid_t[ir_m]
            z_val = z_grid_t[iz_mid]
            for _ in range(n_iter):
                G = _update_G_from_ct(c_t.reshape(-1), r_use, z_val, G, b_grid_t, Ty_t, nb_spg, ny)
        return G

    # ── Vectorised z-path sampler — pure numpy, zero MPS touch ───────────────
    def _presample_iz_paths(N_traj, T_eff):
        """Pre-sample all z Markov paths on CPU.  Eliminates N_traj×T_eff multinomial.item() Metal flushes."""
        paths       = np.empty((N_traj, T_eff + 1), dtype=np.int32)
        paths[:, 0] = np.random.randint(0, nz_spg, size=N_traj)
        for t in range(T_eff):
            rows        = paths[:, t]
            u           = np.random.rand(N_traj, 1)
            paths[:, t + 1] = (Tz_cdf_np[rows] < u).sum(axis=1).clip(0, nz_spg - 1)
        return paths

    # ── One-period step: accepts pre-computed c -- 0 theta_to_c calls inside ──
    def _one_period_step(c, G_mat, iz, beta_pow, b_grid_t, Ty_t, nb_spg, ny):
        """c: pre-computed consumption grid (nz, nr, nb, ny).  No recomputation."""
        c_iz      = c[iz]                                              # (nr, nb, ny) -- ONE slice
        c_iz_flat = c_iz.permute(1, 2, 0).reshape(nb_spg * ny, nr_spg)
        z_val     = z_grid_t[iz]
        r_star, ir_lo, ir_hi, w_r = P_star_bracket(
            c_iz_flat, G_mat, z_val, b_flat_precomp, y_flat_precomp, b_grid_t, r_grid_t)
        c_t    = (1 - w_r) * c_iz[ir_lo] + w_r * c_iz[ir_hi]         # (nb, ny)
        L_term = beta_pow * (G_mat.detach() * u_torch(c_t)).sum()
        G_new  = _update_G_from_ct(c_t.reshape(-1), r_star, z_val, G_mat, b_grid_t, Ty_t, nb_spg, ny)
        return L_term, G_new.detach()

    # ── Time-minibatch objective / 时间 minibatch 目标函数 ─────────────────────
    # We first simulate the whole path without storing a full gradient graph,
    # then replay only short windows around sampled time steps.
    # 先无梯度地模拟整条路径，再只对抽样时点附近的短窗口重放梯度。
    def simulate_path_no_grad_single_traj(theta, G0, iz0, T_horizon, beta_t, warm_up):
        """Simulate one full path while treating prices as environment objects.
        无梯度地模拟一条完整路径，把价格更新看成环境给定对象。"""
        G_hist = []
        iz_hist = []
        G_cur = _G_to_mat_spg(G0, nb_spg, ny).detach()
        iz_cur = int(iz0)
        c = theta_to_consumption_grid(theta).detach()
        T_eff = T_horizon
        for t in range(T_horizon):
            if (beta_t ** t) < cal["e_trunc"]:
                T_eff = t
                break
            G_hist.append(G_cur.clone())
            iz_hist.append(iz_cur)
            c_iz = c[iz_cur]
            c_iz_flat = c_iz.permute(1, 2, 0).reshape(nb_spg * ny, nr_spg)
            r_star, ir_lo, ir_hi, w_r = P_star_bracket(
                c_iz_flat, G_cur, z_grid_t[iz_cur], b_flat_precomp, y_flat_precomp, b_grid_t, r_grid_t
            )
            c_t = (1 - w_r) * c_iz[ir_lo] + w_r * c_iz[ir_hi]
            if not warm_up:
                G_cur = _update_G_from_ct(c_t.reshape(-1), r_star, z_grid_t[iz_cur], G_cur, b_grid_t, Ty_t, nb_spg, ny).detach()
            probs = Tz_t[iz_cur].detach().cpu()
            iz_cur = int(torch.multinomial(probs, 1).item())
        G_hist.append(G_cur.clone())
        iz_hist.append(iz_cur)
        return G_hist, iz_hist, G_cur.reshape(-1), T_eff

    def truncated_time_batch_objective_single_traj(theta, G_path, iz_path, sample_ts, T_eff, n_update_eff, g_grad_window, beta_t, warm_up):
        """Replay only sampled time steps with a short differentiable G window.
        只对抽样时点做短窗口回放，从而保留局部分布梯度。"""
        c = theta_to_consumption_grid(theta)
        window = max(int(g_grad_window), 1)
        scale = float(T_eff) / float(max(n_update_eff, 1))
        terms = []
        for t_idx in sample_ts.tolist():
            start = max(int(t_idx) - window + 1, 0)
            G_cur = G_path[start].clone().detach()
            term_t = torch.zeros((), device=device, dtype=dtype)
            for global_t in range(start, int(t_idx) + 1):
                iz_cur = int(iz_path[global_t])
                c_iz = c[iz_cur]
                c_iz_flat = c_iz.permute(1, 2, 0).reshape(nb_spg * ny, nr_spg)
                r_star, ir_lo, ir_hi, w_r = P_star_bracket(
                    c_iz_flat, G_cur, z_grid_t[iz_cur], b_flat_precomp, y_flat_precomp, b_grid_t, r_grid_t
                )
                c_t = (1 - w_r) * c_iz[ir_lo] + w_r * c_iz[ir_hi]
                weight = (beta_t ** global_t) if (beta_t ** global_t) >= cal["e_trunc"] else 0.0
                if global_t == int(t_idx):
                    term_t = weight * (G_cur * u_torch(c_t)).sum()
                if not warm_up:
                    G_cur = _update_G_from_ct(c_t.reshape(-1), r_star.detach(), z_grid_t[iz_cur], G_cur, b_grid_t, Ty_t, nb_spg, ny)
            terms.append(term_t)
        if not terms:
            return torch.zeros((), device=device, dtype=dtype)
        return scale * torch.stack(terms).mean()

    def spg_objective_single_traj(theta, G0, iz0, T_horizon, n_update, g_grad_window, beta_t, warm_up):
        """Single-trajectory objective plus final distribution.
        单条轨迹的目标值与终端分布。"""
        G_path, iz_path, G_final, T_eff = simulate_path_no_grad_single_traj(theta, G0, iz0, T_horizon, beta_t, warm_up)
        n_update_eff = min(max(int(n_update), 1), int(T_eff))
        sample_ts = torch.randperm(T_eff, device=device)[:n_update_eff]
        L_n = truncated_time_batch_objective_single_traj(
            theta, G_path, iz_path, sample_ts, T_eff, n_update_eff, g_grad_window, beta_t, warm_up
        )
        return L_n, G_final

    def spg_objective(theta, N_traj, T_horizon, n_update, g_grad_window, beta_t, G0=None, warm_up=False):
        """Mean SPG objective across trajectories using time minibatching.
        通过时间 minibatch 计算跨轨迹平均目标。"""
        if G0 is None:
            G0 = torch.ones(nb_spg, ny, device=device, dtype=dtype) / (nb_spg * ny)
        else:
            G0 = _G_to_mat_spg(G0, nb_spg, ny)

        L_list = []
        G_final_list = []
        iz_paths = _presample_iz_paths(N_traj, T_horizon)
        for n in range(N_traj):
            L_n, G_final = spg_objective_single_traj(
                theta, G0, int(iz_paths[n, 0]), T_horizon, n_update, g_grad_window, beta_t, warm_up
            )
            L_list.append(L_n)
            G_final_list.append(G_final)
        return torch.stack(L_list).mean(), torch.stack(G_final_list).mean(dim=0)

    def save_visualizations(out_dir, theta_cur, loss_cur, suffix=""):
        """Save loss curve + consumption policy + c vs r to out_dir. suffix e.g. '_ep050' for checkpoints."""
        c_grid = theta_to_consumption_grid(theta_cur.detach(), c_min_val=c_min)  # (nz, nr, nb, ny)
        c_grid_np = c_grid.permute(2, 3, 0, 1).cpu().numpy()   # (nb, ny, nz, nr) for policy_from_grid
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
    log(f"Grid sizes: b={nb_spg}, y={ny}, z={nz_spg}, r={nr_spg}")
    log(f"Time minibatch: n_update={args.n_update}, g_grad_window={args.g_grad_window}, T_horizon={cal['T_trunc']}")
    log(f"Exporting visualizations every 50 epochs to {args.out_dir}/")

    theta = init_theta(b_grid_spg, y_grid_t, z_grid_t, r_grid_t)
    theta = theta.requires_grad_(True)
    optimizer = torch.optim.Adam([theta], lr=cal["lr_ini"])
    log(f"Computing initial steady state G0...")
    G0_steady = steady_state_G0(theta, b_grid_spg, y_grid_t, z_grid_t, r_grid_t, Ty_t, nb_spg, ny, nz_spg, nr_spg)
    T_horizon = cal["T_trunc"]
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
        L, G_end_mean = spg_objective(
            theta, cal["N_sample"], T_horizon, args.n_update, args.g_grad_window,
            beta, G0=G0_phase, warm_up=warm_up
        )
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
        if (epoch + 1) % args.log_every == 0 or (epoch + 1) <= 5 or (epoch + 1) == cal["N_warmup"]:
            phase = "warm-up (G fixed)" if warm_up else "G evolves"
            elapsed = time.time() - start_time
            g_mass = float(G_end_mean.sum().item()) if not warm_up else 1.0
            log(f"Epoch {epoch+1}/{cal['N_epoch']}, L(θ) = {L.item():.6f}, lr = {lr_t:.2e}, |Δθ| = {param_change:.2e}, {phase}, G_end_sum={g_mass:.4f}, elapsed={elapsed:.1f}s")

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
