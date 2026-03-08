# Huggett SPG 每期时间复杂度与瓶颈分析 / Per-period time complexity and bottlenecks

## MacBook Pro (Apple Silicon) 使用说明

- **设备选择:** 在 Mac 上会自动使用 **MPS**（Metal Performance Shaders），无需改代码。`hugget.ipynb` 与 `run_hugget_cluster.py` 均为：CUDA → MPS → CPU。
- **命令行:** `python run_hugget_cluster.py` 在 Mac 上会打 `Device: mps (Apple Silicon)`。若 MPS 报错可强制 CPU：`python run_hugget_cluster.py --device cpu`。
- **同步成本:** MPS 下 `.item()` / `.cpu().numpy()` 会触发 device↔CPU 同步，本仓库已做优化（见下文「已做优化」）。尽量保持热路径在 device 上。

---

## 符号 / Notation

- `nb_spg` = 50, `ny` = 3, `J` = nb×ny = 150  
- `nz_spg` = 10, `nr_spg` = 10  
- `N_traj` = 512 (每 epoch 轨迹数), `T_horizon` ≈ 50 (每轨迹期数)  
- 设备: GPU/MPS 时存在 device↔host 拷贝与同步

---

## 每 epoch 结构 / Per-epoch structure

```
for epoch in range(N_epoch):
    L = spg_objective(theta, N_traj, T_horizon, ...)   # 1 次
    (-L).backward()
    optimizer.step()
```

**spg_objective 内部:**

```
for n in range(N_traj):                    # N_traj 次
    G = G0.clone()                        # ① 拷贝 G0 (nb×ny)
    for t in range(T_horizon):            # T_horizon 次
        r_t = P_star_detach(...)          # ②
        ir = r_to_ir(r_t, r_grid_t)       # ③
        c_t = consumption_at_r_continuous(...)  # ④
        L_n = L_n + ...
        G = update_G_pi_direct(..., r_val=r_t)    # ⑤ (内层再次调 ④)
        iz = torch.multinomial(...).item()       # ⑥
    L_list.append(L_n)
return torch.stack(L_list).mean()
```

---

## 每个时间步 t 的详细复杂度与瓶颈

### ② P_star_detach(theta, G, iz, ...)

| 操作 | 时间复杂度 | 拷贝/同步 | 说明 |
|------|------------|-----------|------|
| theta_to_consumption_grid(theta) | O(nz·nr·nb·ny) | 无 | 整张 c 网格 |
| c_all = c[iz,:,:,:].permute().reshape | O(nr·nb·ny) | 无 | 切片+重排 |
| b_flat, y_flat repeat | O(nb·ny) | 无 | |
| resources, b_next_all, S_all | O(nb·ny·nr) | 无 | 向量化 |
| **binary search 中 S_all[mid].item()** | **O(log nr) 次** | **每次 .item() = 1 次 GPU→CPU 同步** | **瓶颈 1** |
| S_lo, S_hi = S_all[ir_lo].item(), ... | 2 次 | 2 次 GPU→CPU 同步 | **瓶颈 1** |
| r_star.detach() 返回 | O(1) | 无 | |

**小计:** 计算量 O(nb·ny·nr) 可接受；**主要代价是 binary search 中约 log₂(nr)+2 次 `.item()`，每次都会导致 device 同步，GPU/MPS 上极贵。**

---

### ③ r_to_ir(r_t, r_grid_t)

| 操作 | 时间复杂度 | 拷贝/同步 | 说明 |
|------|------------|-----------|------|
| r = r_t.item() | O(1) | **1 次 GPU→CPU 同步** | **瓶颈 2** |
| grid = r_grid_t.cpu().numpy() | O(nr) | **1 次 拷贝 + 同步** | **瓶颈 2** |
| np.searchsorted + clip | O(log nr) | 无 | CPU |

**小计:** **每期 1 次 .item() + 1 次 .cpu().numpy()，即每期至少 2 次 device 同步。**

---

### ④ consumption_at_r_continuous(theta, iz, r_val, ...)

| 操作 | 时间复杂度 | 拷贝/同步 | 说明 |
|------|------------|-----------|------|
| theta_to_consumption_grid(theta) | O(nz·nr·nb·ny) | 无 | **整张 c 重新算+可能新分配** |
| r_val.item() | O(1) | **1 次 GPU→CPU 同步** | |
| r_grid_t.cpu().numpy() | O(nr) | **1 次 拷贝+同步** | **瓶颈 2（重复）** |
| c[iz, ir_lo, :, :], c[iz, ir_hi, :, :] | O(nb·ny) | 无 | |

**该函数每期被调用 2 次**（一次算 L_n 的 c_t，一次在 update_G_pi_direct 里用 r_val 算 c_val），所以 **每期至少 4 次 .item()/.cpu().numpy() 类同步**（若 r_to_ir 与 consumption_at_r 都调用）。

---

### ⑤ update_G_pi_direct(..., r_val=r_t)

| 操作 | 时间复杂度 | 拷贝/同步 | 说明 |
|------|------------|-----------|------|
| consumption_at_r_continuous(...) | 见 ④ | 同上 | 第二次调用 |
| r_use = r_val.item() | O(1) | 1 次同步 | |
| b_next = ... | O(J) | 无 | |
| torch.searchsorted(b_grid_t, b_next) | O(J log nb) | 无 | 在 device 上 |
| **eye = torch.eye(nb_spg, ...)** | **O(nb²)** | **每期新建 (nb×nb) 矩阵** | **瓶颈 3：分配+可能初始化** |
| w_b = w_lo*eye[idx_lo] + w_hi*eye[idx_hi] | O(J·nb) | 无 | 大索引矩阵 |
| M, Q, G_new = Q @ Ty | O(nb²·ny) | 无 | |

**小计:** **每期一次 `torch.eye(nb_spg)`，50×50=2500 元素，每轨迹每期都重新分配+创建，总次数 N_traj × T_horizon。**

---

### ⑥ iz = torch.multinomial(Tz_t[iz, :], 1).squeeze().item()

| 操作 | 时间复杂度 | 拷贝/同步 | 说明 |
|------|------------|-----------|------|
| multinomial | O(nz_spg) | 无 | |
| .item() | O(1) | **1 次 GPU→CPU 同步** | **瓶颈 2** |

---

## ① G0.clone()

- 每轨迹 1 次，O(nb·ny)，拷贝量小；若 G0 在 GPU 上，clone 在 device 上，无 host 同步。

---

## 瓶颈汇总（按影响排序）

1. **Device 同步（.item() / .cpu().numpy()）**  
   - **P_star_detach:** 二分查找中约 **log₂(nr)+2 次 .item()** 每期 → 约 5 次同步/期。  
   - **r_to_ir:** 每期 **1 次 .item() + 1 次 .cpu().numpy()**。  
   - **consumption_at_r_continuous:** 每期调用 2 次，每次 **.item() + .cpu().numpy()** → 再 4 次同步/期。  
   - **multinomial(...).item():** 每期 1 次。  
   - **合计约 10+ 次 device 同步/期**，在 GPU/MPS 上会拖慢整体 10–100 倍。

2. **torch.eye(nb_spg) 每期新建**  
   - 每期 O(nb²) 分配与写入，**N_traj × T_horizon** 次（例如 512×50 = 25,600 次/epoch）。  
   - 可改为 **预分配一次、全程复用** 或 用 one-hot/scatter 替代 full eye。

3. **theta_to_consumption_grid 重复计算**  
   - P_star_detach 内算一次整张 c；consumption_at_r_continuous 被调 2 次，每次内部再算一次整张 c → 每期 **至少 3 次** 全网格 clamp。  
   - 可在一期内的 P_star、L_n、update_G 之间 **只算一次 c，复用**。

4. **r_grid 重复拷贝**  
   - r_to_ir 与 consumption_at_r_continuous 每次都 `r_grid_t.cpu().numpy()`，r_grid 可 **在 epoch 或程序开始时转为 numpy 一次，传入使用**，避免每期拷贝。

---

## 每期总复杂度（量级）

- **计算量:** O(nz·nr·nb·ny) + O(nb·ny·nr) + O(J log nb) + O(nb²·ny) ≈ O(nb²·ny) 主导（若 nb=50, ny=3）。  
- **真正拖慢的是:**  
  - **同步次数:** O(1) 但常数很大（10+ 次/期），且每次同步阻塞整条 pipeline。  
  - **重复分配:** O(nb²) 的 eye 每期一次。  
  - **重复计算:** 同一期 c 网格算多遍。

---

## 建议修改（优先级）

1. **去掉 P_star_detach 中二分查找的 .item()**  
   - 用 `torch.argmin` 在 device 上求 bracket（或整根 S_all 在 device 上算，再一次取 r_star），**只保留最后 1 次 .detach() 或 .item() 用于 r_to_ir**。

2. **r_grid 只转 numpy 一次**  
   - 在 spg_objective 外或 epoch 初：`r_grid_np = r_grid_t.cpu().numpy()` 一次；r_to_ir 和 consumption_at_r_continuous 只接收/使用 `r_grid_np`，不再在热路径里 .cpu().numpy()。

3. **预分配并复用 torch.eye(nb_spg)**  
   - 在模块或循环外创建一次 `eye_nb = torch.eye(nb_spg, device=..., dtype=...)`，update_G_pi_direct 中改为使用该 `eye_nb`。

4. **每期只算一次 c，复用**  
   - 在单期循环内：先 P_star_detach（内部算 c 一次）；再用同一 c 和 r_star 算 c_t（线性插值）和 update_G_pi_direct 的 c_val，避免 consumption_at_r_continuous 内重复 theta_to_consumption_grid。

5. **multinomial 的 .item()**  
   - 若需完全留在 device 上，可让 iz 保持为 0-dim tensor 参与后续索引（部分代码需接受 tensor 索引）；否则保留 .item() 但同步次数已因上面几项大幅减少。

实施上述 1–4 后，每期 device 同步可从 10+ 次降到约 1–2 次，且去掉每期 eye 分配与重复 c 计算，预期可明显改善“计算时间不对”的问题。

---

## 已做优化（run_hugget_cluster.py 与 hugget.ipynb）

- **P_star_detach:** 二分查找改为在 device 上向量化求 bracket（`ge.cumsum` + `argmax`），不再在循环内调用 `.item()`。
- **r_grid_np:** 在开始时一次 `r_grid_t.cpu().numpy()`，`r_to_ir` 与 `consumption_at_r_continuous` 使用 `r_grid_np`，热路径不再重复 `.cpu().numpy()`。
- **eye_nb:** 预分配 `torch.eye(nb_spg, ...)` 一次，`update_G_pi_direct` 中复用，不再每期新建。
- **r_to_ir / consumption_at_r_continuous:** 增加可选参数 `r_grid_np`，传入时避免 device 拷贝。
