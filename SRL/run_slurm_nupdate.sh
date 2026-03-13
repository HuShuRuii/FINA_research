#!/bin/bash
#SBATCH --job-name=hugget_nupd
#SBATCH --account=geopoltrade
#SBATCH --partition=amd
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=srl_nupdate_%j.out
#SBATCH --error=srl_nupdate_%j.err

set -euo pipefail

OUT_DIR="${1:-hugget_output_nupdate_v2}"
mkdir -p "$OUT_DIR"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"
mkdir -p "$MPLCONFIGDIR"

module purge
module load anaconda3/2023.09-0-biybti3

which python
python -m pip install --user -r /home/shuav/SRL/requirements.txt

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-64}"

# More conservative memory setting for the truncated G-gradient experiment.
# 对截断分布梯度实验使用更保守的内存配置，先优先保证能完整跑通。
echo "Running SRL Huggett JAX nupdate run: out_dir=$OUT_DIR, n_sample=128, n_update=8, g_grad_window=5" | tee -a "$OUT_DIR/slurm_env.log"

python -u /home/shuav/SRL/hugget_jax.py \
  --out_dir "$OUT_DIR" \
  --n_sample 128 \
  --g0_mode steady_high_mix \
  --g0_high_mix_warmup 0.8 \
  --g0_high_mix_after 0.5 \
  --g0_high_power 5 \
  --coverage_traj_share 0.25 \
  --coverage_decay_epochs 150 \
  --post_warmup_broad_share 1.0 \
  --post_warmup_beta_conc 0.7 \
  --n_update 8 \
  --g_grad_window 5 \
  --log_every 10 \
  --solve_steady_state \
  2>&1 | tee "$OUT_DIR/train.log"
