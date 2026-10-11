#!/bin/bash
#SBATCH --job-name=percol
#SBATCH --partition=MGPU-TC2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=30G
#SBATCH --time=06:00:00
#SBATCH --output=logs/percol_%j.out
# Per-column and per-pair breakdown of the error (E5), plus news utility on a log1p-transformed
# target (E7.1). For each cell: regenerates the student (1/2/4 evaluations), the teacher at
# 15/39/99 evaluations (all six reference settings on news) and F4, 10 seeds each; compares the
# regenerated samples with the stored scores (difference recorded; stops only above 0.1 pp);
# then scores every column and column pair.
# Usage: sbatch slurm/percol_one.sbatch [adult:0 adult:1 ...]   (default: all 9 cells)
# Skip-if-done per cell (results/<alias>_PERCOL_DONE). Needs study_one and fast_one to have run first.
set -eo pipefail
CELLS="${*:-adult:0 adult:1 adult:2 default:0 default:1 default:2 news:0 news:1 news:2}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate tabsyn
set -u
cd "$HOME/tabsyn"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
W="${SLURM_CPUS_PER_TASK:-4}"
source slurm/study_lib.sh
echo "== $(date) host=$(hostname) job=${SLURM_JOB_ID:-none} cells=$CELLS cpus=$W"
for c in $CELLS; do
  D="${c%%:*}"; K="${c##*:}"; A="${D}v_s${K}"
  echo "== $(date) CELL $A"; stage_percol
done
echo "== $(date) PERCOL_DONE"
