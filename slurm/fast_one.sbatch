#!/bin/bash
#SBATCH --job-name=fast
#SBATCH --partition=MGPU-TC2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=30G
#SBATCH --time=06:00:00
#SBATCH --output=logs/fast_%j.out
# Ablation plan E1: fair few-step teacher baseline for one dataset, training seeds 0 1 2 (or the ones given).
# Usage: sbatch slurm/fast_one.sbatch <dataset> [seeds...]      e.g. adult     (EXTRA=0 skips MLE/DCR)
# Skip-if-done per cell (results/<alias>_FAST_DONE); re-submit the same command after a kill.
set -eo pipefail
D="${1:?dataset name required}"; shift; SEEDS="${*:-0 1 2}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate tabsyn
set -u
cd "$HOME/tabsyn"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
W="${SLURM_CPUS_PER_TASK:-4}"
source slurm/study_lib.sh
echo "== $(date) host=$(hostname) job=${SLURM_JOB_ID:-none} dataset=$D seeds=$SEEDS cpus=$W extra=${EXTRA:-1}"
python tools/test_samplers.py 2>&1 | tail -3
for K in $SEEDS; do
  A="${D}v_s${K}"
  [ -f "tabsyn/ckpt/$A/DONE" ] || { echo "FAIL: no teacher for $A"; exit 3; }
  echo "== $(date) CELL $A"; stage_fast
done
echo "== $(date) FAST_ONE_DONE $D"
