#!/bin/bash
#SBATCH --job-name=fastlo
#SBATCH --partition=MGPU-TC2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=30G
#SBATCH --time=02:00:00
#SBATCH --output=logs/fastlo_%j.out
# Low-noise check for the tuned 1- and 2-evaluation teacher baselines (F1, F2): tries starting
# noise sigma_max in {0.3, 0.5, 0.7}, below the main grid's smallest value of 1, chooses on
# validation, and scores on 20 seeds any setting that beats the main grid's choice.
# Usage: sbatch slurm/fast_lo.sbatch [adult:0 adult:1 ...]      (default: all 9 cells)
# Skip-if-done per cell (results/<alias>_FASTLO_DONE). Needs slurm/fast_one.sbatch to have run first.
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
for c in $CELLS; do D="${c%%:*}"; K="${c##*:}"; A="${D}v_s${K}"; echo "== $(date) CELL $A"; stage_fast_lo; done
echo "== $(date) FASTLO_ALL_DONE"
