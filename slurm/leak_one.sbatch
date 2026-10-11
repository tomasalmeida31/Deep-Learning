#!/bin/bash
#SBATCH --job-name=leak
#SBATCH --partition=MGPU-TC2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=30G
#SBATCH --time=06:00:00
#SBATCH --output=logs/leak_%j.out
# Leak-delta row: the paper's leaky VAE-selection protocol on identical train' (alias <d>L_s0), seed 0.
# Usage: sbatch slurm/leak_one.sbatch <dataset>     Skip-if-done at every stage; writes results/<d>L_s0_LEAK_DONE.
set -eo pipefail
D="${1:?dataset name required}"; K=0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate tabsyn
set -u
cd "$HOME/tabsyn"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
W="${SLURM_CPUS_PER_TASK:-4}"
A="${D}L_s${K}"
source slurm/study_lib.sh
echo "== $(date) host=$(hostname) job=${SLURM_JOB_ID:-none} dataset=$D alias=$A (leaky protocol) cpus=$W"
# --legacy_target_in_fix: rebuild the exact split used for our results. Since 1 Oct the script's
# default rule leaves news with 37 fewer training rows; adult and default are identical either way.
python tools/make_val_split.py --dataname "$D" --seeds "$K" --legacy_target_in_fix
[ -f "data/$A/info.json" ] || { echo "FAIL: alias $A missing"; exit 3; }
echo "== $(date) STAGE teacher";       stage_teacher
echo "== $(date) STAGE teacher curve"; stage_teacher_curve
echo "== $(date) STAGE teacher extra"; stage_teacher_extra
date > "results/${A}_LEAK_DONE"
echo "== $(date) LEAK_DONE $A"
