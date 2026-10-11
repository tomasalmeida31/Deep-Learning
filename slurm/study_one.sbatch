#!/bin/bash
#SBATCH --job-name=study
#SBATCH --partition=MGPU-TC2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=30G
#SBATCH --time=06:00:00
#SBATCH --output=logs/study_%j.out
# Full study, one (dataset, training seed) cell:
#   teacher (VAE + EDM) at seed K on the leak-free alias  ->  teacher error-vs-NFE curve (20 sampling seeds)
#   ->  CD student with the frozen recipe (tau on val, 20 seeds on train')  ->  C2ST / MLE / DCR (C4) for
#   references, the six teacher reference configs and the student.
# Usage: sbatch slurm/study_one.sbatch <dataset> <seed>       e.g. adult 1
# Env overrides: MU (EMA-target rate, default 0.999), STEPS (CD steps, default 20000), EXTRA_SEEDS (sampling seeds scored by MLE/DCR, default 10).
# Every stage is skip-if-done: re-submit the same command after a kill/timeout and it resumes.
set -eo pipefail
D="${1:?dataset name required}"; K="${2:?training seed required}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate tabsyn
set -u
cd "$HOME/tabsyn"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
W="${SLURM_CPUS_PER_TASK:-4}"
MU="${MU:-0.999}"
STEPS="${STEPS:-20000}"
A="${D}v_s${K}"
source slurm/study_lib.sh
echo "== $(date) host=$(hostname) job=${SLURM_JOB_ID:-none} dataset=$D seed=$K alias=$A cpus=$W mu=$MU steps=$STEPS"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python tools/check_env.py > "logs/check_env_${SLURM_JOB_ID:-x}.txt" 2>&1 || { tail -20 "logs/check_env_${SLURM_JOB_ID:-x}.txt"; exit 3; }
# --legacy_target_in_fix: rebuild the exact split used for our results. Since 1 Oct the script's
# default rule leaves news with 37 fewer training rows; adult and default are identical either way.
python tools/make_val_split.py --dataname "$D" --seeds "$K" --legacy_target_in_fix
echo "== $(date) STAGE teacher";       stage_teacher
echo "== $(date) STAGE teacher curve"; stage_teacher_curve
python tools/test_distill.py --dataname "$A" 2>&1 | grep -v -i warn
echo "== $(date) STAGE student";       run_student study --steps "$STEPS" --N 18 --mu "$MU" --lr 1e-4
echo "== $(date) STAGE C4";            stage_c4 study
date > "results/${A}_STUDY_DONE"
echo "== $(date) STUDY_DONE $A"
