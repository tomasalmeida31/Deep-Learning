#!/bin/bash
#SBATCH --job-name=ablate
#SBATCH --partition=MGPU-TC2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=30G
#SBATCH --time=06:00:00
#SBATCH --output=logs/ablate_%j.out
# Ablation plan E2-E4 for one (dataset, training seed) cell: arms from {ct, ict, N8, N40, long, ctlong}.
# ctlong (follow-up, 11 Oct 2026) continues the 20k-step ct arm to 40k steps, so ct must be done first.
# Usage: sbatch slurm/ablate_one.sbatch <dataset> <seed> <arm> [arm ...]     e.g. adult 0 ct ict N8 N40 long
# Every arm is skip-if-done (results/<alias>_student_<arm>_s<seed>_DONE); re-submit the same command after a kill.
set -eo pipefail
D="${1:?dataset name required}"; K="${2:?training seed required}"; shift 2; ARMS="${*:?at least one arm}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate tabsyn
set -u
cd "$HOME/tabsyn"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
W="${SLURM_CPUS_PER_TASK:-4}"
A="${D}v_s${K}"
source slurm/study_lib.sh
echo "== $(date) host=$(hostname) job=${SLURM_JOB_ID:-none} alias=$A arms=$ARMS cpus=$W"
[ -f "tabsyn/ckpt/$A/DONE" ] && [ -f "results/${A}_student_study_s${K}_DONE" ] || { echo "FAIL: teacher or study student missing for $A"; exit 3; }
python tools/test_distill.py --dataname "$A" 2>&1 | grep -v -i warn
for arm in $ARMS; do echo "== $(date) ARM $arm"; run_arm "$arm"; done
date > "results/${A}_ABLATE_$(echo $ARMS | tr ' ' '_')_DONE"
echo "== $(date) ABLATE_DONE $A $ARMS"
