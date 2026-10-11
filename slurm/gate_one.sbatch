#!/bin/bash
#SBATCH --job-name=gate
#SBATCH --partition=MGPU-TC2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=30G
#SBATCH --time=06:00:00
#SBATCH --output=logs/gate_%j.out
# Week-1 gate for one table, training seed 0: builds the leak-free split, trains the teacher
# (VAE + diffusion model), samples it at every setting of the `gate` grid (770 scored samples:
# the authors' Heun schedule at 1-99 NFE with churn 0 and 1, Euler, and a noise-schedule sweep),
# scores them against train', computes the perfect-generator floor and applies the go/no-go rule.
# It produced the seed-0 teachers and teacher curves used in the results. To reproduce them
# exactly, run this before slurm/study_one.sbatch for seed 0; study_one then skips the teacher
# stages. Seeds 1-2 go straight to study_one, whose teacher curve leaves out the sweep.
# Usage: sbatch slurm/gate_one.sbatch <dataset>     e.g. adult | default | news
# Every stage is skip-if-done, so re-submitting after a kill resumes at the stage level.
set -eo pipefail
D="${1:?dataset name required}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate tabsyn
set -u
cd "$HOME/tabsyn"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
W="${SLURM_CPUS_PER_TASK:-4}"
A="${D}v_s0"
echo "== $(date) host=$(hostname) job=${SLURM_JOB_ID:-none} dataset=$D alias=$A cpus=$W"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python tools/check_env.py > "logs/check_env_${SLURM_JOB_ID:-x}.txt" 2>&1 || { tail -20 "logs/check_env_${SLURM_JOB_ID:-x}.txt"; exit 3; }
# --legacy_target_in_fix: rebuild the exact split used for our results. Since 1 Oct the script's
# default rule leaves news with 37 fewer training rows; adult and default are identical either way.
python tools/make_val_split.py --dataname "$D" --seeds 0 --legacy_target_in_fix
python tools/train_seeded.py --stage vae       --dataname "$A" --seed 0 --gpu 0
python tools/train_seeded.py --stage diffusion --dataname "$A" --seed 0 --gpu 0
python tools/sample_grid.py --dataname "$A" --grid gate --gpu 0
python tools/eval_fast.py --dataname "$A" --real real --glob "synthetic/$A/grid/*.csv" --out "results/${A}_gate.jsonl" --workers "$W" --delete_csv
python tools/floor_boot.py --dataname "$A" --n_boot 5 --workers "$W"
N=$(wc -l < "results/${A}_gate.jsonl")
echo "gate rows for $A: $N"
[ "$N" -ge 700 ] || { echo "FAIL: expected >= 700 evaluated configs, got $N"; exit 4; }
python tools/gate_decide.py --datanames "$A" || echo "(single-dataset verdict is informational; run gate_decide on all three at the end)"
date > "results/${A}_GATE_DONE"
echo "== $(date) GATE_DONE $A"
