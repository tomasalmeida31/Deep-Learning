#!/bin/bash
#SBATCH --job-name=bench
#SBATCH --partition=MGPU-TC2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=30G
#SBATCH --time=02:00:00
#SBATCH --output=logs/bench_%j.out
# Follow-up (11 Oct 2026): wall-clock generation benchmark (tools/bench_speed.py) for the given "dataset:seed"
# cells (default: all 9): one-step student vs the tuned teacher samplers and E99, same GPU, same table size,
# then the summary over cells. No training. Writes results/<alias>_bench_speed.json, results/bench_speed_cells.csv,
# results/bench_speed_summary.csv and results/bench_speed_report.txt.
# Usage: sbatch slurm/bench_speed.sbatch [adult:0 adult:1 ...]
set -eo pipefail
CELLS="${*:-adult:0 adult:1 adult:2 default:0 default:1 default:2 news:0 news:1 news:2}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate tabsyn
set -u
cd "$HOME/tabsyn"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1        # the same thread settings as every sampling job in the study
echo "== $(date) host=$(hostname) job=${SLURM_JOB_ID:-none} cells=$CELLS"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
for c in $CELLS; do D="${c%%:*}"; K="${c##*:}"; python tools/bench_speed.py --dataname "${D}v_s${K}" --seed "$K" --repeats 5; done
python tools/bench_speed.py --report | tee results/bench_speed_report.txt
echo "== $(date) BENCH_DONE"
