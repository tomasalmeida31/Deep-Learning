#!/bin/bash
#SBATCH --job-name=rescore
#SBATCH --partition=MGPU-TC2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=30G
#SBATCH --time=06:00:00
#SBATCH --output=logs/rescore_%j.out
# Re-scoring after the external review (1 Oct 2026): measured F_6, seeded C2ST, full-record DCR, common-metric
# pairs and tail diagnostics for the given "dataset:seed" cells. No training. Skip-if-done per cell
# (results/<alias>_RESCORE_DONE). Usage: sbatch slurm/rescore_one.sbatch adult:0 adult:1 adult:2
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
python tools/test_samplers.py > "logs/test_samplers_rescore_${SLURM_JOB_ID:-x}.txt" 2>&1 && echo "test_samplers: green" || { echo "FAIL: test_samplers red"; exit 4; }
for c in $CELLS; do
  D="${c%%:*}"; K="${c##*:}"; A="${D}v_s${K}"
  echo "== $(date) CELL $A"; stage_rescore
done
echo "== $(date) RESCORE_ALL_DONE"
