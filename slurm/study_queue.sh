#!/bin/bash
# Feed the 9 study cells to Slurm under the 2-submitted-jobs-per-user QoS limit, in order, each chained
# afterany on the previous one; then up to 2 retry passes for any cell without its STUDY_DONE marker
# (cells are skip-if-done, so a retry resumes at the stage level).
# Run on the login node:  setsid nohup bash slurm/study_queue.sh <last_submitted_jobid> <cell> ... > logs/study_queue.log 2>&1 &
#   cells are "dataset:seed", e.g. adult:0 default:0 ...
set -u
cd "$HOME/tabsyn"
PREV="$1"; shift
ALL=(adult:1 adult:2 adult:0 default:0 default:1 default:2 news:0 news:1 news:2)
njobs () { squeue -u "$USER" -h 2>/dev/null | wc -l; }
inq () { squeue -h -j "$1" 2>/dev/null | grep -q .; }
submit () {                      # submit <dataset> <seed>; sets PREV
  local dep=""; while [ "$(njobs)" -ge 2 ]; do sleep 120; done
  if [ -n "$PREV" ] && inq "$PREV"; then dep="--dependency=afterany:$PREV"; fi
  local id; id=$(sbatch --parsable $dep slurm/study_one.sbatch "$1" "$2" 2>&1 | tail -1 | cut -d';' -f1)
  if [[ "$id" =~ ^[0-9]+$ ]]; then PREV="$id"; echo "$(date) submitted $1 $2 -> $id ($dep)"; else echo "$(date) sbatch FAILED for $1 $2: $id"; sleep 300; submit "$1" "$2"; fi
}
for c in "$@"; do submit "${c%%:*}" "${c##*:}"; done
for pass in 1 2; do
  while [ "$(njobs)" -gt 0 ]; do sleep 300; done
  missing=0
  for c in "${ALL[@]}"; do d="${c%%:*}"; k="${c##*:}"
    if [ ! -f "results/${d}v_s${k}_STUDY_DONE" ]; then missing=1; echo "$(date) retry pass $pass: $d $k"; submit "$d" "$k"; fi
  done
  [ "$missing" -eq 0 ] && { echo "$(date) ALL 9 CELLS DONE"; exit 0; }
done
while [ "$(njobs)" -gt 0 ]; do sleep 300; done
echo "$(date) queue finished; cells still missing STUDY_DONE:"; for c in "${ALL[@]}"; do d="${c%%:*}"; k="${c##*:}"; [ -f "results/${d}v_s${k}_STUDY_DONE" ] || echo "  $d $k"; done
