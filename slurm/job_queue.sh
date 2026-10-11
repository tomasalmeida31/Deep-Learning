#!/bin/bash
# Generic sequential queue under the 2-submitted-jobs QoS: each entry is "<sbatch script>|<args>|<done-marker glob count>",
# e.g. "slurm/fast_one.sbatch|adult|results/adultv_s[012]_FAST_DONE:3". Entries are submitted in order, each afterany on the
# previous; when the queue drains, entries whose markers are incomplete are resubmitted (at most 2 passes).
# Run on the login node:  setsid nohup bash slurm/job_queue.sh "<entry>" ... > logs/job_queue_<name>.log 2>&1 < /dev/null &
set -u
cd "$HOME/tabsyn"
PREV=""
njobs () { squeue -u "$USER" -h 2>/dev/null | wc -l; }
inq () { squeue -h -j "$1" 2>/dev/null | grep -q .; }
done_ok () { local g="${1%:*}" n="${1##*:}"; [ "$(ls $g 2>/dev/null | wc -l)" -ge "$n" ]; }
submit () {                      # submit <script> <args>
  local dep=""; while [ "$(njobs)" -ge 2 ]; do sleep 120; done
  if [ -n "$PREV" ] && inq "$PREV"; then dep="--dependency=afterany:$PREV"; fi
  local id; id=$(sbatch --parsable $dep "$1" $2 2>&1 | tail -1 | cut -d';' -f1)
  if [[ "$id" =~ ^[0-9]+$ ]]; then PREV="$id"; echo "$(date) submitted $1 $2 -> $id ($dep)"; else echo "$(date) sbatch FAILED for $1 $2: $id"; sleep 300; submit "$1" "$2"; fi
}
ENTRIES=("$@")
for e in "${ENTRIES[@]}"; do IFS='|' read -r sc ar mk <<< "$e"; done_ok "$mk" && { echo "$(date) already done: $sc $ar"; continue; }; submit "$sc" "$ar"; done
for pass in 1 2; do
  while [ "$(njobs)" -gt 0 ]; do sleep 300; done
  missing=0
  for e in "${ENTRIES[@]}"; do IFS='|' read -r sc ar mk <<< "$e"; done_ok "$mk" || { missing=1; echo "$(date) retry pass $pass: $sc $ar"; submit "$sc" "$ar"; }; done
  [ "$missing" -eq 0 ] && { echo "$(date) ALL ENTRIES DONE"; exit 0; }
done
while [ "$(njobs)" -gt 0 ]; do sleep 300; done
echo "$(date) queue finished; incomplete:"; for e in "${ENTRIES[@]}"; do IFS='|' read -r sc ar mk <<< "$e"; done_ok "$mk" || echo "  $sc $ar"; done
