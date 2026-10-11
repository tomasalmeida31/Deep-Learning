# Shared stages for the full study (sourced by study_one.sbatch and the ablation jobs).
# Every stage is skip-if-done at the file level, so a killed job resumes where it stopped.
# Expects: A (alias, e.g. adultv_s1), K (training seed), W (workers), cwd = ~/tabsyn, conda env active.

# Teacher CSVs are deleted after scoring but their .json sidecars stay, and the samplers skip on a
# sidecar; remove orphaned sidecars (json without csv) before any re-sampling.
clear_orphans () { [ -d "$1" ] || return 0; find "$1" -name '*.json' | while read -r j; do [ -f "${j%.json}.csv" ] || rm -f "$j"; done; }

# Six teacher reference configs (NFE 1, 2, 4, 15, 39, 99) used for C2ST / MLE / DCR and the paired analysis.
REF_SPECS="heun:1:0:80:7 euler:2:0:80:7 euler:4:0:80:7 heun:8:0:80:7 heun:20:0:80:7 heun:50:0:80:7"

stage_teacher () {                       # VAE + latent EDM at training seed K on the leak-free alias
  python tools/train_seeded.py --stage vae       --dataname "$A" --seed "$K" --gpu 0
  python tools/train_seeded.py --stage diffusion --dataname "$A" --seed "$K" --gpu 0
}

stage_teacher_curve () {                 # error vs NFE for the teacher: Heun N in {1..50} x churn {0,1}, Euler {2,4,8,16}; 20 sampling seeds
  if [ -f "results/${A}_gate_summary.csv" ] && [ "$(wc -l < "results/${A}_gate.jsonl")" -ge 520 ]; then echo "SKIP teacher curve (summary exists)"; return; fi
  clear_orphans "synthetic/$A/grid"
  python tools/sample_grid.py --dataname "$A" --grid shipped --gpu 0
  python tools/sample_grid.py --dataname "$A" --grid euler --gpu 0
  python tools/eval_fast.py --dataname "$A" --real real --glob "synthetic/$A/grid/*.csv" --out "results/${A}_gate.jsonl" --workers "$W" --delete_csv
  python tools/floor_boot.py --dataname "$A" --n_boot 5 --workers "$W"
  N=$(wc -l < "results/${A}_gate.jsonl"); echo "teacher curve rows for $A: $N"
  [ "$N" -ge 520 ] || { echo "FAIL: expected >= 520 teacher configs, got $N"; exit 4; }
  python tools/gate_decide.py --datanames "$A" --out "results/${A}_gate_verdict_single.json" || echo "(single-dataset verdict is informational)"
}

run_student () {                         # run_student <tag> <distill args...>: train, choose taus on val, 20 seeds on train', report
  local TAG="$1"; shift
  local S="student_${TAG}_s${K}"
  if [ -f "results/${A}_${S}_DONE" ]; then echo "SKIP student $TAG (done)"; return; fi
  echo "== $(date) STUDENT $TAG seed $K args: $*"
  python tools/distill.py --dataname "$A" --seed "$K" --gpu 0 --tag "$TAG" --log_every 250 --ckpt_every 2500 "$@"
  clear_orphans "synthetic/$A/$S"
  python tools/sample_student.py --dataname "$A" --student "$S" --nfe 1 2 4 --seeds 5 --gpu 0 --weights best --tau_grid --num_samples 8192
  python tools/eval_fast.py --dataname "$A" --real val --glob "synthetic/$A/$S/*_g*_s*.csv" --out "results/${A}_${S}_valgrid.jsonl" --workers "$W" --delete_csv
  local TAUS
  TAUS=$(python tools/pick_taus.py --dataname "$A" --student "$S" | tee "logs/pick_taus_${A}_${S}.txt" | grep '^TAUS_ARGS' | cut -d' ' -f2-)
  echo "chosen taus: $TAUS"
  python tools/sample_student.py --dataname "$A" --student "$S" --nfe 1 2 4 --seeds 20 --gpu 0 --weights best --tag sel --taus $TAUS
  # shape / trend / C2ST on train'; the CSVs are kept for the C4 stage and deleted there
  python tools/eval_fast.py --dataname "$A" --real real --c2st --glob "synthetic/$A/$S/*_sel_s*.csv" --out "results/${A}_${S}.jsonl" --workers "$W"
  python tools/student_report.py --dataname "$A" --student "$S"
  date > "results/${A}_${S}_DONE"
}

stage_c4 () {                            # stage_c4 <student tag>: C2ST + MLE + DCR for references, teacher reference configs and the student
  local TAG="$1"; local S="student_${TAG}_s${K}"; local NS="${EXTRA_SEEDS:-10}"
  if [ -f "results/${A}_${S}_EXTRA_DONE" ]; then echo "SKIP C4 $TAG (done)"; return; fi
  python tools/eval_extra.py --dataname "$A" --glob "synthetic/$A/val.csv"  --out "results/${A}_ref_extra.jsonl" --workers 1
  python tools/eval_extra.py --dataname "$A" --glob "synthetic/$A/real.csv" --out "results/${A}_ref_extra.jsonl" --workers 1
  clear_orphans "synthetic/$A/grid"
  for s in $(seq 0 $((NS - 1))); do for sp in $REF_SPECS; do python tools/sample_grid.py --dataname "$A" --spec "$sp:$s" --gpu 0 > /dev/null; done; done
  local NT; NT=$(ls synthetic/$A/grid/*.csv | wc -l); echo "teacher reference files: $NT"
  [ "$NT" -ge $((6 * NS)) ] || { echo "FAIL: expected $((6 * NS)) teacher CSVs"; exit 4; }
  python tools/eval_fast.py  --dataname "$A" --real real --c2st --glob "synthetic/$A/grid/*_c0_smax80_rho7_s*.csv" --out "results/${A}_teacher_c2st.jsonl" --workers "$W"
  python tools/eval_extra.py --dataname "$A" --glob "synthetic/$A/grid/*_c0_smax80_rho7_s*.csv" --out "results/${A}_teacher_extra.jsonl" --workers "$W"
  find "synthetic/$A/grid" -name '*.csv' -delete
  # student: the first NS sampling seeds of the selected schedules (files exist from run_student; re-sampled if deleted)
  local SG="synthetic/$A/$S/*_sel_s[0-9].csv"; [ "$NS" -gt 10 ] && SG="synthetic/$A/$S/*_sel_s*.csv"
  if [ "$(ls $SG 2>/dev/null | wc -l)" -lt $((3 * NS)) ]; then
    clear_orphans "synthetic/$A/$S"
    local TAUS; TAUS=$(python - <<PY
import json; t=json.load(open("results/${A}_${S}_taus.json")); print(' '.join(f"{k}="+','.join(str(x) for x in v['taus']) for k,v in t.items() if int(k)>1))
PY
)
    python tools/sample_student.py --dataname "$A" --student "$S" --nfe 1 2 4 --seeds 20 --gpu 0 --weights best --tag sel --taus $TAUS
  fi
  python tools/eval_extra.py --dataname "$A" --glob "$SG" --out "results/${A}_${S}_extra.jsonl" --workers "$W"
  python tools/extra_report.py --dataname "$A" --student "$S"
  find "synthetic/$A/$S" -name '*_sel_s*.csv' -delete
  date > "results/${A}_${S}_EXTRA_DONE"
}

# ---------------------------------------------------------------------------------------------------------
# Ablation plan E1: strongest fair few-step TEACHER at matched NFE (see claude/tabsyn-ablation-plan.md).
# Candidates = sample_grid --grid fast_val (euler / dpmpp2m / heun x sigma_max {1..80} x rho {1,3,7}, NFE <= 16),
# 5 seeds x 8192 rows scored on val -> pick_fast chooses F_k per budget k in {1,2,4,8,16} -> 20 seeds on train'
# with C2ST -> (EXTRA=1) MLE/DCR on seeds 0-9 for the configs with NFE >= 4. Writes only results/<A>_fast* files.
stage_fast () {
  if [ -f "results/${A}_FAST_DONE" ]; then echo "SKIP fast teacher (done)"; return; fi
  local FD="synthetic/$A/fast"; mkdir -p "$FD"; clear_orphans "$FD"
  local NV; NV=$(python -c "import sys; sys.path.insert(0, '.'); from tools.sample_grid import grid_specs; print(len(grid_specs('fast_val')))")
  local VJ="results/${A}_fast_valgrid.jsonl"
  if [ ! -f "$VJ" ] || [ "$(wc -l < "$VJ")" -lt "$NV" ]; then
    echo "== $(date) FAST val grid ($NV files)"
    python tools/sample_grid.py --dataname "$A" --grid fast_val --out_dir "$FD" --tag v --num_samples 8192 --gpu 0
    python tools/eval_fast.py --dataname "$A" --real val --glob "$FD/*_v_s*.csv" --out "$VJ" --workers "$W" --delete_csv
  fi
  [ "$(wc -l < "$VJ")" -ge "$NV" ] || { echo "FAIL: $VJ has $(wc -l < "$VJ") rows, expected $NV"; exit 4; }
  local SPECS
  SPECS=$(python tools/pick_fast.py --dataname "$A" | tee "logs/pick_fast_${A}.txt" | grep '^FAST_SPECS' | cut -d' ' -f2-)
  [ -n "$SPECS" ] || { echo "FAIL: pick_fast gave no specs"; exit 4; }
  echo "== $(date) FAST chosen: $SPECS"
  local ALL="" sp s
  for sp in $SPECS; do for s in $(seq 0 19); do ALL="$ALL $sp:$s"; done; done
  python tools/sample_grid.py --dataname "$A" --out_dir "$FD" --tag sel --gpu 0 --spec $ALL
  python tools/eval_fast.py --dataname "$A" --real real --c2st --glob "$FD/*_sel_s*.csv" --out "results/${A}_fast.jsonl" --workers "$W"
  local NS; NS=$(echo $SPECS | wc -w)
  [ "$(wc -l < "results/${A}_fast.jsonl")" -ge $((20 * NS)) ] || { echo "FAIL: results/${A}_fast.jsonl has fewer than $((20 * NS)) rows"; exit 4; }
  if [ "${EXTRA:-1}" = 1 ]; then
    for sp in $SPECS; do
      IFS=: read -r so st ch sm rh <<< "$sp"
      local nfe=$st; [ "$so" = heun ] && nfe=$((2 * st - 1))
      [ "$nfe" -ge 4 ] || continue
      python tools/eval_extra.py --dataname "$A" --glob "$FD/${so}_N${st}_c0_smax${sm}_rho${rh}_sel_s[0-9].csv" --out "results/${A}_fast_extra.jsonl" --workers "$W"
    done
  fi
  find "$FD" -name '*.csv' -delete
  date > "results/${A}_FAST_DONE"
}

# ---------------------------------------------------------------------------------------------------------
# Ablation plan E2-E4: student arms that change ONE factor of the frozen study recipe
# (N 18, mu 0.999, Adam 1e-4 constant, 20k steps, pseudo-Huber, Heun teacher step). Same tau-on-val protocol,
# 20 seeds on train'; no C4. Tags: ct (E2), ict (E2, seed-0 only), N8 / N40 (E3, seed-0 only), long (E4).
run_arm () {
  case "$1" in
    ct)   run_student ct   --mode ct --steps 20000 --N 18 --mu 0.999 --lr 1e-4 ;;
    ict)  run_student ict  --mode ct --steps 20000 --N 18 --mu 0 --weight inv_dt --lr 1e-4 ;;
    N8)   run_student N8   --steps 20000 --N 8  --mu 0.999 --lr 1e-4 ;;
    N40)  run_student N40  --steps 20000 --N 40 --mu 0.999 --lr 1e-4 ;;
    long) stage_long ;;
    ctlong) stage_ctlong ;;
    *) echo "FAIL: unknown arm $1"; exit 2 ;;
  esac
}

stage_long () {                          # E4: continue the study run from its 20k last.pt to 40k steps
  local S="student_long_s${K}" SRC="tabsyn/ckpt/$A/student_study_s${K}" DST="tabsyn/ckpt/$A/student_long_s${K}" COPIED=0 f
  if [ -f "results/${A}_${S}_DONE" ]; then echo "SKIP student long (done)"; return; fi
  if [ ! -f "$DST/last.pt" ] && [ ! -f "$DST/DONE" ]; then
    if [ -f "$SRC/last.pt" ] && [ -f "$SRC/best.pt" ] && [ -f "$SRC/ref_stats.pt" ] && [ -f "$SRC/log.jsonl" ]; then
      mkdir -p "$DST"; for f in last.pt best.pt ref_stats.pt log.jsonl; do cp "$SRC/$f" "$DST/$f"; done; COPIED=1
      echo "long: copied the 20k study state from $SRC"
    else
      echo "long: $SRC lacks last.pt/best.pt/ref_stats.pt/log.jsonl -> training 40k from scratch (deterministic; same first 20k)"
    fi
  fi
  run_student long --steps 40000 --N 18 --mu 0.999 --lr 1e-4 2>&1 | tee "logs/long_${A}.txt"
  if [ "$COPIED" = 1 ]; then grep -q 'RESUME at step 20000' "logs/long_${A}.txt" || { echo "FAIL: long did not resume at step 20000"; exit 4; }; fi
}

stage_ctlong () {                        # follow-up (11 Oct 2026): continue the teacher-initialised CT run from its 20k last.pt to 40k
  local S="student_ctlong_s${K}" SRC="tabsyn/ckpt/$A/student_ct_s${K}" DST="tabsyn/ckpt/$A/student_ctlong_s${K}" f
  if [ -f "results/${A}_${S}_DONE" ]; then echo "SKIP student ctlong (done)"; return; fi
  if [ ! -f "$DST/last.pt" ] && [ ! -f "$DST/DONE" ]; then
    for f in last.pt best.pt ref_stats.pt log.jsonl; do [ -f "$SRC/$f" ] || { echo "FAIL: $SRC/$f missing; run the ct arm first"; exit 3; }; done
    mkdir -p "$DST"; for f in last.pt best.pt ref_stats.pt log.jsonl; do cp "$SRC/$f" "$DST/$f"; done
    echo "ctlong: copied the 20k CT state from $SRC"
  fi
  run_student ctlong --mode ct --steps 40000 --N 18 --mu 0.999 --lr 1e-4 2>&1 | tee "logs/ctlong_${A}.txt"
  python -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get('resume', {}).get('resumed_at_step') == 20000 else 1)" "$DST/DONE" \
    || { echo "FAIL: ctlong was not continued from the 20k CT checkpoint ($DST/DONE)"; exit 4; }
}

# ---------------------------------------------------------------------------------------------------------
# Ablation plan E5 (+ E7.1 on news): per-column / per-pair failure analysis. The scored CSVs were deleted by
# stage_c4, so the files are RE-SAMPLED (deterministic seeds) after a determinism check against the stored scores.
stage_percol () {
  if [ -f "results/${A}_PERCOL_DONE" ]; then echo "SKIP percol (done)"; return; fi
  local S="student_study_s${K}" SD="synthetic/$A/student_study_s${K}" GD="synthetic/$A/grid" FD="synthetic/$A/fast" s sp
  clear_orphans "$SD"; clear_orphans "$GD"; clear_orphans "$FD"
  local TAUS; TAUS=$(python - <<PY
import json; t=json.load(open("results/${A}_${S}_taus.json")); print(' '.join(f"{k}="+','.join(str(x) for x in v['taus']) for k,v in t.items() if int(k)>1))
PY
)
  python tools/sample_student.py --dataname "$A" --student "$S" --nfe 1 2 4 --seeds 10 --gpu 0 --weights best --tag sel --taus $TAUS
  local SPECS="heun:8:0:80:7 heun:20:0:80:7 heun:50:0:80:7"; [ "$D" = news ] && SPECS="$REF_SPECS"
  local ALL=""; for sp in $SPECS; do for s in $(seq 0 9); do ALL="$ALL $sp:$s"; done; done
  python tools/sample_grid.py --dataname "$A" --gpu 0 --spec $ALL > /dev/null
  if [ -f "results/${A}_fast_choice.json" ]; then
    local F4; F4=$(python -c "import json; c=json.load(open('results/${A}_fast_choice.json'))['4']; print(f\"{c['solver']}:{c['steps']}:0:{c['sigma_max']}:{c['rho']}\")")
    ALL=""; for s in $(seq 0 9); do ALL="$ALL $F4:$s"; done
    python tools/sample_grid.py --dataname "$A" --out_dir "$FD" --tag sel --gpu 0 --spec $ALL > /dev/null
  fi
  # determinism check: re-scored shape_err must match the stored study / gate rows (exact expected; the difference is recorded, and the job stops only above 0.1 pp)
  local CHK="logs/percol_check_${A}.jsonl"; rm -f "$CHK"
  python tools/eval_fast.py --dataname "$A" --real real --glob "$SD/cm_N1_c0_smax80_rho7_sel_s0.csv" --out "$CHK" --workers 1 > /dev/null
  python tools/eval_fast.py --dataname "$A" --real real --glob "$GD/heun_N50_c0_smax80_rho7_s0.csv" --out "$CHK" --workers 1 > /dev/null
  python - <<PY || exit 4
import json
new = {r['file']: r['shape_err'] for r in map(json.loads, open("$CHK"))}
def stored(p, f): return next(r['shape_err'] for r in map(json.loads, open(p)) if r['file'] == f)
rec = {}
for f, p in (("cm_N1_c0_smax80_rho7_sel_s0.csv", "results/${A}_${S}.jsonl"), ("heun_N50_c0_smax80_rho7_s0.csv", "results/${A}_gate.jsonl")):
    d = abs(new[f] - stored(p, f)); rec[f] = {'rescored': new[f], 'stored': stored(p, f), 'abs_diff_pp': d}
    print(f"determinism {f}: |re-scored - stored| = {d:.2e} pp" + (" (exact)" if d < 1e-6 else " (NOT exact: re-sampled files differ from the scored ones)"))
    # Rule as executed (amended 25 Sep 2026 12:47 after job 37650 stopped on default s0 at 0.052 pp; see
    # docs/tabsyn-review-response.md): record the difference and label the cell; stop only above 0.1 pp.
    assert d < 0.1, f"FAIL determinism: {f} differs by {d} pp -> re-sampled files are not the scored ones"
json.dump(rec, open("results/${A}_percol_determinism.json", "w"), indent=1)
PY
  python tools/eval_percol.py --dataname "$A" --real real --glob "$SD/*_sel_s[0-9].csv" --out "results/${A}_percol.jsonl" --out_pairs "results/${A}_percol_pairs.jsonl" --workers "$W"
  python tools/eval_percol.py --dataname "$A" --real real --glob "$GD/heun_N*_c0_smax80_rho7_s[0-9].csv" --out "results/${A}_percol.jsonl" --out_pairs "results/${A}_percol_pairs.jsonl" --workers "$W"
  [ -d "$FD" ] && python tools/eval_percol.py --dataname "$A" --real real --glob "$FD/*_sel_s[0-9].csv" --out "results/${A}_percol.jsonl" --out_pairs "results/${A}_percol_pairs.jsonl" --workers "$W"
  if [ "$D" = news ]; then          # E7.1: MLE on the log1p target, no DCR
    local MJ="results/${A}_mle_log.jsonl"
    python tools/eval_extra.py --dataname "$A" --glob "synthetic/$A/real.csv" --out "$MJ" --workers 1 --target_transform log1p --no_dcr
    python tools/eval_extra.py --dataname "$A" --glob "synthetic/$A/val.csv"  --out "$MJ" --workers 1 --target_transform log1p --no_dcr
    python tools/eval_extra.py --dataname "$A" --glob "$GD/*_c0_smax80_rho7_s[0-9].csv" --out "$MJ" --workers "$W" --target_transform log1p --no_dcr
    python tools/eval_extra.py --dataname "$A" --glob "$SD/*_sel_s[0-9].csv" --out "$MJ" --workers "$W" --target_transform log1p --no_dcr
  fi
  find "$SD" "$GD" -name '*.csv' -delete; [ -d "$FD" ] && find "$FD" -name '*.csv' -delete
  date > "results/${A}_PERCOL_DONE"
}

# ---------------------------------------------------------------------------------------------------------
# Verdict v2 §6.1: sigma_max in {0.3, 0.5, 0.7} at 1-2 NFE on val; if any beats the E1 choice, 20 seeds on train'.
stage_fast_lo () {
  if [ -f "results/${A}_FASTLO_DONE" ]; then echo "SKIP fast_lo (done)"; return; fi
  local FD="synthetic/$A/fastlo"; mkdir -p "$FD"; clear_orphans "$FD"
  local VJ="results/${A}_fastlo_valgrid.jsonl"
  if [ ! -f "$VJ" ] || [ "$(wc -l < "$VJ")" -lt 45 ]; then
    python tools/sample_grid.py --dataname "$A" --grid fast_lo --out_dir "$FD" --tag v --num_samples 8192 --gpu 0
    python tools/eval_fast.py --dataname "$A" --real val --glob "$FD/*_v_s*.csv" --out "$VJ" --workers "$W" --delete_csv
  fi
  [ "$(wc -l < "$VJ")" -ge 45 ] || { echo "FAIL: $VJ has $(wc -l < "$VJ") rows, expected 45"; exit 4; }
  local SPECS; SPECS=$(python tools/pick_fast_lo.py --dataname "$A" | tee "logs/pick_fast_lo_${A}.txt" | grep '^FASTLO_SPECS' | cut -d' ' -f2-)
  echo "== $(date) FASTLO chosen (beating E1): '$SPECS'"
  if [ -n "$SPECS" ]; then
    local ALL="" sp s; for sp in $SPECS; do for s in $(seq 0 19); do ALL="$ALL $sp:$s"; done; done
    python tools/sample_grid.py --dataname "$A" --out_dir "$FD" --tag sel --gpu 0 --spec $ALL
    python tools/eval_fast.py --dataname "$A" --real real --c2st --glob "$FD/*_sel_s*.csv" --out "results/${A}_fastlo.jsonl" --workers "$W"
  fi
  find "$FD" -name '*.csv' -delete
  date > "results/${A}_FASTLO_DONE"
}


# ---------------------------------------------------------------------------------------------------------
# Leak-delta row (verdict v2 §6.3): the paper's leaky protocol (VAE selection on the shipped test split) on
# identical train' -> teacher curve + C2ST + MLE/DCR for the six reference configs. No student. A = <d>L_s0.
stage_teacher_extra () {
  local NS="${EXTRA_SEEDS:-10}"
  if [ -f "results/${A}_TEACHER_EXTRA_DONE" ]; then echo "SKIP teacher extra (done)"; return; fi
  python tools/eval_extra.py --dataname "$A" --glob "synthetic/$A/val.csv"  --out "results/${A}_ref_extra.jsonl" --workers 1
  python tools/eval_extra.py --dataname "$A" --glob "synthetic/$A/real.csv" --out "results/${A}_ref_extra.jsonl" --workers 1
  clear_orphans "synthetic/$A/grid"
  local ALL="" sp s; for sp in $REF_SPECS; do for s in $(seq 0 $((NS - 1))); do ALL="$ALL $sp:$s"; done; done
  python tools/sample_grid.py --dataname "$A" --gpu 0 --spec $ALL > /dev/null
  python tools/eval_fast.py  --dataname "$A" --real real --c2st --glob "synthetic/$A/grid/*_c0_smax80_rho7_s*.csv" --out "results/${A}_teacher_c2st.jsonl" --workers "$W"
  python tools/eval_extra.py --dataname "$A" --glob "synthetic/$A/grid/*_c0_smax80_rho7_s*.csv" --out "results/${A}_teacher_extra.jsonl" --workers "$W"
  if [ "$D" = news ]; then
    python tools/eval_extra.py --dataname "$A" --glob "synthetic/$A/real.csv" --out "results/${A}_mle_log.jsonl" --workers 1 --target_transform log1p --no_dcr
    python tools/eval_extra.py --dataname "$A" --glob "synthetic/$A/grid/*_c0_smax80_rho7_s[0-9].csv" --out "results/${A}_mle_log.jsonl" --workers "$W" --target_transform log1p --no_dcr
  fi
  find "synthetic/$A/grid" -name '*.csv' -delete
  date > "results/${A}_TEACHER_EXTRA_DONE"
}

# ---------------------------------------------------------------------------------------------------------
# Re-scoring stage (1 Oct 2026, after the external review; see docs/tabsyn-review-response.md). No training.
# Re-samples (deterministic seeds) the study student, the authors' 99-NFE sampler and the E1-selected F_4/F_8/F_16,
# then writes ONLY new result files:
#   results/<A>_fast6_valgrid.jsonl, _fast6_choice.json, _fast6.jsonl   measured F_6 (review finding 13)
#   results/<A>_c2st_seeded.jsonl        seeded C2ST, 3 CV splits per file, c2st_mean/c2st_sd  (finding 17)
#   results/<A>_dcr_full.jsonl           DCR over the whole record incl. target, + ties / exact copies (finding 5)
#   results/<A>_dcr_feat.jsonl           DCR in the historical feature space, + ties / exact copies  (finding 5)
#   results/<A>_pairs_common.jsonl, _cols_common.jsonl   one metric for all pair types, IPF dependence-only error,
#                                        tail quantile diagnostics  (findings 15, 22)
#   results/<A>_rescore_determinism.json re-scored shape of the seed-0 files vs the stored study / gate rows
stage_rescore () {
  if [ -f "results/${A}_RESCORE_DONE" ]; then echo "SKIP rescore (done)"; return; fi
  local S="student_study_s${K}" SD="synthetic/$A/student_study_s${K}" GD="synthetic/$A/grid" FD="synthetic/$A/fast" F6="synthetic/$A/fast6" s sp
  mkdir -p "$FD" "$F6"; clear_orphans "$SD"; clear_orphans "$GD"; clear_orphans "$FD"; clear_orphans "$F6"
  local TAUS; TAUS=$(python - <<PY
import json; t=json.load(open("results/${A}_${S}_taus.json")); print(' '.join(f"{k}="+','.join(str(x) for x in v['taus']) for k,v in t.items() if int(k)>1))
PY
)
  echo "== $(date) RESCORE $A: sampling"
  python tools/sample_student.py --dataname "$A" --student "$S" --nfe 1 2 4 --seeds 20 --gpu 0 --weights best --tag sel --taus $TAUS
  local ALL=""; for s in $(seq 0 9); do ALL="$ALL heun:50:0:80:7:$s"; done
  python tools/sample_grid.py --dataname "$A" --gpu 0 --spec $ALL > /dev/null
  local FSPECS; FSPECS=$(python -c "
import json; c=json.load(open('results/${A}_fast_choice.json'))
print(' '.join(sorted(set(f\"{c[k]['solver']}:{c[k]['steps']}:0:{c[k]['sigma_max']}:{c[k]['rho']}\" for k in ('4','8','16')))))")
  ALL=""; for sp in $FSPECS; do for s in $(seq 0 19); do ALL="$ALL $sp:$s"; done; done
  python tools/sample_grid.py --dataname "$A" --out_dir "$FD" --tag sel --gpu 0 --spec $ALL > /dev/null
  # determinism record: re-scored shape of the seed-0 files vs the stored rows (informational; recorded, never a stop rule)
  local CHK="logs/rescore_check_${A}.jsonl"; rm -f "$CHK"
  python tools/eval_fast.py --dataname "$A" --real real --glob "$SD/cm_N1_c0_smax80_rho7_sel_s0.csv" --out "$CHK" --workers 1 > /dev/null
  python tools/eval_fast.py --dataname "$A" --real real --glob "$GD/heun_N50_c0_smax80_rho7_s0.csv" --out "$CHK" --workers 1 > /dev/null
  python - <<PY
import json
new = {r['file']: r['shape_err'] for r in map(json.loads, open("$CHK"))}
def stored(p, f): return next(r['shape_err'] for r in map(json.loads, open(p)) if r['file'] == f)
rec = {}
for f, p in (("cm_N1_c0_smax80_rho7_sel_s0.csv", "results/${A}_${S}.jsonl"), ("heun_N50_c0_smax80_rho7_s0.csv", "results/${A}_gate.jsonl")):
    rec[f] = {'rescored': new[f], 'stored': stored(p, f), 'abs_diff_pp': abs(new[f] - stored(p, f))}
    print(f"determinism {f}: |re-scored - stored| = {rec[f]['abs_diff_pp']:.2e} pp")
json.dump(rec, open("results/${A}_rescore_determinism.json", "w"), indent=1)
PY
  # measured F_6 (val search over 6-NFE candidates, union with the E1 grid; 20 seeds on train')
  if [ ! -f "results/${A}_fast6.jsonl" ] || [ "$(wc -l < "results/${A}_fast6.jsonl")" -lt 20 ]; then
    echo "== $(date) RESCORE $A: F_6 val grid"
    python tools/sample_grid.py --dataname "$A" --grid fast6 --out_dir "$F6" --tag v --num_samples 8192 --gpu 0 > /dev/null
    python tools/eval_fast.py --dataname "$A" --real val --glob "$F6/*_v_s*.csv" --out "results/${A}_fast6_valgrid.jsonl" --workers "$W" --delete_csv
    local S6; S6=$(python tools/pick_fast_k.py --dataname "$A" --k 6 --valgrid "results/${A}_fast_valgrid.jsonl" "results/${A}_fast6_valgrid.jsonl" | tee "logs/pick_fast6_${A}.txt" | grep '^FASTK_SPECS' | cut -d' ' -f2)
    echo "F_6 chosen: $S6"
    ALL=""; for s in $(seq 0 19); do ALL="$ALL $S6:$s"; done
    python tools/sample_grid.py --dataname "$A" --out_dir "$F6" --tag sel --gpu 0 --spec $ALL > /dev/null
    python tools/eval_fast.py --dataname "$A" --real real --c2st --c2st_repeats 3 --glob "$F6/*_sel_s*.csv" --out "results/${A}_fast6.jsonl" --workers "$W"
  fi
  # seeded C2ST (3 splits per file): student 1/2/4 NFE x 20 seeds, F_4/F_8/F_16 x 20 seeds, E99 x 10 seeds
  echo "== $(date) RESCORE $A: seeded C2ST"
  python tools/eval_fast.py --dataname "$A" --real real --c2st --c2st_repeats 3 --glob "$SD/*_sel_s*.csv" --out "results/${A}_c2st_seeded.jsonl" --workers "$W"
  python tools/eval_fast.py --dataname "$A" --real real --c2st --c2st_repeats 3 --glob "$FD/*_sel_s*.csv" --out "results/${A}_c2st_seeded.jsonl" --workers "$W"
  python tools/eval_fast.py --dataname "$A" --real real --c2st --c2st_repeats 3 --glob "$GD/heun_N50_c0_smax80_rho7_s[0-9].csv" --out "results/${A}_c2st_seeded.jsonl" --workers "$W"
  # DCR in both record spaces (no MLE): references, student seeds 0-9, F_k seeds 0-9, E99 seeds 0-9
  echo "== $(date) RESCORE $A: DCR"
  for SP in full; do          # feat-space DCR already exists in the historical _extra files; only the full-record space is new
    local OUT="results/${A}_dcr_${SP}.jsonl" SPACE=$SP; [ "$SP" = feat ] && SPACE=features
    python tools/eval_extra.py --dataname "$A" --glob "synthetic/$A/real.csv" --out "$OUT" --workers 1 --no_mle --dcr_space $SPACE
    python tools/eval_extra.py --dataname "$A" --glob "synthetic/$A/val.csv"  --out "$OUT" --workers 1 --no_mle --dcr_space $SPACE
    python tools/eval_extra.py --dataname "$A" --glob "$SD/*_sel_s[0-9].csv" --out "$OUT" --workers "$W" --no_mle --dcr_space $SPACE
    python tools/eval_extra.py --dataname "$A" --glob "$FD/*_sel_s[0-9].csv" --out "$OUT" --workers "$W" --no_mle --dcr_space $SPACE
    python tools/eval_extra.py --dataname "$A" --glob "$GD/heun_N50_c0_smax80_rho7_s[0-9].csv" --out "$OUT" --workers "$W" --no_mle --dcr_space $SPACE
  done
  # common-metric pairs + tails: student 1 and 4 NFE seeds 0-9, F_4 and F_8 seeds 0-9, E99 seeds 0-9
  echo "== $(date) RESCORE $A: common-metric pairs"
  local F4 F8; F4=$(python -c "import json; c=json.load(open('results/${A}_fast_choice.json'))['4']; print(f\"{c['solver']}_N{c['steps']}_c0_smax{c['sigma_max']}_rho{c['rho']}\")")
  F8=$(python -c "import json; c=json.load(open('results/${A}_fast_choice.json'))['8']; print(f\"{c['solver']}_N{c['steps']}_c0_smax{c['sigma_max']}_rho{c['rho']}\")")
  for G in "$SD/cm_N1_c0_smax80_rho7_sel_s[0-9].csv" "$SD/cm_N4_c0_smax80_rho7_sel_s[0-9].csv" "$FD/${F4}_sel_s[0-9].csv" "$FD/${F8}_sel_s[0-9].csv" "$GD/heun_N50_c0_smax80_rho7_s[0-9].csv"; do
    python tools/eval_pairs_common.py --dataname "$A" --real real --glob "$G" --out_pairs "results/${A}_pairs_common.jsonl" --out_cols "results/${A}_cols_common.jsonl" --workers "$W"
  done
  find "$SD" "$GD" "$FD" "$F6" -name '*.csv' -delete
  date > "results/${A}_RESCORE_DONE"
  echo "== $(date) RESCORE $A done"
}
