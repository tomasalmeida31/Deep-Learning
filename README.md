# TabSyn with a one-step student — group handover

This repo holds everything from the TabSyn part of our AI6103 project: the code we wrote, the results it produced, and three short fact sheets for writing the report. Read this file first, then `docs/`.

**One rule for the report:** take every number from `docs/numbers.md` or from the results file it names. Don't take numbers from memory, chats or older drafts. If you need a number that isn't there, ask Tomás.

---

## 1. What we did, in plain words

**TabSyn** (Zhang et al., ICLR 2024) generates synthetic tables, i.e. fake rows that look like a real dataset. It works in two parts:

1. A **VAE** turns every row of the table into a short list of numbers, called a *latent*. It can also turn a latent back into a row.
2. A **diffusion model** learns to create new latents. It starts from random noise and removes the noise a little at a time. With the authors' settings this takes **99 passes** through the network for every batch of rows. We call each pass one **NFE** (network function evaluation).

**Our improvement:** we trained a second network, the **student**, to jump from noise to a finished latent in **one pass**. It learns by copying the diffusion model, which we call the **teacher**. The method is *consistency distillation* (Song et al., ICML 2023). It is not covered in the lectures.

**The fair comparison:** a one-pass student is only interesting if the teacher can't do as well when it is also given very few passes. So we searched settings for the teacher too: three step methods (solvers) and many noise schedules, 268 settings per cell. For each budget of 1, 2, 4, 6, 8 and 16 passes we kept the setting that scored best on validation data. These tuned teachers are called **F1 … F16**. The search is large but finite; `docs/method.md` §5 lists exactly what was tried. The full 99-pass teacher is called **E99**.

**The result:**

- One student pass has lower error than the tuned 4-pass teacher in all 9 cells.
- Against the tuned 6-pass teacher the data cannot tell which is better (shape error +0.07 percentage points, 95 % CI −0.49 to +0.64). That is not the same as showing they are equal.
- It is worse than the tuned 8-pass teacher and the full 99-pass sampler in all 9 cells.
- Downstream usefulness of the data is slightly lower than with the full sampler.
- Fewer passes does not mean faster by the same factor, because turning latents back into rows costs the same for every method. On one GPU, a whole table takes about 0.1–0.3 s with the student, 0.2–0.45 s with the tuned 6-pass teacher and 2.6–3.7 s with the 99-pass sampler. The student's 6–12 minutes of training only pay for themselves after millions of generated rows (`docs/numbers.md` §12).

The exact numbers, with their uncertainty, are in `docs/numbers.md`.

## 2. How the pieces fit

```
real table ──► split into train' / validation / test
                    │
                    ▼
              VAE (authors' code) ──► latents
                    │
                    ▼
        teacher = diffusion model (authors' code)
           │                         │
           │ sampled with 1–99       │ copied into
           │ passes (our sampler)    ▼
           │                   student (our code, 1–4 passes)
           ▼                         ▼
     synthetic latents ──► VAE decoder ──► synthetic table ──► scores
```

- **Cell:** one table × one training seed. There are 3 tables (adult, default, news) × 3 seeds = **9 cells**. Every comparison is made inside a cell, so student and teacher always share the same VAE.
- **train', validation, test:** we split off 10 % of each table's training data as validation, and the rest is train'. The authors' code used the *test* data to steer VAE training; we use validation instead. Test data is only used for the utility and memorisation scores.
- **Scores:** shape error (do single columns look right?), trend error (do relationships between columns look right?), C2ST (can a classifier tell fake from real?), MLE (does a model trained on fake data work on real data?) and DCR (are fake rows closer to the training rows than to unseen real rows? a memorisation check). `docs/method.md` §6 defines each one exactly.

## 3. What's in the repo

```
README.md          this file
requirements.txt   exact library versions we used
docs/
  numbers.md       every quotable number, with what each table is telling you
  method.md        every setting needed to describe what we ran
  limits.md        claims the data does not support, and things that must be disclosed
tools/             our Python code (goes inside the authors' repo, see §4)
slurm/             our cluster job scripts
results/           everything our runs produced
```

### `tools/`, by stage

| Stage | File | What it does |
|---|---|---|
| Setup | `check_env.py` | Checks versions, data and the authors' files. `gate_one` and `study_one` run it first and stop if it fails. |
| | `patch_verbose.py` | Removes one argument from two authors' scripts that crashes on current PyTorch. Run once. |
| Data | `make_val_split.py` | Makes the train' / validation split and the dataset folders (`adultv_s0`, …). |
| Teacher | `train_seeded.py` | Runs the authors' VAE and diffusion training unchanged, with a fixed seed. |
| Sampling | `samplers.py` | Our copy of the authors' sampler, with adjustable steps, solver and noise schedule. |
| | `sample_grid.py` | Generates tables from the teacher for a list of sampler settings. |
| | `sample_student.py` | Generates tables from the student at 1, 2 or 4 passes. |
| **Student** | `distill.py` | **Our method.** Trains the student. |
| Scoring | `eval_fast.py` | Shape, trend, C2ST. |
| | `eval_extra.py` | MLE (utility) and DCR (memorisation). |
| | `eval_percol.py`, `eval_pairs_common.py` | Error per column and per column pair; tail checks. |
| | `floor_boot.py` | The "floor": the error of a resample of the real data, a reference for how noisy each score is. |
| Choosing settings | `gate_decide.py` | Teacher summary per cell and the week-1 go/no-go rule. |
| | `pick_taus.py` | Student's settings for 2 and 4 passes, chosen on validation. |
| | `pick_fast.py`, `pick_fast_k.py`, `pick_fast_lo.py` | Best teacher setting per pass budget (F1…F16, F6, low-noise check), chosen on validation. |
| Reports | `review_report.py` | Produces the numbers in `docs/numbers.md` §1–2, §4–6, §7b–c and §9. Every section of `numbers.md` names its source file. |
| | `study_report.py`, `ablation_report.py`, `percol_report.py`, `student_report.py`, `extra_report.py`, `stats.py` | Supporting tables. |
| | `leak_delta_report.py`, `seed_spread_report.py` | The leak rows: our setup vs the authors', and how much the teacher varies between seeds. |
| | `manifest.py` | Records file fingerprints, versions and GPU-hours. |
| Follow-ups (11 Oct) | `bench_speed.py` | Times generation of a full table (network passes and decoding) for the student and the teacher samplers, and the break-even point for the student's training time. |
| | `ctlong_report.py` | Compares CT continued to 40k steps with CT at 20k and with our method. |
| Tests | `test_*.py` | Show our sampler matches the authors' exactly, the untrained student almost equals the 1-pass teacher (not exactly; `docs/method.md` §4 gives the difference), and the split and metrics behave. |

### `slurm/`

| File | What it runs |
|---|---|
| `study_lib.sh` | The pipeline. Every stage is a function that skips work already done. |
| `gate_one.sbatch` | Seed-0 teacher plus its full quality-vs-passes curve, for one table. |
| `study_one.sbatch` | **Main job.** One cell end to end: split → teacher → curve → student → scores. |
| `fast_one.sbatch` | Tuned teacher baselines F1…F16 for one table (3 seeds). |
| `rescore_one.sbatch` | Extra scoring added after the external review (F6, seeded C2ST, DCR, pairs). |
| `ablate_one.sbatch` | Ablations: CT (the student's target comes from its own averaged copy instead of a teacher step; it still starts from the teacher's weights), N = 8 / 40, 40k steps, and the follow-up CT run to 40k steps (`ctlong`). |
| `bench_speed.sbatch` | Follow-up speed benchmark, all 9 cells. |
| `percol_one.sbatch`, `fast_lo.sbatch`, `leak_one.sbatch` | Per-column breakdown; low-noise check; the authors' leaky setup re-run once per table. |
| `job_queue.sh`, `study_queue.sh`, `*_queue_entries.txt` | Optional helpers that submit jobs one after another under the cluster's 2-job limit. |

## 4. How to rerun

You need the CCDS GPU cluster (or another Slurm GPU cluster) and conda. All job scripts assume the code lives in `~/tabsyn` and use the partition `MGPU-TC2`. On another machine, change the `cd "$HOME/tabsyn"` line and the `#SBATCH --partition` line.

**Step 1. Get the authors' code at the exact version we used.**
```
git clone https://github.com/amazon-science/tabsyn.git ~/tabsyn
cd ~/tabsyn && git checkout cb5ac0f
```
Then copy our `tools/` and `slurm/` folders and our `requirements.txt` into `~/tabsyn/`. Our `requirements.txt` replaces the authors' one, which is not pinned. **Don't copy `results/`.** Every job skips work whose output file already exists, so copying our results in would make a rerun skip most of the work.
```
cp -r path/to/this-repo/tools path/to/this-repo/slurm path/to/this-repo/requirements.txt ~/tabsyn/
```

**Step 2. Environment.**
```
cd ~/tabsyn
conda create -n tabsyn python=3.11 -y && conda activate tabsyn
pip install -r requirements.txt
pip install libzero==0.0.8 --no-deps
```

**Step 3. Data.** Download and process **all six** of the authors' datasets. We only use three, but `check_env.py` checks all six.
```
python download_dataset.py
python process_dataset.py
```

**Step 4. One-time setup.**
```
python tools/patch_verbose.py
python tools/check_env.py      # must end with CHECK_ENV OK
mkdir -p logs results
```
`gate_one` and `study_one` run this check again at the start; the other jobs don't.

**Step 5. GPU jobs, in this order.** The GPU-hours are what each step cost us. The cluster runs at most 2 of your jobs at once.

| # | Command | Needs | GPU-h |
|---|---|---|---|
| a | `sbatch slurm/gate_one.sbatch adult` (then `default`, `news`) | — | 4.3 |
| b | `sbatch slurm/study_one.sbatch adult 0` (all 9 table/seed pairs) | a for seed 0 | 14.9 |
| c | `sbatch slurm/fast_one.sbatch adult` (then `default`, `news`) | b | 8.9 |
| d | `sbatch slurm/rescore_one.sbatch` | b, c | 7.3 |
| e | ablations: the 9 `ablate_one` lines in `slurm/ablation_queue_entries.txt` | b | 8.9 |
| f | `sbatch slurm/percol_one.sbatch` | b, c | 1.3 |
| g | `sbatch slurm/fast_lo.sbatch` | c | 0.3 |
| h | `sbatch slurm/leak_one.sbatch adult` (then `default`, `news`) | — | 5.0 |
| i | `sbatch slurm/bench_speed.sbatch` (follow-up) | b, c, d | 0.1 |
| j | `sbatch slurm/ablate_one.sbatch adult 0 ctlong` (then `default 0`, `news 0`; follow-up) | e | 1.0 |

Every job except the speed benchmark (i) skips work that is already done: if a job is killed, submit the same command again and it carries on. The benchmark reruns every cell it is given, so after a kill resubmit only the missing cells, e.g. `sbatch slurm/bench_speed.sbatch news:1 news:2`.

**Step 6. Reports** (login node, CPU only, a few minutes). These only work after Step 5, because some of them read files stored next to the trained models or in `logs/`. Run them inside `~/tabsyn`, never inside this repo: they overwrite the files in `results/`.
```
python tools/gate_decide.py --datanames adultv_s0 adultv_s1 adultv_s2 defaultv_s0 defaultv_s1 defaultv_s2 newsv_s0 newsv_s1 newsv_s2 --out results/gate_verdict_9cells.json
python tools/study_report.py --datasets adult default news --seeds 0 1 2 --tag study > results/study_report_final.txt
python tools/ablation_report.py --section all > results/ablation_report_all.txt
python tools/percol_report.py > results/percol_report.txt
python tools/review_report.py > results/review_report_final.txt
python tools/leak_delta_report.py > results/leak_delta_report.txt
python tools/seed_spread_report.py > results/seed_spread_report.txt
python tools/ctlong_report.py > results/ctlong_report.txt
python tools/manifest.py
```
`bench_speed.sbatch` writes its own report, `results/bench_speed_report.txt`.

A full rerun will differ from ours in two known places, apart from GPU noise:

- `review_report.py` (R8) lists the recipe-search runs on adult s0. Those aren't part of this repo, so a rerun lists none instead of 19.
- Adult s0's E99 utility score and its predictor-only DCR (the leak rows) used 20 sampling seeds in our run; a fresh run uses 10, like every other cell (`docs/method.md` §3).

**Exact match:** fresh runs on a GPU will not reproduce our numbers to the last digit, because GPU arithmetic is not fully deterministic. `results/MANIFEST.json` records SHA-256 fingerprints of the code, the model weights and the result files as they were on the cluster on 2 Oct 2026. The weights aren't in this repo; their fingerprints identify the exact models behind our numbers.

**Differences from the code that ran.** Every file in `tools/` and `slurm/` is byte-identical to the cluster version recorded in `results/MANIFEST.json`, except for these changes made after it:

| File | Change | Effect on results |
|---|---|---|
| `slurm/study_one.sbatch`, `leak_one.sbatch`, `gate_one.sbatch` | Added `--legacy_target_in_fix` to the split command, with a comment. Without it, a rerun on news builds a split with 37 fewer training rows than ours. | None on adult or default. On news it makes a rerun use the split we used. |
| `slurm/gate_one.sbatch`, `fast_lo.sbatch`, `percol_one.sbatch`, `ablate_one.sbatch`; `tools/check_env.py` | New or extended description at the top of the file. | None (comments only). |
| `tools/eval_percol.py` | Corrected one out-of-date comment. | None (comments only). |
| `tools/leak_delta_report.py`, `tools/seed_spread_report.py` | New description at the top. E99 now also requires churn 0; before, these two scripts averaged churn 0 and churn 1 together. | Only these two scripts' printed tables change; no experiment is affected. |
| `tools/eval_pairs_common.py` | Fixed the handling of synthetic values outside the real categories or bins, in the column error, the pair error and the per-pair copies of the column errors (11 Oct). | None: such values never occurred in the scored files (checked for all 13,050 column × file combinations). |
| `tools/review_report.py` | F1 and F2 "including σ_max < 1" now use the σ_max < 1 setting whenever it won on validation (before: the lower of the two final scores). R6 reads the shipped `*_percol_determinism.json` records when the job logs are absent. R8 counts the follow-up `ctlong` arm as an ablation, not a recipe-search run. | None: the values are identical. |
| `slurm/study_lib.sh` | Corrected one out-of-date comment; added the follow-up `ctlong` stage (CT continued to 40k steps). After the follow-up runs, its final check was changed to read the model's run record, so a rerun that only finishes scoring no longer fails. | None on earlier results; adds one arm. |
| `tools/bench_speed.py`, `tools/ctlong_report.py`, `slurm/bench_speed.sbatch` | New follow-up files (11 Oct). After the runs, `bench_speed.py` was changed to synchronise the GPU it measures by name; it matters only with `--gpu` other than 0, and the runs used GPU 0. | Add the follow-up results only. |

In `results/`, every file matches its fingerprint in `MANIFEST.json`; the 18 `.gz` files match once unzipped. These files are exceptions:

- `sacct.txt` and `MANIFEST.json` itself, which the manifest script rewrites after fingerprinting.
- 19 files that the manifest doesn't cover:
  - the ablation, per-column and study summary tables, which were regenerated for this handover from fingerprinted files and are identical to the versions used before;
  - `review_report_final.txt`, a copy of the cluster log;
  - the two leak tables.
- The follow-up results from 11 Oct (speed benchmark and CT to 40k steps), made after the manifest.

Not shipped, for reference:

- the recipe-search and pilot scripts and their results;
- the figures and the scripts that drew them;
- a chat helper;
- an older analysis script, replaced by `review_report.py`;
- three files that are out of date or no longer needed (`_stage_rescore.sh`, `test_percol.py`, `patch_2026-10-01.py`).

## 5. Reading the results

**Start with `docs/numbers.md`.** It has every quotable number, and each section opens with a note on what it shows. The results files behind it:

| File | What's in it |
|---|---|
| `results/review_R1_paired.csv` | Student vs every baseline: mean difference, CI, wins out of 9, p-value. **The main result.** |
| `results/review_cells.csv` | The raw scores for every method in every cell. |
| `results/review_R2_utility_*.csv`, `review_R3_dcr.csv`, `review_R4_c2st*.csv` | Utility, memorisation, C2ST. |
| `results/review_R5_pairs.csv`, `review_R5_tails.csv`, `percol_*.csv` | Where the student's remaining error is. |
| `results/ablation_*_summary.csv`, `ablation_report_all.txt` | Ablations. Lines starting `READING:` are the automatic output of rules written in the project's ablation plan. A rule's output is not a conclusion: check each one against `docs/limits.md` before using it (the CT one says more than the evidence supports). |
| `results/*bench_speed*`, `results/*ctlong*` | Follow-ups: measured generation times (`numbers.md` §12); CT continued to 40k steps (`numbers.md` §8). |

**File names:** per-cell files start with the cell, e.g. `newsv_s2_fast.jsonl` is news, seed 2, tuned baselines. A `v` means our leak-free split; `L` means the authors' leaky setup. The 18 files ending in `.jsonl.gz` are compressed raw per-pair scores; unzip them with `gunzip -k results/*.jsonl.gz` before opening them.

## 6. Who needs what for the report

| Section | Read | Code to look at, if you want to see how it works |
|---|---|---|
| Method | `docs/method.md` | `tools/distill.py` (the student), `tools/samplers.py` (baselines), `slurm/study_lib.sh` (the pipeline) |
| Results | `docs/numbers.md` §1–6 and §12 | `tools/review_report.py`, `tools/bench_speed.py` |
| Analysis | `docs/numbers.md` §7–8 | `tools/eval_pairs_common.py`, `tools/ablation_report.py` |
| Limitations and conclusion | `docs/limits.md` | — |

Tomás is writing the introduction and background.

## 7. Easy mistakes

- **Over-claiming.** `docs/limits.md` section A lists the claims that were checked and are not supported. Read it before writing any sentence that says "better", "same as" or "because".
- **"Can't tell apart" is not "equal".** A CI that crosses zero means the data can't say which method is better. We ran no test that shows two methods are equal.
- **E99 is not the authors' exact sampler setting.** We ran it without noise injection (churn 0). The authors' default (churn 1) differs by 0.007 pp on average.
- **Counting passes.** Heun uses 2 passes per step except the last, so 50 steps = 99 passes. Always count passes (NFE), not steps.
- **Passes are not time.** 99 passes vs 1 pass does not mean 99× faster. Use the measured times in `docs/numbers.md` §12.
- **Higher or lower is better?** Errors: lower is better. C2ST, AUROC and R²: higher is better. DCR: closest to 0.5 is best.
- **Default's trend error** is excluded from all reported tests and from the go/no-go rule, because the metric is too noisy on that table. It is still used to break ties when choosing settings.

## 8. Glossary

| Term | Meaning |
|---|---|
| NFE / pass | One run of the network. It counts network computation, not wall-clock time. |
| Teacher | The authors' trained diffusion model. |
| Student | Our one-pass network, trained to copy the teacher. |
| Consistency distillation (CD) | Training the student so that any noisy point on the teacher's path maps to the same final output. |
| Consistency training (CT) | The same training, but the target comes from the student's own averaged copy instead of a step of the teacher. In our ablation the student still starts from the teacher's weights. |
| E99 | Teacher with the authors' 99-pass schedule. |
| F_k | Teacher with the setting that scored best on validation among those we searched, using at most k passes. |
| S1, S2, S4 | Student with 1, 2 or 4 passes. |
| Solver (Euler, Heun, DPM-Solver++) | Different rules for taking a denoising step. |
| σ_max, ρ | Starting noise level, and how the noise levels are spaced. Part of the "noise schedule". |
| Churn | Extra noise the authors add at every step. |
| EMA | A slowly updated running average of the network's weights. |
| τ schedule | The noise levels the student re-adds between passes when using 2 or 4 passes. |
| Cell | One table × one training seed (9 in total). |
| train' | The 90 % of the training split we train on (the other 10 % is validation). |
| Floor | The average error of a resample of the real training data, scored against that data. A reference for how noisy a score is, not a hard limit. |
| pp | Percentage points. |
| Sign test, CI | Wins out of 9 turned into a p-value; 95 % range for the mean difference. |
| IPF | Iterative proportional fitting. Used as a sensitivity check: each two-column synthetic table is rescaled so its single columns match the real data, then scored again. It does not split the error into causes. |
