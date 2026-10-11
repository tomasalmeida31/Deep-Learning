# Method facts

Everything needed to describe what was run, stated as facts. Where a fact comes from a specific file, it is named. The code is the final authority: if this file and the code disagree, the code is right, and Tomás needs to know.

## 1. The base model: TabSyn (Zhang et al., ICLR 2024)

| Part | What it is | What we did with it |
|---|---|---|
| VAE | Turns each table row into tokens: one 4-number token per column plus one extra (CLS) token. The diffusion model uses only the column tokens: the authors' `latent_utils.py` drops the CLS token. | Trained with the authors' code, unchanged: 4,000 epochs. β starts at 1e-2. The selection-slot loss is the categorical cross-entropy only; after 10 epochs without improvement β is multiplied by λ = 0.7, as long as β > 1e-5. The encoder and decoder are taken from the final epoch (`tools/train_seeded.py`; authors' `tabsyn/vae/main.py`). |
| Latent scaling | z ← (z − mean) / 2 | Unchanged; the student uses the same scaling |
| Diffusion model (the **teacher**) | EDM denoiser D(x, σ): the authors' `MLPDiffusion(dim, 1024)`, an MLP with a 1024-wide input projection and time embedding and two hidden layers of width 2048, with EDM preconditioning, σ_data = 0.5 | Trained with the authors' code, unchanged: batch 4,096, Adam lr 1e-3 with ReduceLROnPlateau, up to 10,001 epochs, checkpoint on best training loss |
| Authors' sampler | Heun on the Karras noise grid σ_max = 80 → σ_min = 0.002, ρ = 7; 50 steps = 99 NFE (Heun uses 2 evaluations per step except the last); S_churn = 1 | Re-implemented with adjustable settings in `tools/samplers.py`. A test (`tools/test_samplers.py`) shows it equals the authors' sampler bit for bit at 50, 10 and 2 steps |
| One code change to the authors' files | `verbose=True` removed from `ReduceLROnPlateau(...)` in `tabsyn/vae/main.py` and `tabsyn/main.py` | Needed because current PyTorch rejects the argument (`tools/patch_verbose.py`) |

## 2. Data

| Table | Task | Columns | Latent size used by the diffusion model | Train' rows | Validation rows | Test rows (shipped) |
|---|---|---|---|---|---|---|
| adult | binary classification | 15 | 60 | 29,306 | 3,255 | 16,281 |
| default | binary classification | 24 | 96 | 24,300 | 2,700 | 3,000 |
| news | regression | 48 | 192 | 32,148 | 3,531 | 3,965 |

Sources: `results/MANIFEST.json` → `splits`; columns from `results/review_summary.json` → `R7`. R7's "latent_dim" (64 / 100 / 196) and its variance figures include the CLS token. The diffusion model, the student and the pseudo-Huber constant use 4 × columns (60 / 96 / 192).

- **Split** (`tools/make_val_split.py`): 10 % of the shipped training split becomes validation (fixed seed 1234, the same for every training seed); the other 90 % is train'. A validation row is moved back into train' if it holds a category value (or, for news, a target value) that train' lacks. This moved 1 row on adult, 0 on default and 37 on news (`results/*_split_audit.json`). On news the move depends on the target value, so it changes the training data and the train' reference used for scoring, as well as the validation set.
- **Why:** in the authors' code the VAE's learning-rate schedule and β annealing are driven by the loss on the *test* split. We put the validation carve in that slot, so the test split is used only by MLE and DCR.
- **Leak rows:** `adultL_s0`, `defaultL_s0`, `newsL_s0` re-run the authors' original arrangement (shipped test in the selection slot) once per table, teacher only, as a sensitivity check (`slurm/leak_one.sbatch`).

## 3. Seeds

- Training seeds 0, 1, 2. One seed fixes the VAE, the teacher and the student of that cell. Teacher and student in a cell share the same VAE.
- Sampling seed s uses `torch.manual_seed(1000003·s + 17)`, the same for every generator.
- Validation searches use seeds 0–4 with 8,192 rows each, scored against validation. Final scores use seeds 0–19 with |train'| rows, scored against train'. MLE and DCR use seeds 0–9.
- Exceptions: E99's C2ST uses 10 seeds (0–9) per cell, against 20 for the student and the F_k. Adult s0's E99 utility (MLE) and its predictor-only DCR (the leak rows, `numbers.md` §11) use 20 seeds, because an earlier job used 20; a fresh run uses 10. The full-record DCR in `numbers.md` §6 uses 10 for every generator, adult s0's E99 included.

## 4. The improvement: the consistency-distilled student (`tools/distill.py`)

| Item | Setting |
|---|---|
| Method | Consistency distillation, Song et al., ICML 2023, Algorithm 2 |
| Student function | f(x, σ) = c_skip(σ)·x + (1 − c_skip(σ))·D_θ(x, σ), with c_skip(σ) = σ_d² / ((σ − σ_min)² + σ_d²), σ_d = 0.5, σ_min = 0.002 |
| Boundary condition | f(x, σ_min) = x exactly |
| Initialisation | D_θ starts as a copy of the teacher. Because c_skip(80) ≈ 4·10⁻⁵, the untrained student differs from the 1-evaluation teacher D(80z, 80) only by c_skip(80)·(80z − D(80z, 80)), which is almost nothing |
| Parametrisation note | This applies the skip/output mix to the EDM-preconditioned denoiser. It differs from the c_out·F_θ scaling in Song et al. |
| Noise grid | Karras, N = 18 points from 0.002 to 80, ρ = 7 |
| One training step | Draw latents x₀ from train' minus the 4,096 held-out latents (25,210 / 20,204 / 28,052 rows for adult / default / news; the teacher was trained on all of train') and noise z; pick n uniformly from {1, …, N−1}; set x_{n+1} = x₀ + t_{n+1}·z. Take one teacher Heun step from t_{n+1} to t_n (2 teacher evaluations) to get x̂_n. Loss = d(f_θ(x_{n+1}, t_{n+1}), f_θ⁻(x̂_n, t_n)) |
| Target network θ⁻ | EMA of θ with μ = 0.999 |
| Distance d | pseudo-Huber, √(‖·‖² + c²) − c with c = 0.00054·√(latent size), latent size 60 / 96 / 192 |
| Weighting | uniform |
| Optimiser | Adam, lr 1e-4 constant, batch 4,096, gradient-norm clip 1.0 |
| Length | 20,000 updates |
| Weights used for sampling | a separate EMA of θ with rate 0.999 |
| Checkpoint choice | every 250 steps, the EMA weights with the lowest self-consistency loss on 4,096 held-out train' latents (never trained on). In all 9 headline runs this was the last checkpoint (step 20,000), so the choice did not change the headline student |
| 1-evaluation sampling | x = f(80·z, 80) |
| 2 / 4-evaluation sampling | multistep consistency sampling (Song et al. 2023, Algorithm 1): after the first step, re-noise to level τ and apply f again, for each τ in a schedule |
| τ schedule | chosen on validation, lowest mean shape error: 1 / 7 / 13 candidate schedules for 1 / 2 / 4 evaluations (`tools/sample_student.py`, `tools/pick_taus.py`). For 2 evaluations the candidates are τ ∈ {0.5, 1, 2, 4, 8, 16} plus the default; validation chose the smallest, 0.5, in 6 of 9 cells, so smaller values were never tried |

**How the recipe was fixed:** 19 recipe variants (μ, N, steps, pilot settings) were trained on adult seed 0, plus one μ check on adult seed 1 (`results/review_summary.json` → `R8`). The recipe above was then frozen and applied unchanged to all 9 cells. Adult s0 and s1 are among the 9 evaluated cells; `numbers.md` §1 repeats the main comparisons without them.

**CT ablation:** the `ct` arm uses the same student, initialisation from the teacher's weights, grid, EMA and number of updates. The only difference is the training target: x̂_n = x₀ + t_n·z with the same noise z (Song et al. 2023, Algorithm 3) instead of a teacher Heun step. It is teacher-initialised consistency training, not training from scratch. The `ict` arm additionally sets μ = 0 and uses 1/Δt weighting; it does not include the other parts of the improved-CT recipe (noise-schedule curriculum, lognormal noise sampling). A follow-up (11 Oct) continued the `ct` arm to 40k updates on seed 0 of each table (`ctlong`, `slurm/study_lib.sh`), in the same way as the 40k CD arm.

## 5. What the student is compared with

| Baseline | Definition | Source |
|---|---|---|
| E99 | Authors' schedule, Heun 50 steps = 99 NFE, run with S_churn = 0 (deterministic). The authors' S_churn = 1 version is also scored. | `results/<cell>_teacher_c2st.jsonl`, `results/review_cells.csv` |
| 1-evaluation teacher | D(σ_max·z, σ_max), one denoiser pass. **Our definition**: the authors' code returns NaN at 1 step. | `tools/samplers.py` |
| Teacher curve | Heun with {1, 2, 3, 4, 5, 6, 8, 10, 13, 20, 50} steps × churn {0, 1}, plus Euler with {2, 4, 8, 16} steps, 20 seeds each (520 scored samples). Seed-0 cells add a σ_max {80, 10, 3} × ρ {7, 3, 1} sweep at 5 seeds (770 scored samples in total). | `results/<cell>_gate*.{jsonl,csv}` |
| **F_k (tuned baselines)** | Candidates: Euler, DPM-Solver++(2M) and Heun; σ_max ∈ {1, 2, 3, 5, 10, 20, 80}; ρ ∈ {1, 3, 7} where it matters, giving **196 configurations**. Added later: 63 six-evaluation configurations (for F6) and 9 configurations with σ_max ∈ {0.3, 0.5, 0.7} (for F1, F2). For each budget k, F_k is the configuration with NFE ≤ k and the lowest mean validation shape error (5 seeds × 8,192 rows; ties broken by trend). The search is finite. Budgets are 1, 2, 4, 6, 8 and 16 evaluations (no 3-evaluation Euler or DPM-Solver++ candidates); σ_min is fixed at 0.002; other solver families (e.g. UniPC) were not tried. | `tools/sample_grid.py` (grids `fast_val`, `fast6`, `fast_lo`), `tools/pick_fast*.py`, `results/<cell>_fast*_choice.json` |
| What validation chose | F4, F6, F8 and F16: DPM-Solver++(2M) in every cell, σ_max between 2 and 20. F1: one Euler step from σ_max = 1 (0.7 in three cells). | `results/review_cells.csv` (`*_cfg` columns) |

Search effort per cell: the teacher got 196 + 63 + 9 sampler configurations; the student got 1 / 7 / 13 τ schedules.

## 6. Metrics

| Metric | Definition | Tool |
|---|---|---|
| Shape error | 100 × (1 − SDMetrics Column Shapes score). Per column: KS complement for numerical columns, TV complement for categorical. Scored against train' with \|train'\| synthetic rows. Same construction as the authors' `eval/eval_density.py`. | `tools/eval_fast.py` |
| Trend error | 100 × (1 − SDMetrics Column Pair Trends score). Correlation similarity for numerical-numerical pairs, contingency similarity for all others. | `tools/eval_fast.py` |
| C2ST | SDMetrics LogisticDetection, 1 − mean over 3 folds of (max(AUC, 0.5)·2 − 1). 1 = indistinguishable. Re-scored on 1–2 Oct with a seeded split, 3 repeats. | `tools/eval_fast.py` |
| MLE | XGBoost trained on synthetic data and tested on the shipped test split. Grid: n_estimators {50, 100} × max_depth {4, 8}. The model is chosen on a held-out 1/9 slice of the synthetic data (by accuracy for classification, by MSE for regression) and used as fitted on the other 8/9. AUROC for adult and default; R² on log1p(target) for news. Re-implemented and different from the authors' `eval/mle/mle.py`, which uses its own feature transform, a 36-setting grid, selection by the reported metric (e.g. AUROC), a refit on all synthetic rows, and XGBoost's GPU tree method ('gpu_hist'; ours uses CPU 'hist'). The same protocol is applied to every generator. | `tools/eval_extra.py` |
| DCR | L1 distance over range-scaled numerical and one-hot categorical columns, target included. Score = share of 5,000 synthetic rows closer to a training reference than to test, where the training reference is a random subsample of train' the size of the test set. 0.5 is ideal. "Exact copies" are exact matches with that subsample only. Differs from the authors' `eval_dcr.py` in the equal-size reference. | `tools/eval_extra.py` |
| Floor | Mean error of 5 bootstrap resamples of train' scored against train'. A reference for metric noise, not a lower bound. | `tools/floor_boot.py` |
| Per-column / per-pair breakdown | SDMetrics per-column and per-pair scores | `tools/eval_percol.py` |
| One-metric pair score | Binned contingency distance (total variation) with 10 bins at real-data quantiles, for every pair type. The "after IPF" version first adjusts each two-column synthetic table to the real one-column distributions (iterative proportional fitting, 50 sweeps, no convergence check). It is a sensitivity check, not a decomposition of the error. Tails: 99th-percentile error divided by the real IQR (by the sd, or 1, if the IQR is 0), and synthetic mass beyond the real 1st and 99th percentiles. A coding error for synthetic values outside the real categories or bins was fixed on 11 Oct 2026; it never occurred in the scored files. | `tools/eval_pairs_common.py` |

## 7. Statistics

- The unit is the cell (9). Within a cell, scores are averaged over sampling seeds first. Comparisons are paired differences within a cell, so every pair shares the same VAE and teacher.
- Reported for each comparison: mean difference, wins out of 9, an exact two-sided sign test, a 95 % t-interval over cells, and the range over cells. The interval is about the mean difference; the sign test is about how often each method wins. They can disagree.
- p-values are per comparison, not adjusted for the number of comparisons (§1 of `numbers.md` alone has 15). For example, a Bonferroni correction over those 15 would turn p = 0.004 into 0.06.
- The 9 cells are 3 training seeds of 3 fixed tables, not 9 independent datasets. Results are about these three tables; at table level, the same sign on all 3 tables gives p = 0.25.
- No equivalence test was run. A CI that includes zero is not evidence that two methods are equal.
- Trend error on default is excluded from every reported test and from the gate, because its floor varies too much: floor sd 2.38 pp against a limit of δ/3 = 0.167 pp (`results/gate_verdict_9cells.json`). Validation trend error is still used as a tie-breaker when selecting samplers and τ schedules, on every table.

## 8. Decision rules

These rules come from the project's protocol (dated 20 Sep 2026) and ablation plan (dated 24 Sep 2026). Those documents are kept by Tomás and are not in this repo. That each rule was written before the runs it governs is the authors' account and cannot be checked from the repo; the code that applies each rule is named in the table.

| Rule | Setting | Source |
|---|---|---|
| Gate (week 1) | tolerance per metric = max(δ, 3 × sampling sd), with δ_shape = 0.25 pp and δ_trend = 0.5 pp. NFE* = the smallest NFE, among the gate's settings (the authors' Heun schedule plus a small schedule sweep), within tolerance of E99. GO if NFE* ≥ 7 on at least two-thirds of the units given: 2 of 3 tables at the week-1 gate, or 6 of 9 cells when run on all cells. **Result: GO** (9 of 9). | `tools/gate_decide.py`, `results/gate_verdict_9cells.json` |
| Headline (E1) | the largest k ∈ {4, 8, 16} with S1 < F_k in at least 8 of 9 cells | `tools/ablation_report.py` |
| CT vs CD (E2) | three readings, each needing at least 7 of 9 cells at both 1 and 4 evaluations: CT worse by ≥ 0.25 pp; within ±0.25 pp; or better by ≥ 0.25 pp. Otherwise "inconclusive". | same |
| Grid size N (E3) | "N = 18 is enough" if N = 8 is worse by ≥ 0.25 pp at all of 1/2/4 evaluations in ≥ 2 of 3 cells and N = 40 is within ±0.25 pp at all of 1/2/4 in ≥ 2 of 3 cells; "N = 40 better" if better by ≥ 0.25 pp at all of 1/2/4 in ≥ 2 of 3; otherwise no reading | same |
| Training length (E4) | adopt 40k only if 4-evaluation shape improves by at least 0.25 pp in at least 7 of 9 cells; otherwise the headline stays at 20k | same |
| Per-column (E5) | (a) "concentrated" if the worst 20 % of columns carry at least 50 % of the positive excess; (b) a type statement only if categorical and numerical columns differ by at least 0.5 pp in the same direction on at least 2 tables; (c) "same error mode as the coarse teacher" only if Spearman ρ ≥ 0.5 on all 3 tables | `tools/percol_report.py` |
| News MLE validity (E7) | Raw-target RMSE on news was unstable (raw R² is negative, e.g. −1.44 for E99 on news s0), so news utility is scored as R² on log1p(target). The rule: report it only if real > E99 > 1-evaluation teacher on all three news cells. It held, 3/3. This is a sanity ordering, not a validity test, and it was checked on the final test scores. | ablation plan E7.1; `results/<cell>_mle_log.jsonl` |
| **Recorded amendment** | The E5 determinism rule ("stop if a regenerated sample differs at all") stopped the job on default s0 at 0.052 pp (job 37650, 25 Sep). The analysis was re-run the same day (job 37697) under "record the difference, label the cell, stop only above 0.1 pp". | `results/review_R6_determinism.csv`, `results/sacct.txt` |

## 9. Environment and compute

- CCDS GPU cluster: one GPU per job, 6-hour job limit, at most 2 jobs queued.
- NFE counts network evaluations, not wall-clock time. Generation time, including decoding, is measured separately by `tools/bench_speed.py` (`numbers.md` §12).
- Python 3.11.16, torch 2.11.0 + CUDA 12.8, SDMetrics 0.11.1, numpy 1.26.4, pandas 2.0.3, scikit-learn 1.9.1, xgboost 3.2.0, scipy 1.17.1. Full list in `results/conda_list.txt`.
- GPU-hours are in `numbers.md` §10.
