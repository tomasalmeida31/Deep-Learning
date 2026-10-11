# Numbers

Every number in this file is copied from the results file named in its section heading. Quote numbers from here or from those files, never from memory, chat logs or older drafts. If a number you need is missing, take it from the results file and tell Tomás which one.

## Notation

| Symbol | Meaning |
|---|---|
| cell | one table × one training seed: adult, default or news × seeds 0, 1, 2 (9 cells) |
| S1, S2, S4 | our consistency-distilled student, sampled with 1, 2 or 4 network evaluations |
| F_k | the teacher sampler with at most k evaluations that scored best on validation data among the settings we searched (k = 1, 2, 4, 6, 8, 16; `method.md` §5 lists what was searched) |
| E99 | the authors' sampler: Heun, 50 steps = 99 evaluations, run with churn 0 |
| NFE | number of network evaluations used to generate one sample. It counts network computation, not wall-clock time |
| shape / trend | column-shape error and pair-trend error, in percentage points (pp), against the training data; lower is better |
| C2ST | classifier two-sample test score; 1 = indistinguishable from real, higher is better |
| A − B | paired difference averaged over cells; for errors, negative = A better |
| wins | number of cells where the first-named method is better |
| p | exact two-sided sign test over cells, for that comparison alone (not adjusted for the number of comparisons made) |
| CI | 95 % t-interval over cells |

## 1. Shape error, paired differences (`results/review_R1_paired.csv`)

*What this section shows:* the main result. Each row compares the student with one baseline over the same 9 cells. Read the sign of the mean together with the Wins column. A negative mean with 9/9 wins means the student had lower error in every cell. A CI that crosses zero means the data cannot tell which of the two is better; it does not show that they are equal, and no equivalence test was run. Going down the table, the baselines get more evaluations (F1 → F16 → E99), so the rows show the budget at which the student stops winning.

| Comparison | Mean (pp) | 95 % CI | Range over cells | Wins | p |
|---|---|---|---|---|---|
| S1 − F1 | −15.25 | −16.86 .. −13.64 | −17.71 .. −11.51 | 9/9 | 0.004 |
| S1 − F1 (incl. σ_max < 1 candidates) | −14.96 | −16.85 .. −13.08 | −17.71 .. −10.26 | 9/9 | 0.004 |
| S2 − F2 (incl. σ_max < 1 candidates) | −15.20 | −17.12 .. −13.27 | −18.16 .. −10.43 | 9/9 | 0.004 |
| S4 − F4 | −2.40 | −3.36 .. −1.45 | −3.93 .. −0.59 | 9/9 | 0.004 |
| S1 − F4 | −2.01 | −2.75 .. −1.28 | −3.15 .. −0.43 | 9/9 | 0.004 |
| S1 − F6 | +0.07 | −0.49 .. +0.64 | −0.83 .. +1.39 | 4/9 | 1.0 |
| S1 − F8 | +0.93 | +0.53 .. +1.33 | +0.35 .. +1.65 | 0/9 | 0.004 |
| S1 − F16 | +1.54 | +1.23 .. +1.85 | +0.94 .. +2.35 | 0/9 | 0.004 |
| S1 − E99 | +1.53 | +1.21 .. +1.84 | +0.88 .. +2.21 | 0/9 | 0.004 |
| S2 − E99 | +1.15 | +0.89 .. +1.42 | +0.57 .. +1.73 | 0/9 | 0.004 |
| S4 − E99 | +1.14 | +0.92 .. +1.36 | +0.72 .. +1.67 | 0/9 | 0.004 |
| S1 − E99 with the authors' churn = 1 | +1.52 | +1.20 .. +1.85 | +0.88 .. +2.19 | 0/9 | 0.004 |
| E99 churn 1 − E99 churn 0 | +0.007 | −0.024 .. +0.038 | −0.041 .. +0.076 | 6/9 | 0.51 |

Trend error, 6 cells (default excluded, see `method.md` §7). *The same comparison on the second metric, which measures how well relationships between columns are kept.*

| Comparison | Mean (pp) | 95 % CI | Range | Wins | p |
|---|---|---|---|---|---|
| S1 − F4 | −1.48 | −3.26 .. +0.31 | −4.17 .. +0.04 | 5/6 | 0.22 |
| S1 − F8 | +0.89 | +0.32 .. +1.45 | +0.20 .. +1.50 | 0/6 | 0.031 |

Sampling seeds 5–19 only (`results/review_summary.json` → `R1_seeds5to19`). *Seeds 0–4 were also used to tune the baselines on validation data, so this checks the result doesn't depend on them:* S1 − F4 −2.02 (9/9), S1 − F8 +0.92 (0/9), S4 − F4 −2.40 (9/9).

Excluding adult s0, the cell the student recipe was tuned on (computed from `results/review_cells.csv`): S1 − F4 −1.92 (8/8), S1 − F6 +0.16 (3/8), S1 − F8 +0.98 (0/8), S1 − E99 +1.50 (0/8). Excluding adult s1 as well, which was used for one recipe check: S1 − F4 −1.87 (7/7), S1 − F6 +0.22 (2/7), S1 − F8 +1.05 (0/7), S1 − E99 +1.51 (0/7).

Per table, mean over 3 seeds (means: `results/review_summary.json` → `R1_per_dataset`; ranges: `results/review_report_final.txt`, R1). *The main comparisons split by table, to see whether one table drives the average:*

| Table | S1 − F4 | S1 − F6 | S1 − F8 | S1 − E99 | S4 − F4 |
|---|---|---|---|---|---|
| adult | −1.83 (−2.79 .. −0.43) | −0.21 (−0.61 .. +0.27) | +0.57 (+0.46 .. +0.74) | +1.36 (+0.88 .. +1.77) | −2.22 (−3.42 .. −0.59) |
| default | −2.35 (−3.09 .. −1.50) | −0.31 (−0.83 .. +0.59) | +0.71 (+0.35 .. +1.31) | +1.67 (+1.52 .. +1.93) | −2.93 (−3.60 .. −2.13) |
| news | −1.87 (−3.15 .. −1.06) | +0.73 (+0.22 .. +1.39) | +1.49 (+1.20 .. +1.65) | +1.55 (+1.12 .. +2.21) | −2.06 (−3.93 .. −0.71) |

Brackets give the range over the 3 seeds; they are not confidence intervals. Counting tables instead of cells, a result with the same sign on all 3 tables has sign-test p = 0.25 (`results/review_report_final.txt`, R1).

Multistep sampling, computed from `results/review_cells.csv`. *Whether giving the student a second or fourth evaluation improves it:* S2 − S1 = −0.37 pp (S2 better in 8/9); S4 − S2 = −0.02 pp (S4 better in 5/9). On news seed 0, S2 and S4 are both worse than S1.

## 2. Shape error per cell, pp (`results/review_cells.csv`)

*What this section shows:* the raw errors behind §1, one row per cell. Use it to show the spread across tables and seeds, or to check any single comparison. Lower is better. Compare along a row rather than down a column, because some tables are harder than others: averaged over seeds, news errors are higher for every method.

| Cell | S1 | S2 | S4 | F1 | F4 | F6 | F8 | F16 | E99 |
|---|---|---|---|---|---|---|---|---|---|
| adult s0 | 2.53 | 1.97 | 1.89 | 15.16 | 5.32 | 3.14 | 2.07 | 0.91 | 0.76 |
| adult s1 | 2.46 | 2.22 | 2.07 | 17.04 | 4.72 | 2.75 | 1.95 | 1.19 | 1.02 |
| adult s2 | 3.22 | 2.91 | 3.06 | 14.73 | 3.65 | 2.95 | 2.48 | 2.27 | 2.34 |
| default s0 | 2.99 | 2.42 | 2.48 | 20.70 | 6.08 | 3.68 | 2.51 | 1.58 | 1.43 |
| default s1 | 2.64 | 2.11 | 2.03 | 18.77 | 5.08 | 3.46 | 2.29 | 1.28 | 1.12 |
| default s2 | 4.78 | 4.20 | 4.15 | 21.56 | 6.28 | 4.19 | 3.48 | 2.92 | 2.85 |
| news s0 | 5.70 | 5.87 | 6.06 | 21.29 | 6.77 | 4.32 | 4.05 | 4.05 | 4.39 |
| news s1 | 4.68 | 4.20 | 3.90 | 22.00 | 7.84 | 4.47 | 3.07 | 2.34 | 2.47 |
| news s2 | 5.10 | 4.84 | 4.95 | 20.11 | 6.50 | 4.52 | 3.90 | 3.71 | 3.98 |

Each value is the mean over 20 sampling seeds. On news s0, F16 is the 8-evaluation configuration (validation chose it for k = 16).

## 3. The teacher on its own (`results/gate_verdict_9cells.json`, `results/review_report_final.txt` R1)

*What this section shows:* the starting point, before any student exists. It gives three things per table:

- the floor: the error that a resample of the real training data gets, which shows how noisy the metric is;
- how good the full 99-evaluation teacher is;
- how many evaluations the teacher needs to get within tolerance of that, first with the authors' schedule and then with our tuned samplers.

The last two paragraphs show how bad the teacher is with a single evaluation, and that its starting-noise setting was searched properly rather than left at a poor default.

| Table | Floor, shape (pp) | Floor, trend (pp) | E99 shape, seeds 0 / 1 / 2 (pp) | Smallest NFE within tolerance of E99: week-1 gate grid (seeds 0 / 1 / 2) | … with the tuned samplers F_k (seeds 0 / 1 / 2) |
|---|---|---|---|---|---|
| adult | 0.37 ± 0.03 | 0.95 ± 0.11 | 0.76 / 1.02 / 2.34 | 39 / 39 / 39 | 16 / 16 / 8 |
| default | 0.48 ± 0.04 | 4.95 ± 2.38 | 1.43 / 1.12 / 2.85 | 39 / 39 / 39 | 16 / 16 / 16 |
| news | 1.42 ± 0.01 | 0.38 ± 0.03 | 4.39 / 2.47 / 3.98 | 25 / 39 / 25 | 6 / 16 / 8 |

The floor is the mean error of 5 bootstrap resamples of train' scored against train' (± their sd). It is a reference for how noisy the metric is, not a hard lower bound: a generator can in principle score below it.

"Within tolerance" uses the gate rule (`method.md` §8): shape, and trend except on default, no more than the tolerance above E99. The gate-grid column is the smallest NFE among the week-1 gate's settings: Heun with churn 0 and 1 and Euler on the authors' noise schedule (σ_max = 80, ρ = 7), plus a small σ_max × ρ sweep on the seed-0 cells. Every match came from Heun on the authors' schedule. The tuned-sampler column applies the same rule to F4, F6, F8 and F16 as measured later (computed from `results/review_cells.csv` and the tolerances in `results/gate_verdict_9cells.json`). Both are the smallest budgets among the settings tried, not a proven minimum.

Best 1-evaluation teacher (F1, including σ_max < 1): 13.48 to 22.00 pp across the 9 cells.

Validation shape error of the 1-evaluation teacher as a function of starting noise σ_max, adult s0: 0.3 → 21.8, 0.5 → 17.2, 0.7 → 15.7, 1.0 → 15.7, 2.0 → 20.7, 3.0 → 25.5. Every cell's minimum is at σ_max = 0.7 or 1.0 (0.7 in adult s1, adult s2, news s0).

## 4. C2ST, seeded (`results/review_R4_c2st_paired.csv`; higher = better, wins = student higher)

*What this section shows:* a second, independent check of quality: can a simple classifier tell synthetic rows from real ones? The layout is the same as §1, but here **higher is better**, so a positive mean favours the student. Use it to see whether this metric agrees with shape error at each budget.

| Comparison | Mean | 95 % CI | Wins | p |
|---|---|---|---|---|
| S4 − F4 | +0.150 | +0.104 .. +0.195 | 9/9 | 0.004 |
| S1 − F6 | +0.002 | −0.029 .. +0.033 | 5/9 | 1.0 |
| S1 − F8 | −0.034 | −0.058 .. −0.010 | 2/9 | 0.18 |
| S1 − E99 | −0.057 | −0.088 .. −0.026 | 1/9 | 0.039 |
| S2 − E99 | −0.057 | −0.088 .. −0.027 | 0/9 | 0.004 |
| S4 − E99 | −0.053 | −0.074 .. −0.032 | 0/9 | 0.004 |

The same comparisons on the original unseeded scores: S4 − F4 +0.150 (9/9), S1 − F8 −0.034 (2/9).

E99's C2ST uses 10 sampling seeds per cell; the student and the F_k use 20.

How noisy C2ST is (`results/review_report_final.txt`, R4): re-splitting the same file gives a median sd of 0.0029; changing the sampling seed gives a median sd of 0.0046.

## 5. Downstream utility (`results/review_R2_utility_paired.csv`, `results/review_R2_utility_cells.csv`)

*What this section shows:* the practical question. If you train a model on the synthetic data, does it work as well on real data as one trained on the teacher's data? Higher is better. The differences are small in absolute terms, so judge them against their CIs and against the absolute levels listed under the first table.

AUROC of XGBoost trained on synthetic data and tested on the real held-out test split. 6 cells (adult and default), 10 sampling seeds, except adult s0's E99, which has 20; wins = student higher. Our MLE protocol differs from the authors' in several ways (`method.md` §6), so these values are not comparable with the TabSyn paper's.

| Comparison | Mean | 95 % CI | Wins | p |
|---|---|---|---|---|
| S1 − E99 | −0.0032 | −0.0060 .. −0.0003 | 1/6 | 0.22 |
| S2 − E99 | −0.0041 | −0.0066 .. −0.0017 | 0/6 | 0.031 |
| S4 − E99 | −0.0036 | −0.0070 .. −0.0002 | 1/6 | 0.22 |
| S1 − F8 | −0.0019 | −0.0050 .. +0.0013 | 1/6 | 0.22 |
| S1 − F4 | +0.0040 | +0.0008 .. +0.0072 | 5/6 | 0.22 |

Absolute AUROC: adult around 0.90–0.91, default around 0.75–0.76 (per cell in `review_R2_utility_cells.csv`).

How noisy AUROC is (`results/review_report_final.txt`, R2): within a cell, the sd over sampling seeds has a median of 0.0035 for the student and 0.0036 for E99. Across the three teachers of one table, E99's AUROC has an sd of 0.0014 on adult and 0.0042 on default.

news, R² on a log1p-transformed target. 3 cells. Real training data reaches 0.160; E99 reaches 0.122 / 0.138 / 0.114 by seed. This is R² on a log scale, not R² of the raw target. No F_k utility on the log1p target was computed for news, so there is no student-vs-F8 utility comparison on news. The news F_k files hold raw-target scores only (`results/newsv_s*_fast_extra.jsonl`), which cannot be compared with the log1p scores.

| Comparison | Mean | 95 % CI | Wins |
|---|---|---|---|
| S1 − E99 | −0.011 | −0.029 .. +0.008 | 0/3 |
| S2 − E99 | −0.018 | −0.028 .. −0.007 | 0/3 |
| S4 − E99 | −0.020 | −0.027 .. −0.013 | 0/3 |

## 6. DCR, memorisation check (`results/review_R3_dcr.csv`, `results/review_summary.json` → `R3`)

*What this section shows:* whether generated rows sit closer to the training data than to unseen data, a check for memorisation. A value near 0.5 means synthetic rows are no closer to the training data than to unseen data. The last two rows are reference points: validation data shows what "no copying" looks like, and training data compared with itself shows what "copying" looks like.

Fraction of synthetic rows closer to the training reference than to the test data; 0.5 is ideal and higher means closer to the training rows. Full record (target included), 5,000 synthetic rows, 10 seeds, mean over 9 cells. The training reference is a random subsample of train' the same size as the test set, so the "exact matches" column only counts matches with that subsample; copies of other training rows would not be counted.

| Generator | DCR | Exact matches with the training reference |
|---|---|---|
| S1 / S2 / S4 | 0.502 / 0.503 / 0.503 | ≤ 0.00002 |
| E99 | 0.505 | 0.00003 |
| F4 / F8 / F16 | 0.502 / 0.505 / 0.505 | ≤ 0.00003 |
| validation data vs training data | 0.492 | 0.0002 |
| training data vs itself | 0.631 | 0.265 (roughly the share of train' rows that land in the subsample; not a count of duplicate rows) |

Rows exactly equidistant to train and test, mean over cells: ≤ 0.0002 for every generator (up to 0.0004 in a single cell).

Distance from the ideal 0.5, student − E99, 9 cells (`results/review_report_final.txt`, R3): S1 −0.0013 (CI −0.0037 .. +0.0012), S2 −0.0016 (CI −0.0041 .. +0.0009), S4 −0.0019 (CI −0.0040 .. +0.0002).

## 7. Where the student's remaining error sits

*What this section shows:* §1 shows the student has more error than E99. This section breaks that extra error down, so that it can be described accurately instead of guessed at. Every number here is an **excess**: student error minus E99 error, on the same cell.

### 7a. Per column, SDMetrics column-shape error, S1 − E99 (`results/percol_report.txt`, `results/percol_columns.csv`)

*What this subsection shows:* the excess split by column, which answers three questions:

- Is it concentrated in a few columns or spread out? (the share column)
- Is one column type worse? (categorical vs numerical)
- Does the student go wrong in the same columns as the teacher does with few steps? (the Spearman line)

Each question has a threshold set in the project's ablation plan (`method.md` §8, row E5).

| Table | Share of the positive excess carried by the worst 20 % of columns | Mean excess, categorical vs numerical columns (pp) | Five worst columns (excess, pp) |
|---|---|---|---|
| adult | 46 % | +1.29 vs +1.47 | education.num (num) +4.02, occupation (cat) +3.33, education (cat) +2.13, capital.gain (num) +1.62, hours.per.week (num) +1.55 |
| default | 43 % | +1.63 vs +1.71 | PAY_0 (cat) +4.16, PAY_3 (cat) +3.80, PAY_AMT5 (num) +3.40, PAY_AMT2 (num) +3.38, PAY_AMT6 (num) +3.15 |
| news | 58 % | +2.60 vs +1.50 | kw_min_max +6.56, kw_min_min +6.03, max_positive_polarity +5.85, abs_title_subjectivity +5.70, num_videos +5.28 (all num) |

In the share, columns where the student is better than E99 count as zero (2 such columns on default, 10 on news), and "20 %" is rounded to whole columns (3, 5 and 10 columns). Using the net excess instead, the shares are 46 %, 44 % and 61 % (computed from `results/percol_columns.csv`).

Spearman correlation over columns between the student's excess (S1 − E99) and the 15-evaluation teacher's excess (Heun 8 steps − E99): adult 0.27, default 0.01, news −0.18.

### 7b. Column pairs (`results/review_summary.json` → `R5`; per seed in `results/review_R5_pairs.csv`)

*What this subsection shows:* the excess split by type of column pair, all scored with one common metric so the pair types can be compared. The "After IPF" column is a sensitivity check: it asks how much excess remains once each two-column synthetic table is adjusted to match the real per-column distributions. It is not a split of the error into a "per-column part" and a "relationship part"; IPF can even increase the error for a single pair. The SDMetrics column shows the library's own breakdown, which scores numerical-numerical pairs with a different metric (correlation similarity) from all other pairs (contingency similarity). Its pair-type pattern does not hold on the common metric, so it should not be used to compare pair types.

Excess error S1 − E99 in pp, mean over 3 seeds. "One metric" scores every pair with the same binned-contingency distance (10 bins at the real data's quantiles for numerical columns). "After IPF" repeats this after iterative proportional fitting (50 sweeps) of each two-column synthetic table to the real one-column distributions. "SDMetrics" is the library's own score.

The common-metric code had an error (fixed 11 Oct 2026): synthetic values outside the real data's categories or bins were handled wrongly. No such values occurred in any file scored here, so every number below is unaffected (`tools/eval_pairs_common.py`, header).

| Table | Pair type | One metric | After IPF | SDMetrics |
|---|---|---|---|---|
| adult | cat-cat | 2.57 | 1.15 | 2.57 |
| adult | num-cat | 2.68 | 0.85 | 2.89 |
| adult | num-num | 2.74 | 0.33 | 0.13 |
| default | cat-cat | 3.23 | 1.46 | 3.23 |
| default | num-cat | 2.92 | 1.18 | 1.08 |
| default | num-num | 5.70 | 4.58 | 0.70 |
| news | cat-cat | 2.99 | 0.36 | 2.99 |
| news | num-cat | 3.06 | 1.07 | 4.03 |
| news | num-num | 2.52 | 1.00 | 0.26 |

Pairs that include the target column versus pairs that don't, one metric (`results/review_report_final.txt`, R5): adult 2.04 vs 2.74, default 1.50 vs 4.11, news 1.46 vs 2.61.

### 7c. Tails (`results/review_summary.json` → `R5_tails`; per seed in `results/review_R5_tails.csv`)

*What this subsection shows:* whether the columns with the most excess are ones with extreme values (long tails), which generative models often handle badly. The "Excess" column shows how much worse the student is on each column. The last two columns measure tail error specifically. If tails were the problem, high excess would come with high tail error for the student but not for E99.

Marginal distance on common bins, mean over seeds. The q99 error is |synthetic 99th percentile − real 99th percentile| divided by the real interquartile range (by the standard deviation, or 1, when the interquartile range is 0). "Mass > real p99" is the share of synthetic rows above the real 99th percentile; a perfect generator gives about 0.01, or less when many rows share the value at the 99th percentile.

The columns below are the largest excesses on the common marginal metric. They are not the same list as the worst SDMetrics columns in §7a. Example: kw_min_max, the worst news column in §7a, has q99 error 0.32 for S1 and 6.56 for E99, and mass above p99 0.011 vs 0.008 (`results/review_R5_tails.csv`).

| Table | Column | S1 | E99 | Excess | q99 error S1 / E99 | Mass > real p99, S1 / E99 |
|---|---|---|---|---|---|---|
| news | abs_title_subjectivity | 14.42 | 5.68 | +8.74 | 0.000 / 0.000 | 0.000 / 0.000 |
| news | kw_min_min | 9.02 | 3.04 | +5.98 | 0.000 / 0.000 | 0.000 / 0.000 |
| news | kw_max_max | 7.60 | 2.15 | +5.44 | 0.000 / 0.000 | 0.000 / 0.000 |
| news | min_negative_polarity | 10.84 | 5.61 | +5.24 | 0.000 / 0.000 | 0.000 / 0.000 |
| news | title_sentiment_polarity | 8.45 | 3.44 | +5.02 | 0.000 / 0.000 | 0.000 / 0.000 |
| news | num_videos | 8.35 | 3.42 | +4.93 | 1.87 / 2.35 | 0.009 / 0.007 |
| adult | education.num | 8.89 | 2.83 | +6.05 | 0.19 / 0.01 | 0.000 / 0.000 |
| default | PAY_AMT3 | 9.50 | 4.63 | +4.87 | 3.45 / 3.28 | 0.007 / 0.007 |
| default | PAY_AMT2 | 7.96 | 3.39 | +4.56 | 1.40 / 2.30 | 0.009 / 0.009 |

0.000 means below 0.0005, not exactly zero: on news, kw_min_min has a mass above p99 of up to 0.00008 for S1 (`results/review_R5_tails.csv`).

## 8. Ablations (`results/ablation_report_all.txt` and `results/ablation_*_summary.csv`)

*What this section shows:* which parts of the method matter. Each row changes one thing in the student's recipe and keeps everything else fixed:

- CT: the training target uses no teacher step (the student still starts from the teacher's weights);
- N: use fewer or more noise levels;
- 40k: train for twice as long (and, as a follow-up, the same for CT).

A large positive number means that part of the recipe matters a lot. Rows marked "still improving" or run on 3 cells only carry less weight. The diagnostic in the notes is the held-out self-consistency loss that also selects the checkpoint: "left its early level at" is the first step where it fell below 80 % of its mean over the first 2,000 steps. That is a rough marker, not a convergence test. For the CT arms (CT, iCT-style, CT 40k) this loss is the CT loss itself, with a target that uses no teacher step, so it is a different quantity from the CD arms' loss: when CT and CD "left their early level" is not directly comparable.

Differences are ablation − headline student, shape pp, at 1 / 2 / 4 evaluations. Positive = the ablation is worse.

| Ablation | Cells | 1 eval | 2 evals | 4 evals | Wins (ablation better) | Notes |
|---|---|---|---|---|---|---|
| CT (teacher-initialised, no teacher step in the target), same recipe and number of updates | 9 | +5.97 | +5.03 | +4.54 | 0/9 at each | p = 0.004 at each. Left its early level at 14.75k–17.75k steps (CD: 9.25k–9.75k). |
| iCT-style CT (target EMA 0, 1/Δt weighting; two of the improved-CT changes, not the full recipe) | 3 (seed 0) | adult +6.85, default +8.23, news +5.24 | adult +6.68, default +8.13, news +2.69 | adult +6.52, default +7.28, news +2.24 | — | descriptive only |
| N = 8 grid points (instead of 18) | 3 (seed 0) | +1.52 | +1.85 | +1.03 | 0/3 | left its early level at 3.75k–4k; best checkpoint at the final 20k step in all 3 cells |
| N = 40 grid points | 3 (seed 0) | +15.10 | +0.39 | +0.05 | 0/3, 1/3, 1/3 | **still improving at 20k steps**: never fell below 80 % of its early level, and still falling 7.4–8.5 % over the last 2k steps |
| 40k training steps (instead of 20k) | 9 | −0.21 | −0.05 | −0.03 | 9/9, 6/9, 7/9 (p 0.004, 0.51, 0.18) | best checkpoint after 20k in 9/9. Continued from the 20k checkpoint with a fresh GPU random stream. |

CT continued to 40k updates (follow-up of 11 Oct; `results/ctlong_report.txt`, `results/ctlong_cells.csv`). *Whether CT only needed more training:* seed 0 of each table, continued from the 20k CT checkpoint in the same way as the 40k CD arm. Shape pp at 1 / 2 / 4 evaluations; positive = CT worse. Descriptive only: 3 cells, no test, no rule fixed in advance.

| Cell | CT 20k − CD | CT 40k − CD | CT 40k − CT 20k | Best checkpoint of the 40k run |
|---|---|---|---|---|
| adult s0 | +6.51 / +6.34 / +6.39 | +6.53 / +6.52 / +6.48 | +0.02 / +0.18 / +0.09 | step 37,750 |
| default s0 | +7.70 / +7.45 / +7.16 | +7.68 / +7.54 / +7.29 | −0.02 / +0.09 / +0.13 | step 36,750 |
| news s0 | +3.61 / +2.51 / +1.35 | +4.04 / +2.83 / +1.33 | +0.42 / +0.33 / −0.03 | step 40,000 (the last) |
| mean | +5.94 / +5.43 / +4.97 | +6.08 / +5.63 / +5.03 | +0.14 / +0.20 / +0.07 | |

Over the last 2,000 updates the held-out self-consistency loss (for CT, the CT loss; see the note above the table) changed by at most 0.12 % (adult up 0.05 %, default down 0.01 %, news down 0.12 %). This is a rough marker, not a convergence test.

## 9. Reproducibility (`results/review_R6_determinism.csv`)

*What this section shows:* whether regenerating the samples from the saved models gives the same scores. It is evidence that the stored results can be trusted, and it states exactly where they can't be reproduced to the last digit.

Two files per cell were regenerated and re-scored: one student file (1 evaluation) and one E99 file, both at sampling seed 0. Their shape scores matched the stored scores exactly in 7 of 9 cells. Differences in the other two: default s0, 0.052 pp (S1) and 0.000 (E99); default s1, 0.039 pp (S1) and 0.004 pp (E99). Only these scores were compared, not the samples row by row.

Beyond those two files, the later re-scorings show that on default s0 and s1 the regeneration is not exact for every file (computed by comparing `shape_err` in `results/<cell>_c2st_seeded.jsonl`, and the per-column scores in `results/<cell>_percol.jsonl`, with the originally stored scores):

- All 60 student files (S1, S2, S4 × 20 seeds) differ in both cells, by up to 0.17 pp per file. The 20-seed means shift by +0.008 / +0.021 / +0.022 pp (s0) and −0.004 / +0.008 / −0.007 pp (s1) for S1 / S2 / S4.
- On default s1 the 10 E99 files differ by up to 0.10 pp per file (mean −0.012 pp), and the 15-evaluation teacher files used in §7a by up to 0.21 pp.
- The tuned F_k files, and every file in the other 7 cells, regenerate exactly.

§4, §6 and §7 were scored on the regenerated samples, so for default s0 and s1 they describe samples slightly different from those behind §1–2. The 0.1 pp stop rule (`method.md` §8) was applied to the seed-0 pair only; several single files exceed 0.1 pp. No mean moves by more than 0.022 pp.

## 10. Compute (`results/sacct.txt`, `results/MANIFEST.json` → `slurm`)

*What this section shows:* how much GPU time the project used, and on what. It's for the methodology or limitations section.

GPU-hours of TabSyn jobs: 60.4 in all job states and 52.0 in completed jobs, up to 2 Oct. By job type: study 14.9, tuned-sampler search 8.9, ablations 8.9, post-review re-scoring 7.3, leak rows 5.0, recipe search 4.7, teacher curves (gate) 4.3, MLE/DCR 3.8, per-column analysis 1.3, pilots 1.0, low-noise check 0.3. The study plus ablations together come to 23.8. These are GPU allocation hours from the scheduler, including the time jobs spent sampling and scoring, not pure training time.

The follow-ups of 11 Oct added 1.1 GPU-hours (`results/sacct_followup.txt`): speed benchmark 0.1, CT to 40k updates 1.0.

## 11. The leak rows (`results/leak_delta_report.txt`, `results/seed_spread_report.txt`)

*What this section shows:* how much the teacher changes when its VAE is trained the authors' way (test data steering it) instead of ours (validation data). The authors' way was run for one seed only, so compare the difference with how much the teacher varies across our three seeds.

| Table | E99 shape, our setup, seeds 0 / 1 / 2 | E99 shape, authors' setup, seed 0 | C2ST, ours / authors' (seed 0) | DCR, predictor columns only, ours / authors' | Utility, ours / authors' |
|---|---|---|---|---|---|
| adult | 0.76 / 1.02 / 2.34 | 2.69 | 0.989 / 0.894 | 0.509 / 0.499 | AUROC 0.909 / 0.903 |
| default | 1.43 / 1.12 / 2.85 | 1.19 | 0.970 / 0.977 | 0.505 / 0.504 | AUROC 0.759 / 0.760 |
| news | 4.39 / 2.47 / 3.98 | 2.62 | 0.834 / 0.932 | 0.502 / 0.499 | R² (log) 0.122 / 0.135 |

E99 shape: 20 sampling seeds. C2ST, DCR and utility: 10 sampling seeds, except our adult seed-0 teacher, which has 20. The C2ST here is the original, unseeded score, not the seeded re-score in §4. The DCR here leaves out the target column, unlike §6.

## 12. Generation time (`results/bench_speed_report.txt`, `results/bench_speed_summary.csv`; per cell in `results/bench_speed_cells.csv`)

*What this section shows:* whether fewer network evaluations make generation faster in practice. NFE (§1) counts only network evaluations. Here we time the whole job of producing a synthetic table, including turning the latents back into rows, and set the time saved against the time it took to train the student.

Setup (follow-up of 11 Oct, `tools/bench_speed.py`): one NVIDIA A40. Each generator produces as many rows as train' has (24,300 to 32,148) in a single batch. One untimed warm-up, then 5 timed runs; medians are reported. "Network" is generating the latents; "decoding" is the VAE decoder, the inverse transforms and assembling the table (writing it to disk is not included). The F_k are the same settings as in §1.

Median over the 9 cells. Each column is a separate median, so network + decoding need not add up to the total.

| Generator | NFE | Network (ms) | Decoding (ms) | Total (ms) | Total, range over cells (ms) | Total ÷ S1, median (range) |
|---|---|---|---|---|---|---|
| S1 | 1 | 36 | 76 | 106 | 81 – 298 | 1 |
| S2 | 2 | 74 | 76 | 138 | 118 – 338 | 1.30 (1.13 – 1.46) |
| S4 | 4 | 151 | 77 | 204 | 196 – 426 | 1.92 (1.43 – 2.43) |
| F4 | 4 | 118 | 77 | 176 | 163 – 387 | 1.66 (1.30 – 2.02) |
| F6 | 6 | 179 | 77 | 227 | 223 – 454 | 2.14 (1.53 – 2.76) |
| F8 | 8 | 239 | 77 | 285 | 277 – 524 | 2.62 (1.76 – 3.50) |
| F16 | 16 (8 on news s0) | 478 | 77 | 524 | 482 – 796 | 4.55 (1.77 – 6.52) |
| E99 | 99 | 3,011 | 79 | 3,056 | 2,616 – 3,681 | 24.7 (12.4 – 37.9) |

Total time per table, ms, seeds 0 / 1 / 2:

| Table | S1 | F6 | E99 |
|---|---|---|---|
| adult | 82 / 81 / 82 | 224 / 223 / 225 | 3,039 / 3,056 / 3,096 |
| default | 106 / 106 / 106 | 228 / 227 / 227 | 2,641 / 2,620 / 2,616 |
| news | 292 / 294 / 298 | 451 / 452 / 454 | 3,677 / 3,669 / 3,681 |

Network time grows almost in proportion to NFE: E99's network time is 83–84 times S1's. Decoding takes the same time for every generator in a cell, about 45 ms on adult, 77 ms on default and 250 ms on news. It is 55 % (adult), 71 % (default) and 86 % (news) of the student's total time. That is why E99 takes 12–38 times as long as S1 rather than 99 times, and why the ratio is smallest on news. F16 on news s0 is an 8-evaluation setting: validation chose it for the 16 budget.

*Training cost and break-even.* Training the student (20,000 updates) took 367–739 s, i.e. 6–12 minutes (wall-clock of each run, including its periodic diagnostics; `student_train_s` in `results/bench_speed_cells.csv`, copied from each run's record). The two runs at 367 s (default s0 and s1) were twice as fast throughout with the same settings; the record does not name the GPU, so the cause is unknown. Not counted: the τ search, and the teacher and VAE, which every method needs. The F_k's own sampler search is not charged to them either (196 + 63 + 9 settings × 5 validation seeds per cell; the tuned-sampler jobs took 8.9 GPU-hours for the 9 cells, scoring included), so the break-even against F4 and F6 is stated in the baselines' favour.

Break-even = student training time ÷ (time per row of the baseline − time per row of S1), i.e. how many rows must be generated before the training time is paid back.

| Baseline | Break-even rows, median (range over cells) | The same in tables of this size |
|---|---|---|
| E99 | 7.0 million (3.5 – 7.1 million) | 145 – 285 |
| F6 | 148.5 million (73.5 – 150.8 million) | about 3,000 – 5,900 |
| F4 | 255.8 million (128.2 – 264.4 million) | about 5,300 – 10,100 |

The files also list F8 and F16, but those have lower error than the student (§1), so a time trade against them is not like for like.

Scope: one GPU type, one batch per table, this model size. Smaller batches, generation on CPU or larger networks were not measured and could change the ratios.
