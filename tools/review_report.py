"""Post-review analysis (1 Oct 2026): every number in docs/tabsyn-study-verdict-v3.md comes from here.

  python tools/review_report.py [--datasets adult default news] [--seeds 0 1 2]
      prints sections R1-R10, writes results/review_*.csv and results/review_summary.json

Reads (per cell A = <d>v_s<k>): _student_study_s<k>.jsonl, _fast.jsonl + _fast_choice.json, _fastlo.jsonl + _fastlo_choice.json,
_fast6.jsonl + _fast6_choice.json, _gate_summary.csv, _student_study_s<k>_extra.jsonl, _teacher_extra.jsonl, _fast_extra.jsonl,
_mle_log.jsonl, _c2st_seeded.jsonl, _dcr_full.jsonl, _dcr_feat.jsonl, _pairs_common.jsonl, _cols_common.jsonl, _percol_pairs.jsonl,
_percol_determinism.json, _rescore_determinism.json, logs/percol_check_<A>.jsonl; tabsyn/vae/ckpt/<A>/train_z.npy.
Sections whose inputs are missing are skipped with a note. Statistics: paired per cell, exact two-sided sign test over the
9 cells (conditional on these three tables), per-dataset means (n = 3 training seeds), t-based 95 % CI over cells.
"""
import argparse, glob, json, math, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.chdir(ROOT); sys.path.insert(0, ROOT)
import numpy as np, pandas as pd
from scipy import stats as sst
from tools.stats import sign_test
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 60); pd.set_option('display.max_rows', 400)

ap = argparse.ArgumentParser(); ap.add_argument('--datasets', nargs='+', default=['adult', 'default', 'news']); ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
a = ap.parse_args()
DS, SEEDS, NFES = a.datasets, a.seeds, [1, 2, 4]
TREND_EXCLUDED = {'default'}
OUT = {}


def jl(p): return pd.DataFrame([json.loads(l) for l in open(p) if l.strip()]) if os.path.exists(p) else pd.DataFrame()
def jf(p): return json.load(open(p)) if os.path.exists(p) else {}
def gc(D, name): return D[name] if name in D else pd.Series([np.nan] * len(D), index=D.index)
def cfg_mask(df, c): return (df.solver == c['solver']) & (df.steps == c['steps']) & (np.isclose(df.sigma_max.astype(float), float(c['sigma_max']))) & (df.rho == c['rho'])
def ci95(v):
    v = np.asarray([x for x in v if x == x], float); n = len(v)
    if n < 2: return (float('nan'), float('nan'))
    h = sst.t.ppf(0.975, n - 1) * v.std(ddof=1) / math.sqrt(n); return (float(v.mean() - h), float(v.mean() + h))
def paired(label, diffs, lower_better=True):
    v = np.asarray([x for x in diffs if x == x], float)
    if not len(v): return None
    lo, hi = ci95(v)
    return {'comparison': label, 'cells': int(len(v)), 'mean': float(v.mean()), 'sd_cells': float(v.std(ddof=1)) if len(v) > 1 else float('nan'),
            'ci95_lo': lo, 'ci95_hi': hi, 'min': float(v.min()), 'max': float(v.max()),
            'wins': int((v < 0).sum()) if lower_better else int((v > 0).sum()), 'sign_p': sign_test(v)}
def show(rows, title):
    if not rows: print(f'  ({title}: nothing to show)'); return pd.DataFrame()
    R = pd.DataFrame([r for r in rows if r]); print(f'\n-- {title}'); print(R.to_string(index=False, float_format=lambda x: f'{x:.4f}')); return R


cells = {}
for d in DS:
    for k in SEEDS:
        A = f'{d}v_s{k}'
        c = dict(A=A, d=d, k=k, S=jl(f'results/{A}_student_study_s{k}.jsonl'), F=jl(f'results/{A}_fast.jsonl'), Fc=jf(f'results/{A}_fast_choice.json'),
                 Flo=jl(f'results/{A}_fastlo.jsonl'), Floc=jf(f'results/{A}_fastlo_choice.json'), F6=jl(f'results/{A}_fast6.jsonl'), F6c=jf(f'results/{A}_fast6_choice.json'),
                 G=pd.read_csv(f'results/{A}_gate_summary.csv') if os.path.exists(f'results/{A}_gate_summary.csv') else pd.DataFrame(),
                 Sx=jl(f'results/{A}_student_study_s{k}_extra.jsonl'), Tx=jl(f'results/{A}_teacher_extra.jsonl'), Fx=jl(f'results/{A}_fast_extra.jsonl'),
                 Lx=jl(f'results/{A}_mle_log.jsonl'), C2=jl(f'results/{A}_c2st_seeded.jsonl'), Dfull=jl(f'results/{A}_dcr_full.jsonl'), Dfeat=jl(f'results/{A}_dcr_feat.jsonl'),
                 Pc=jl(f'results/{A}_pairs_common.jsonl'), Cc=jl(f'results/{A}_cols_common.jsonl'), Pp=jl(f'results/{A}_percol_pairs.jsonl'))
        if not len(c['S']): print(f'!! {A}: no study student results; cell skipped'); continue
        cells[(d, k)] = c


def student(c, nfe, col='shape_err', seeds=None):
    s = c['S'][c['S'].nfe == nfe]
    if seeds is not None: s = s[s.seed.isin(seeds)]
    return s[col].mean() if col in s else float('nan')


def fair(c, k, col='shape_err', seeds=None, lo=False):
    """F_k on train' (20 seeds). lo=True: the sub-1 sigma_max replacement where it beat the E1 choice on val (k in {1, 2})."""
    if lo:
        ch = c['Floc'].get(str(k));
        if not ch or not ch.get('beats_E1_choice') or not len(c['Flo']): return float('nan')
        f = c['Flo'][cfg_mask(c['Flo'], ch)]
    elif k == 6:
        ch = c['F6c'].get('6')
        if not ch or not len(c['F6']): return float('nan')
        f = c['F6'][cfg_mask(c['F6'], ch)]
    else:
        ch = c['Fc'].get(str(k))
        if not ch or not len(c['F']): return float('nan')
        f = c['F'][cfg_mask(c['F'], ch)]
    if seeds is not None: f = f[f.seed.isin(seeds)]
    return f[col].mean() if (len(f) and col in f) else float('nan')


def e99(c, col='shape_mean', churn=0):
    G = c['G']; r = G[(G.solver == 'heun') & (G.steps == 50) & (G.s_churn == churn) & (G.sigma_max == 80) & (G.rho == 7)]
    return float(r[col].iloc[0]) if len(r) else float('nan')


# ================================================================== R1: paired vs F_k incl. sub-1 and measured F_6
print('=== R1. Student vs val-tuned teacher F_k on train\' (shape pp; negative = student better; 20 sampling seeds; 9 cells)')
rows1, per_cell = [], []
for (d, k), c in cells.items():
    rec = dict(dataset=d, seed=k, E99=e99(c), E99_churn1=e99(c, churn=1))
    for n in NFES: rec[f'S{n}'] = student(c, n); rec[f'S{n}_trend'] = student(c, n, 'trend_err'); rec[f'S{n}_c2st'] = student(c, n, 'c2st')
    for kk in (1, 2, 4, 6, 8, 16): rec[f'F{kk}'] = fair(c, kk); rec[f'F{kk}_trend'] = fair(c, kk, 'trend_err'); rec[f'F{kk}_c2st'] = fair(c, kk, 'c2st')
    for kk in (1, 2): rec[f'F{kk}lo'] = fair(c, kk, lo=True); rec[f'F{kk}lo_sigma'] = c['Floc'].get(str(kk), {}).get('sigma_max') if c['Floc'].get(str(kk), {}).get('beats_E1_choice') else None
    rec['F1_best'] = rec['F1lo'] if rec['F1lo'] == rec['F1lo'] else rec['F1']; rec['F2_best'] = rec['F2lo'] if rec['F2lo'] == rec['F2lo'] else rec['F2']   # 11 Oct 2026: the sub-1 setting is used whenever it won on validation (Flo is NaN otherwise); was the minimum of the two final scores, which gave the same values here
    for kk in (4, 6, 8, 16): rec[f'F{kk}_cfg'] = (c['F6c'] if kk == 6 else c['Fc']).get(str(kk), {})
    rec['F6_nfe'] = c['F6c'].get('6', {}).get('nfe')
    per_cell.append(rec)
PC = pd.DataFrame(per_cell); PC.to_csv('results/review_cells.csv', index=False)
print(PC[['dataset', 'seed', 'S1', 'S2', 'S4', 'F1', 'F1lo', 'F2', 'F2lo', 'F4', 'F6', 'F6_nfe', 'F8', 'F16', 'E99', 'E99_churn1']].to_string(index=False, float_format=lambda x: f'{x:.2f}'))
comps = [('S1-F1 (E1 grid)', PC.S1 - PC.F1), ('S1-F1_best (incl. sub-1 sigma_max where it won on val)', PC.S1 - PC.F1_best), ('S2-F2_best', PC.S2 - PC.F2_best),
         ('S4-F4', PC.S4 - PC.F4), ('S1-F4', PC.S1 - PC.F4), ('S1-F6 (measured)', PC.S1 - PC.F6), ('S1-F8', PC.S1 - PC.F8), ('S1-F16', PC.S1 - PC.F16),
         ('S1-E99', PC.S1 - PC.E99), ('S2-E99', PC.S2 - PC.E99), ('S4-E99', PC.S4 - PC.E99), ('S1-E99_churn1', PC.S1 - PC.E99_churn1), ('E99_churn1-E99_churn0', PC.E99_churn1 - PC.E99),
         ('S1_trend-F4_trend (6 cells)', (PC.S1_trend - PC.F4_trend).where(~PC.dataset.isin(TREND_EXCLUDED))), ('S1_trend-F8_trend (6 cells)', (PC.S1_trend - PC.F8_trend).where(~PC.dataset.isin(TREND_EXCLUDED)))]
R1 = show([paired(l, v) for l, v in comps], 'paired over 9 cells'); R1.to_csv('results/review_R1_paired.csv', index=False)
OUT['R1'] = R1.to_dict('records')
print('\n-- per-dataset means (n = 3 training seeds each; the unit the sign test cannot see)')
PD = PC.assign(**{'S1-F4': PC.S1 - PC.F4, 'S1-F6': PC.S1 - PC.F6, 'S1-F8': PC.S1 - PC.F8, 'S1-E99': PC.S1 - PC.E99, 'S4-F4': PC.S4 - PC.F4}).groupby('dataset')[['S1-F4', 'S1-F6', 'S1-F8', 'S1-E99', 'S4-F4']].agg(['mean', 'min', 'max'])
print(PD.to_string(float_format=lambda x: f'{x:.2f}'))
OUT['R1_per_dataset'] = {d: {c: float(PD.loc[d, (c, 'mean')]) for c in ['S1-F4', 'S1-F6', 'S1-F8', 'S1-E99', 'S4-F4']} for d in PD.index}
print('  dataset-level all-same-sign exact sign test, n = 3: two-sided p = %.3f' % (2 / 2 ** 3))
# sensitivity: sampling seeds 5-19 only (seeds 0-4 were the val-search seeds for tau / F_k selection)
sens = []
for (d, k), c in cells.items():
    s1, s4 = student(c, 1, seeds=range(5, 20)), student(c, 4, seeds=range(5, 20))
    sens.append(dict(dataset=d, seed=k, **{'S1-F4': s1 - fair(c, 4, seeds=range(5, 20)), 'S1-F8': s1 - fair(c, 8, seeds=range(5, 20)), 'S4-F4': s4 - fair(c, 4, seeds=range(5, 20))}))
SE = pd.DataFrame(sens)
R1b = show([paired(f'{col} [sampling seeds 5-19 only]', SE[col]) for col in ('S1-F4', 'S1-F8', 'S4-F4')], 'fresh-sampling-seed sensitivity'); OUT['R1_seeds5to19'] = R1b.to_dict('records')
# sub-1 sigma_max val curves (grid-edge objection)
print('\n-- F_1 val shape vs sigma_max (5 val seeds; the E1 grid edge was sigma_max = 1)')
for (d, k), c in cells.items():
    ch = c['Floc'].get('1', {})
    if ch: print(f'  {d} s{k}: ' + ', '.join(f'{s}:{v:.2f}' for s, v in sorted(ch.get('val_shape_vs_sigma_max', {}).items(), key=lambda t: float(t[0]))) + f'  -> best sigma_max {ch["sigma_max"]} (beats E1: {ch["beats_E1_choice"]})')

# ================================================================== R2: utility effect sizes with uncertainty
print('\n=== R2. Utility: effect sizes with uncertainty (not equivalence tests; margins are the group\'s to justify)')
rows2 = []
for (d, k), c in cells.items():
    Sx, Tx, Fx, Lx = c['Sx'], c['Tx'], c['Fx'], c['Lx']
    t99 = Tx[(Tx.solver == 'heun') & (Tx.steps == 50)] if len(Tx) else pd.DataFrame()
    for n in NFES:
        sx = Sx[Sx.nfe == n] if len(Sx) else pd.DataFrame()
        rec = dict(dataset=d, seed=k, nfe=n)
        if d != 'news' and len(sx) and 'mle_roc_auc' in sx:
            rec.update(auroc_s=sx.mle_roc_auc.mean(), auroc_s_sd=sx.mle_roc_auc.std(ddof=1), auroc_e99=t99.mle_roc_auc.mean(), auroc_e99_sd=t99.mle_roc_auc.std(ddof=1))
            for kk in (4, 8, 16):
                ch = c['Fc'].get(str(kk))
                if ch and len(Fx):
                    f = Fx[cfg_mask(Fx, ch)]; rec[f'auroc_F{kk}'] = f.mle_roc_auc.mean() if len(f) else np.nan
        if d == 'news' and len(Lx) and 'mle_r2_log' in Lx:
            ls = Lx[(Lx.solver == 'cm') & (Lx.nfe == n)]; lt = Lx[(Lx.solver == 'heun') & (Lx.steps == 50)]
            rec.update(r2log_s=ls.mle_r2_log.mean(), r2log_s_sd=ls.mle_r2_log.std(ddof=1), r2log_e99=lt.mle_r2_log.mean(), r2log_e99_sd=lt.mle_r2_log.std(ddof=1),
                       r2log_real=float(Lx[Lx.file == 'real.csv'].mle_r2_log.iloc[0]) if (Lx.file == 'real.csv').any() else np.nan)
        rows2.append(rec)
R2 = pd.DataFrame(rows2); R2.to_csv('results/review_R2_utility_cells.csv', index=False)
print(R2.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
cls = R2[R2.dataset != 'news']; reg = R2[R2.dataset == 'news']
r2rows = []
for n in NFES:
    g = cls[cls.nfe == n]
    if len(g) and 'auroc_s' in g:
        r2rows.append(paired(f'AUROC S{n}-E99 (6 cells)', g.auroc_s - g.auroc_e99, lower_better=False))
        if f'auroc_F8' in g: r2rows.append(paired(f'AUROC S{n}-F8 (6 cells)', g.auroc_s - g.auroc_F8, lower_better=False))
        if f'auroc_F4' in g: r2rows.append(paired(f'AUROC S{n}-F4 (6 cells)', g.auroc_s - g.auroc_F4, lower_better=False))
    h = reg[reg.nfe == n]
    if len(h) and 'r2log_s' in h: r2rows.append(paired(f'news R2(log1p) S{n}-E99 (3 cells)', h.r2log_s - h.r2log_e99, lower_better=False))
R2s = show(r2rows, 'paired utility differences: mean, sd over cells, 95% t-CI, range, wins, sign p'); R2s.to_csv('results/review_R2_utility_paired.csv', index=False)
OUT['R2'] = R2s.to_dict('records')
if len(cls) and 'auroc_s_sd' in cls: print(f'  within-cell sampling-seed sd of AUROC (10 seeds): student median {cls.auroc_s_sd.median():.4f}, E99 median {cls.auroc_e99_sd.median():.4f}; E99 seed spread across training seeds per dataset: ' +
      ', '.join(f'{d} {cls[cls.dataset == d].drop_duplicates("seed").auroc_e99.std(ddof=1):.4f}' for d in cls.dataset.unique()))
if len(reg) and 'r2log_s_sd' in reg: print(f'  news R2(log1p): real {reg.r2log_real.iloc[0]:.3f}; within-cell sd student median {reg.r2log_s_sd.median():.4f}, E99 median {reg.r2log_e99_sd.median():.4f}')

# ================================================================== R3: DCR, feature space vs full record
print('\n=== R3. DCR (fraction of synthetic rows closer to train\' than to the shipped test split; 0.5 ideal)')
rows3 = []
for (d, k), c in cells.items():
    for space, D, suf in (('features', c['Dfeat'], ''), ('full', c['Dfull'], '_full')):
        if not len(D): continue
        def pick(mask, lab):
            x = D[mask]
            if not len(x): return
            rows3.append(dict(dataset=d, seed=k, space=space, what=lab, n=int(len(x)), dcr=x[f'dcr_score{suf}'].mean(), ties=x[f'dcr_ties{suf}'].mean(), copies=x[f'dcr_exact_copies{suf}'].mean(),
                              med_train=x[f'dcr_median_to_train{suf}'].mean(), med_test=x[f'dcr_median_to_test{suf}'].mean()))
        pick(D.file == 'real.csv', 'real(train\')'); pick(D.file == 'val.csv', 'val')
        for n in NFES: pick((gc(D, 'solver') == 'cm') & (gc(D, 'nfe') == n), f'S{n}')
        pick((gc(D, 'solver') == 'heun') & (gc(D, 'steps') == 50), 'E99')
        for kk in (4, 8, 16):
            ch = c['Fc'].get(str(kk))
            if ch and 'solver' in D: pick(cfg_mask(D, ch) & (gc(D, 'tag') == 'sel'), f'F{kk}')
R3 = pd.DataFrame(rows3)
if len(R3):
    R3.to_csv('results/review_R3_dcr.csv', index=False)
    piv = R3.pivot_table(index=['dataset', 'seed', 'what'], columns='space', values=['dcr', 'ties', 'copies']); print(piv.to_string(float_format=lambda x: f'{x:.4f}'))
    summ = R3.groupby(['space', 'what'])[['dcr', 'ties', 'copies']].mean(); print('\n-- mean over cells'); print(summ.to_string(float_format=lambda x: f'{x:.4f}'))
    OUT['R3'] = {f'{s}|{w}': {m: float(summ.loc[(s, w), m]) for m in ('dcr', 'ties', 'copies')} for s, w in summ.index}
    for n in NFES:
        for space in ('features', 'full'):
            s_ = R3[(R3.space == space) & (R3.what == f'S{n}')].set_index(['dataset', 'seed']).dcr; t_ = R3[(R3.space == space) & (R3.what == 'E99')].set_index(['dataset', 'seed']).dcr
            j = s_.index.intersection(t_.index)
            if len(j): r = paired(f'|DCR-0.5| S{n} - E99 [{space}]', (s_[j] - 0.5).abs() - (t_[j] - 0.5).abs()); print('  ', {k_: (round(v, 4) if isinstance(v, float) else v) for k_, v in r.items()})
else: print('  (no DCR re-scoring files yet)')

# ================================================================== R4: seeded C2ST
print('\n=== R4. C2ST with seeded CV splits (3 splits per file; c2st_mean) vs the historical unseeded values')
rows4 = []
for (d, k), c in cells.items():
    C2 = c['C2']
    if not len(C2): continue
    def grp(mask, lab):
        x = C2[mask]
        if len(x): rows4.append(dict(dataset=d, seed=k, what=lab, n=int(len(x)), c2st=x.c2st_mean.mean(), split_sd=x.c2st_sd.mean(), seed_sd=x.c2st_mean.std(ddof=1) if len(x) > 1 else np.nan))
    for n in NFES: grp((C2.solver == 'cm') & (C2.nfe == n), f'S{n}')
    grp((C2.solver == 'heun') & (C2.steps == 50), 'E99')
    for kk in (4, 8, 16):
        ch = c['Fc'].get(str(kk))
        if ch: grp(cfg_mask(C2, ch) & (gc(C2, 'tag') == 'sel'), f'F{kk}')
    ch6 = c['F6c'].get('6')
    if ch6 and len(c['F6']) and 'c2st_mean' in c['F6']:
        x = c['F6'][cfg_mask(c['F6'], ch6)]; rows4.append(dict(dataset=d, seed=k, what='F6', n=int(len(x)), c2st=x.c2st_mean.mean(), split_sd=x.c2st_sd.mean(), seed_sd=x.c2st_mean.std(ddof=1)))
R4 = pd.DataFrame(rows4)
if len(R4):
    R4.to_csv('results/review_R4_c2st.csv', index=False)
    W = R4.pivot_table(index=['dataset', 'seed'], columns='what', values='c2st')
    print(W.to_string(float_format=lambda x: f'{x:.4f}'))
    print(f'  CV-split sd (within file, 3 splits) median {R4.split_sd.median():.4f}; sampling-seed sd (within cell) median {R4.seed_sd.median():.4f}')
    r4 = []
    for lab, aa, bb in (('S4-F4', 'S4', 'F4'), ('S1-F8', 'S1', 'F8'), ('S1-F6', 'S1', 'F6'), ('S1-E99', 'S1', 'E99'), ('S2-E99', 'S2', 'E99'), ('S4-E99', 'S4', 'E99')):
        if aa in W and bb in W: r4.append(paired(f'C2ST {lab} [seeded] (higher = better)', W[aa] - W[bb], lower_better=False))
    # historical, unseeded values for the same comparisons
    r4.append(paired('C2ST S4-F4 [historical unseeded]', PC.S4_c2st - PC.F4_c2st, lower_better=False)); r4.append(paired('C2ST S1-F8 [historical unseeded]', PC.S1_c2st - PC.F8_c2st, lower_better=False))
    R4s = show(r4, 'paired C2ST'); R4s.to_csv('results/review_R4_c2st_paired.csv', index=False); OUT['R4'] = R4s.to_dict('records')
else: print('  (no seeded C2ST files yet)')

# ================================================================== R5: pairs on one metric, and tails
print('\n=== R5. Pair excess by type on ONE metric (binned contingency TVD, real-defined 10 quantile bins) vs SDMetrics\' mixed metrics')
rows5, rows5b = [], []
for (d, k), c in cells.items():
    Pc, Pp = c['Pc'], c['Pp']
    if len(Pc):
        def lab(r):
            if r.get('solver') == 'cm': return f'S{int(r["nfe"])}'
            if r.get('solver') == 'heun' and r.get('steps') == 50: return 'E99'
            for kk in (4, 8):
                ch = c['Fc'].get(str(kk))
                if ch and r.get('solver') == ch['solver'] and r.get('steps') == ch['steps'] and float(r.get('sigma_max')) == float(ch['sigma_max']) and r.get('rho') == ch['rho']: return f'F{kk}'
            return None
        Pc = Pc.assign(what=Pc.apply(lab, axis=1)); Pc = Pc[Pc.what.notna()]
        m = Pc.groupby(['what', 'pair_type'])[['err', 'err_dep']].mean().unstack('what')
        for pt in m.index:
            rec = dict(dataset=d, seed=k, pair_type=pt, n_pairs=int((Pc[(Pc.what == 'E99')].pair_type == pt).sum() / max(1, Pc[Pc.what == 'E99'].file.nunique())))
            for w in ('S1', 'S4', 'F4', 'F8', 'E99'):
                if ('err', w) in m: rec[f'{w}_err'] = m.loc[pt, ('err', w)]; rec[f'{w}_dep'] = m.loc[pt, ('err_dep', w)]
            if 'S1_err' in rec and 'E99_err' in rec: rec['S1-E99_err'] = rec['S1_err'] - rec['E99_err']; rec['S1-E99_dep'] = rec['S1_dep'] - rec['E99_dep']
            if 'F4_err' in rec and 'E99_err' in rec: rec['F4-E99_err'] = rec['F4_err'] - rec['E99_err']
            rows5.append(rec)
        # target involvement under the common metric
        for inv in (True, False):
            x = Pc[Pc.involves_target == inv].groupby('what').err.mean()
            if 'S1' in x and 'E99' in x: rows5b.append(dict(dataset=d, seed=k, involves_target=inv, metric='common', **{'S1-E99': x['S1'] - x['E99']}))
    if len(Pp):   # SDMetrics-based (E5) pair excess by type, same cells, for comparison
        def lab2(r):
            if r.get('solver') == 'cm': return f'S{int(r["nfe"])}'
            if r.get('solver') == 'heun' and r.get('steps') == 50: return 'E99'
            return None
        Pp = Pp.assign(what=Pp.apply(lab2, axis=1)); Pp = Pp[Pp.what.notna()]
        m2 = Pp.groupby(['what', 'pair_type']).err.mean().unstack('what')
        for pt in m2.index:
            if 'S1' in m2 and 'E99' in m2:
                rows5.append(dict(dataset=d, seed=k, pair_type=pt + ' [SDMetrics]', **{'S1_err': m2.loc[pt, 'S1'], 'E99_err': m2.loc[pt, 'E99'], 'S1-E99_err': m2.loc[pt, 'S1'] - m2.loc[pt, 'E99']}))
R5 = pd.DataFrame(rows5)
if len(R5):
    R5.to_csv('results/review_R5_pairs.csv', index=False)
    print(R5.groupby(['dataset', 'pair_type'])[[c_ for c_ in R5.columns if c_.endswith('_err') or c_.endswith('_dep')]].mean().to_string(float_format=lambda x: f'{x:.2f}'))
    OUT['R5'] = R5.groupby(['dataset', 'pair_type'])[['S1-E99_err', 'S1-E99_dep']].mean().reset_index().to_dict('records') if 'S1-E99_dep' in R5 else []
    if rows5b: print('\n-- target involvement (common metric, S1-E99 err):'); print(pd.DataFrame(rows5b).groupby(['dataset', 'involves_target'])['S1-E99'].mean().to_string(float_format=lambda x: f'{x:.2f}'))
else: print('  (no common-metric pair files yet)')
print('\n-- tails: numerical columns, S1 vs E99 (mean over 10 seeds): quantile error at p99 (in IQR units) and synthetic mass above the real p99 (0.01 ideal)')
rows5t = []
for (d, k), c in cells.items():
    Cc = c['Cc']
    if not len(Cc): continue
    Cc = Cc[Cc.col_type == 'numerical']
    s1 = Cc[(Cc.solver == 'cm') & (Cc.nfe == 1)].groupby('column')[['err_tv', 'qerr_0.99', 'qerr_0.01', 'mass_hi99', 'mass_lo1', 'max_ratio']].mean()
    t = Cc[(Cc.solver == 'heun') & (Cc.steps == 50)].groupby('column')[['err_tv', 'qerr_0.99', 'qerr_0.01', 'mass_hi99', 'mass_lo1', 'max_ratio']].mean()
    j = s1.index.intersection(t.index)
    for col in j: rows5t.append(dict(dataset=d, seed=k, column=col, S1_tv=s1.loc[col, 'err_tv'], E99_tv=t.loc[col, 'err_tv'], S1_q99=s1.loc[col, 'qerr_0.99'], E99_q99=t.loc[col, 'qerr_0.99'],
                                     S1_hi99=s1.loc[col, 'mass_hi99'], E99_hi99=t.loc[col, 'mass_hi99'], S1_maxr=s1.loc[col, 'max_ratio'], E99_maxr=t.loc[col, 'max_ratio']))
R5t = pd.DataFrame(rows5t)
if len(R5t):
    R5t.to_csv('results/review_R5_tails.csv', index=False)
    agg = R5t.groupby(['dataset', 'column']).mean(numeric_only=True).drop(columns='seed'); agg['excess_tv'] = agg.S1_tv - agg.E99_tv
    for d in DS:
        if d in agg.index.get_level_values(0):
            g = agg.loc[d].sort_values('excess_tv', ascending=False).head(6)
            print(f'  {d}: top columns by S1-E99 marginal TVD excess'); print(g[['excess_tv', 'S1_q99', 'E99_q99', 'S1_hi99', 'E99_hi99', 'S1_maxr', 'E99_maxr']].to_string(float_format=lambda x: f'{x:.3f}'))
    OUT['R5_tails'] = {d: agg.loc[d].sort_values('excess_tv', ascending=False).head(6).reset_index().to_dict('records') for d in DS if d in agg.index.get_level_values(0)}

# ================================================================== R6-R9: determinism, latent facts, tuning disclosure, churn
print('\n=== R6. Determinism records (|re-scored - stored| shape, pp) for every cell')
rows6 = []
for (d, k), c in cells.items():
    A = c['A']; rec = dict(dataset=d, seed=k)
    chk = f'logs/percol_check_{A}.jsonl'
    pdj = jf(f'results/{A}_percol_determinism.json')                       # 11 Oct 2026: shipped record, used when the job log is absent
    if not os.path.exists(chk) and pdj:                                      # two record formats: {file: {abs_diff_pp}} and {diffs_pp: {file: v}}
        items = pdj['diffs_pp'].items() if 'diffs_pp' in pdj else [(f, v['abs_diff_pp']) for f, v in pdj.items() if isinstance(v, dict) and 'abs_diff_pp' in v]
        for f, v in items: rec['percol_' + ('S1' if f.startswith('cm') else 'E99')] = v
    if os.path.exists(chk):
        new = {r['file']: r['shape_err'] for r in map(json.loads, open(chk))}
        for f, p in (('cm_N1_c0_smax80_rho7_sel_s0.csv', f'results/{A}_student_study_s{k}.jsonl'), ('heun_N50_c0_smax80_rho7_s0.csv', f'results/{A}_gate.jsonl')):
            st = next((r['shape_err'] for r in map(json.loads, open(p)) if r['file'] == f), None)
            if f in new and st is not None: rec['percol_' + ('S1' if f.startswith('cm') else 'E99')] = abs(new[f] - st)
    rr = jf(f'results/{A}_rescore_determinism.json')
    for f, v in rr.items(): rec['rescore_' + ('S1' if f.startswith('cm') else 'E99')] = v['abs_diff_pp']
    rows6.append(rec)
R6 = pd.DataFrame(rows6); print(R6.to_string(index=False, float_format=lambda x: f'{x:.2e}')); R6.to_csv('results/review_R6_determinism.csv', index=False); OUT['R6'] = R6.to_dict('records')
for (d, k), c in cells.items():   # write the per-cell determinism json where missing (adult cells had none)
    p = f'results/{c["A"]}_percol_determinism.json'; r = R6[(R6.dataset == d) & (R6.seed == k)].iloc[0]
    if not os.path.exists(p) and 'percol_S1' in r and r.percol_S1 == r.percol_S1:
        json.dump({'cm_N1_c0_smax80_rho7_sel_s0.csv': {'abs_diff_pp': float(r.percol_S1)}, 'heun_N50_c0_smax80_rho7_s0.csv': {'abs_diff_pp': float(r.percol_E99)}, 'source': f'logs/percol_check_{c["A"]}.jsonl (25 Sep 2026)'}, open(p, 'w'), indent=1)
        print(f'  wrote {p} from the job log')

print('\n=== R7. Latent space facts (from the VAE checkpoints actually used)')
OUT['R7'] = {}
for d in DS:
    p = f'tabsyn/vae/ckpt/{d}v_s0/train_z.npy'
    if os.path.exists(p):
        z = np.load(p); zf = z.reshape(z.shape[0], -1); zs = (zf - zf.mean(0)) / 2
        info = jf(f'data/{d}v_s0/info.json'); ncol = len(info.get('num_col_idx', [])) + len(info.get('cat_col_idx', [])) + len(info.get('target_col_idx', []))
        OUT['R7'][d] = dict(tokens=int(z.shape[1]), token_dim=int(z.shape[2]), latent_dim=int(zf.shape[1]), columns=ncol, raw_var=float(zf.var()), scaled_std=float(zs.std()), scaled_std_min=float(zs.std(0).min()), scaled_std_max=float(zs.std(0).max()))
        print(f'  {d}: {ncol} columns -> {z.shape[1]} tokens x {z.shape[2]} = {zf.shape[1]}-d latent (CLS token + one per column); raw var {zf.var():.3f}; after (z-mean)/2: std {zs.std():.3f} (per-dim {zs.std(0).min():.3f}..{zs.std(0).max():.3f}) vs sigma_data = 0.5')

print('\n=== R8. Tuning-budget disclosure')
arms = sorted(set(os.path.basename(p).replace('adultv_s0_student_', '').replace('_DONE', '') for p in glob.glob('results/adultv_s0_student_*_DONE') if not p.endswith('EXTRA_DONE')))
recipe_arms = [x for x in arms if x not in ('study_s0', 'ct_s0', 'ict_s0', 'N8_s0', 'N40_s0', 'long_s0', 'ctlong_s0')]   # ctlong: 11 Oct follow-up
print(f'  student recipe arms trained on adult s0 before the recipe was frozen (pilot + sweeps): {len(recipe_arms)}: {recipe_arms}')
print(f'  ablation arms on adult s0 (after the freeze): ct, ict, N8, N40, long, and the 11 Oct follow-up ctlong')
nv = {c['A']: (len(c['F']) and jf(f'results/{c["A"]}_fast_choice.json')) for c in cells.values()}
print('  teacher sampler search per cell: E1 196 configs (+ 9 sub-1 configs, + 63 six-NFE configs) x 5 val seeds; student tau search per cell: 1 / 7 / 13 schedules at 1 / 2 / 4 NFE x 5 val seeds')
OUT['R8'] = {'recipe_arms_adult_s0': recipe_arms, 'n_recipe_arms': len(recipe_arms)}

print('\n=== R9. Churn: the authors\' shipped sampler uses S_churn = 1; E99 here is churn 0 (deterministic). Shape, 20 seeds')
print(PC[['dataset', 'seed', 'E99', 'E99_churn1']].assign(diff=PC.E99_churn1 - PC.E99).to_string(index=False, float_format=lambda x: f'{x:.3f}'))

json.dump(OUT, open('results/review_summary.json', 'w'), indent=1, default=float)
print('\nwritten results/review_*.csv, results/review_summary.json')
