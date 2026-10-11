"""Ablation plan report: paired per-cell tables and the PRE-REGISTERED readings (claude/tabsyn-ablation-plan.md).

  python tools/ablation_report.py --section fast|ct|N|long|all [--datasets adult default news] [--seeds 0 1 2]

Sections
  fast  E1  study student S_j vs the val-selected fair few-step teacher F_k (results/<A>_fast_choice.json, _fast.jsonl)
  ct    E2  consistency training arm `ct` (all cells) and `ict` (seed 0, descriptive) vs the study student
  N     E3  N8 / N40 vs N18 (= study) on seed-0 cells, with convergence diagnostics from log.jsonl
  long  E4  40k-step continuation vs the 20k study student
Every comparison is paired within a cell (dataset, training seed) and summarised as mean, sd over cells, wins/cells
and the exact sign test; never means of means. The reading strings are the plan's rules, applied mechanically.
Writes results/ablation_<section>_cells.csv and results/ablation_<section>_summary.csv.
"""
import argparse, json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.chdir(ROOT)
import numpy as np, pandas as pd
from tools.stats import paired_row

TOL = 0.25   # pp, the gate's shape tolerance


def jl(p): return pd.DataFrame([json.loads(l) for l in open(p) if l.strip()]) if os.path.exists(p) else pd.DataFrame()


def student(A, tag, k):
    """mean shape / trend / c2st per NFE for student_<tag>_s<k>, or None if missing."""
    df = jl(f'results/{A}_student_{tag}_s{k}.jsonl')
    if not len(df): return None
    g = df.groupby('nfe').agg(shape=('shape_err', 'mean'), trend=('trend_err', 'mean'), n=('seed', 'count'),
                              **({'c2st': ('c2st', 'mean')} if 'c2st' in df else {}))
    return g


def convergence(A, tag, k):
    p = f'tabsyn/ckpt/{A}/student_{tag}_s{k}/log.jsonl'
    if not os.path.exists(p): return {}
    r = [json.loads(l) for l in open(p) if l.strip()]; r = [x for x in r if 'ema_selfcons' in x]
    if not r: return {}
    early = np.mean([x['ema_selfcons'] for x in r if x['step'] <= 2000])
    exit_step = next((x['step'] for x in r if x['ema_selfcons'] < 0.8 * early), None)
    last = r[-1]; prev = [x for x in r if x['step'] <= last['step'] - 2000]
    drop = (prev[-1]['ema_selfcons'] - last['ema_selfcons']) / prev[-1]['ema_selfcons'] if prev else float('nan')
    best = min(r, key=lambda x: x['ema_selfcons'])
    return {'plateau_exit': exit_step, 'last2k_rel_drop': drop, 'best_step': best['step'], 'last_step': last['step']}


def cells(datasets, seeds):
    for d in datasets:
        for k in seeds: yield d, k, f'{d}v_s{k}'


def summarise(C, cols, lower_better=True):
    return pd.DataFrame([paired_row(c, C[c].tolist(), lower_better) for c in cols if c in C])


def count(v, cond): v = [x for x in v if x == x]; return sum(cond(x) for x in v), len(v)


def sec_fast(a):
    rows = []
    for d, k, A in cells(a.datasets, a.seeds):
        S = student(A, 'study', k); ch = f'results/{A}_fast_choice.json'; F = jl(f'results/{A}_fast.jsonl')
        if S is None or not os.path.exists(ch) or not len(F): print(f'-- {A}: missing study or fast results; skipped'); continue
        choice = json.load(open(ch)); row = {'dataset': d, 'seed': k}
        for kk, c in choice.items():
            f = F[(F.solver == c['solver']) & (F.steps == c['steps']) & (F.sigma_max == c['sigma_max']) & (F.rho == c['rho'])]
            row[f'F{kk}_cfg'] = f"{c['solver']}{c['steps']}_smax{c['sigma_max']}_rho{c['rho']}(nfe{c['nfe']})"
            row[f'F{kk}_shape'] = f.shape_err.mean(); row[f'F{kk}_c2st'] = f.c2st.mean() if 'c2st' in f else np.nan
        for j in (1, 2, 4):
            if j in S.index: row[f'S{j}_shape'] = S.loc[j, 'shape']; row[f'S{j}_c2st'] = S.loc[j].get('c2st', np.nan)
        for j in (1, 2, 4): row[f'd_S{j}_F{j}'] = row.get(f'S{j}_shape', np.nan) - row.get(f'F{j}_shape', np.nan)
        for kk in (4, 8, 16): row[f'd_S1_F{kk}'] = row.get('S1_shape', np.nan) - row.get(f'F{kk}_shape', np.nan)
        row['d_c2st_S4_F4'] = row.get('S4_c2st', np.nan) - row.get('F4_c2st', np.nan)
        row['d_c2st_S1_F8'] = row.get('S1_c2st', np.nan) - row.get('F8_c2st', np.nan)
        # AUROC (classification cells): student _extra vs fast _extra
        Sx, Fx = jl(f'results/{A}_student_study_s{k}_extra.jsonl'), jl(f'results/{A}_fast_extra.jsonl')
        if len(Sx) and len(Fx) and 'mle_roc_auc' in Sx and 'mle_roc_auc' in Fx:
            for sj, fk in ((4, 4), (1, 8)):
                c = choice.get(str(fk))
                if not c: continue
                f = Fx[(Fx.solver == c['solver']) & (Fx.steps == c['steps']) & (Fx.sigma_max == c['sigma_max']) & (Fx.rho == c['rho'])]
                row[f'd_auc_S{sj}_F{fk}'] = Sx[Sx.nfe == sj].mle_roc_auc.mean() - f.mle_roc_auc.mean() if len(f) else np.nan
        rows.append(row)
    C = pd.DataFrame(rows)
    if not len(C): return C, pd.DataFrame(), ['no cells']
    Su = pd.concat([summarise(C, [f'd_S{j}_F{j}' for j in (1, 2, 4)] + [f'd_S1_F{k}' for k in (4, 8, 16)]),
                    summarise(C, ['d_c2st_S4_F4', 'd_c2st_S1_F8', 'd_auc_S4_F4', 'd_auc_S1_F8'], lower_better=False)])
    n = len(C); need = n - 1 if n == 9 else int(np.ceil(0.85 * n))            # >= 8 of 9
    head = None
    for kk in (16, 8, 4):
        w, m = count(C[f'd_S1_F{kk}'], lambda x: x < 0)
        if m and w >= need: head = kk; break
    R = [f'cells: {n}; headline threshold: S_1 < F_k in >= {need}/{n} cells']
    if head: R.append(f'READING: the 1-NFE student beats the best val-tuned teacher sampler up to {head} NFE.')
    else: R.append('READING: S_1 >= F_4 in >= 2 cells -> claim becomes "the student matches the tuned teacher at ~4 NFE"; delete the 17x number (quote it only against the authors\' shipped sampler, labelled).')
    return C, Su, R


def arm_vs_study(a, tag, seeds):
    rows = []
    for d, k, A in cells(a.datasets, seeds):
        S, X = student(A, 'study', k), student(A, tag, k)
        if S is None or X is None: print(f'-- {A}: missing study or {tag}; skipped'); continue
        row = {'dataset': d, 'seed': k, 'arm': tag}
        for j in (1, 2, 4):
            row[f'{tag}{j}_shape'] = X.loc[j, 'shape'] if j in X.index else np.nan
            row[f'd{j}'] = row[f'{tag}{j}_shape'] - (S.loc[j, 'shape'] if j in S.index else np.nan)
        row.update({f'{kk}': v for kk, v in convergence(A, tag, k).items()})
        rows.append(row)
    return pd.DataFrame(rows)


def sec_ct(a):
    C = arm_vs_study(a, 'ct', a.seeds); I = arm_vs_study(a, 'ict', [0])
    if not len(C): return C, pd.DataFrame(), ['no ct cells']
    Su = summarise(C, ['d1', 'd2', 'd4']); n = len(C); need = int(np.ceil(7 / 9 * n))
    hi = all(count(C[f'd{j}'], lambda x: x >= TOL)[0] >= need for j in (1, 4))
    eq = all(count(C[f'd{j}'], lambda x: abs(x) < TOL)[0] >= need for j in (1, 4))
    lo = all(count(C[f'd{j}'], lambda x: x <= -TOL)[0] >= need for j in (1, 4))
    R = [f'cells: {n}; rule threshold {need}/{n} at j = 1 and 4, tolerance {TOL} pp']
    if hi: R.append("READING (i): the teacher's ODE step is necessary; CT does not reach CD quality.")
    elif eq: R.append('READING (ii): CT matches CD; the residual gap is a property of the consistency parametrisation/objective on these latents, not of distillation.')
    elif lo: R.append('READING (iii): CT is better than CD; reported as such; the headline student stays the pre-registered CD one.')
    else: R.append('READING: inconclusive at this budget; the table is shown.')
    if len(I): R.append('ict (descriptive, seed-0 cells): ' + '; '.join(f"{r.dataset}: d1 {r.d1:+.2f} d2 {r.d2:+.2f} d4 {r.d4:+.2f}" for r in I.itertuples()))
    return pd.concat([C, I]), Su, R


def sec_N(a):
    C8, C40 = arm_vs_study(a, 'N8', [0]), arm_vs_study(a, 'N40', [0])
    if not len(C8) and not len(C40): return pd.DataFrame(), pd.DataFrame(), ['no N cells']
    C = pd.concat([C8, C40]); R = [f'cells: N8 {len(C8)}, N40 {len(C40)} (seed 0 only; >= 2/3 rule)']
    Su = pd.concat([summarise(C8, ['d1', 'd2', 'd4']).assign(arm='N8'), summarise(C40, ['d1', 'd2', 'd4']).assign(arm='N40')])
    n8_worse = sum(1 for r in C8.itertuples() if all(getattr(r, f'd{j}') >= TOL for j in (1, 2, 4)))
    n40_same = sum(1 for r in C40.itertuples() if all(abs(getattr(r, f'd{j}')) < TOL for j in (1, 2, 4)))
    n40_better = sum(1 for r in C40.itertuples() if all(getattr(r, f'd{j}') <= -TOL for j in (1, 2, 4)))
    if n8_worse >= 2 and n40_same >= 2: R.append('READING: the grid must be at least ~18 points; finer grids do not help at this training length.')
    elif n40_better >= 2: R.append('READING: N40 better by >= 0.25 pp in >= 2/3 cells; reported; N = 18 stays the headline; N40 seeds 1-2 = future work.')
    else: R.append(f'READING: no pre-registered pattern (N8 worse in {n8_worse}/3, N40 same in {n40_same}/3, N40 better in {n40_better}/3); table shown.')
    for r in C40.itertuples():
        pe, dr = getattr(r, 'plateau_exit', None), getattr(r, 'last2k_rel_drop', np.nan)
        if (pe is None or pe != pe or pe > 12000) or (dr == dr and dr > 0.05): R.append(f'{r.dataset} N40: UNCONVERGED at 20k (plateau exit {pe}, last-2k drop {dr:.1%})')
    return C, Su, R


def sec_long(a):
    C = arm_vs_study(a, 'long', a.seeds)
    if not len(C): return C, pd.DataFrame(), ['no long cells']
    Su = summarise(C, ['d1', 'd2', 'd4']); n = len(C); need = int(np.ceil(7 / 9 * n))
    past = count(C['best_step'], lambda x: x > 20000) if 'best_step' in C else (0, 0)
    R = [f'cells: {n}; best checkpoint past 20k in {past[0]}/{past[1]} cells']
    if count(C['d4'], lambda x: x <= -TOL)[0] >= need:
        R.append('READING: d4 <= -0.25 pp in >= 7/9 cells -> the report MAY use `long` as the headline for ALL cells and must state that the length was chosen by this pre-registered rule after the 20k study.')
    else:
        R.append('READING: headline stays the 20k study; 40k is one ablation row.')
    if past[1] and past[1] - past[0] >= need: R.append('READING: best checkpoint <= 20k in >= 7/9 cells -> "20k is converged; the gap is not a training-length artefact".')
    return C, Su, R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--section', choices=['fast', 'ct', 'N', 'long', 'all'], required=True)
    ap.add_argument('--datasets', nargs='+', default=['adult', 'default', 'news']); ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    a = ap.parse_args()
    secs = ['fast', 'ct', 'N', 'long'] if a.section == 'all' else [a.section]
    fn = {'fast': sec_fast, 'ct': sec_ct, 'N': sec_N, 'long': sec_long}
    for s in secs:
        C, Su, R = fn[s](a)
        print(f'\n===== {s} =====')
        with pd.option_context('display.width', 250, 'display.max_columns', 60, 'display.float_format', lambda x: f'{x:.3f}'):
            if len(C): print(C.to_string(index=False)); C.to_csv(f'results/ablation_{s}_cells.csv', index=False)
            if len(Su): print(Su.to_string(index=False)); Su.to_csv(f'results/ablation_{s}_summary.csv', index=False)
        for r in R: print(r)


if __name__ == '__main__':
    main()
