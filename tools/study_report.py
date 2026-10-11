"""Paired analysis of the full study: datasets x training seeds, student vs teacher at matched NFE.

  python tools/study_report.py --datasets adult default news --seeds 0 1 2 --tag study [--fig]

Per cell (dataset, seed) it reads
  results/<d>v_s<k>_gate_summary.csv           teacher shape/trend per config (mean over sampling seeds)
  results/<d>v_s<k>_floor.json                 perfect-generator floor
  results/<d>v_s<k>_student_<tag>_s<k>.jsonl   student per sampling seed (shape, trend, c2st)
  results/<d>v_s<k>_teacher_c2st.jsonl         teacher reference configs per sampling seed (c2st)
  results/<d>v_s<k>_teacher_extra.jsonl        teacher MLE / DCR       (optional)
  results/<d>v_s<k>_student_<tag>_s<k>_extra.jsonl   student MLE / DCR  (optional)

Comparisons are PAIRED WITHIN A CELL (same teacher, same data), then summarised over cells:
mean difference, sd over cells, wins/cells, and the exact two-sided sign test. Never means of means.
Matched-NFE teacher baseline at NFE k = best config (lowest shape) with nfe == k among Heun/Euler with
sigma_max 80, rho 7, churn 0. Also reported: student vs teacher at NFE* (smallest NFE within tolerance of
99 NFE, from gate_decide) and vs 99 NFE. Writes results/study_<tag>_cells.csv and results/study_<tag>_summary.csv;
--fig writes results/study_<tag>_nfe.png (error vs NFE, teacher curve and student points, one panel per dataset).
"""
import argparse, glob, json, math, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.chdir(ROOT)
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument('--datasets', nargs='+', required=True); ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
ap.add_argument('--tag', default='study'); ap.add_argument('--fig', action='store_true'); ap.add_argument('--nfes', type=int, nargs='+', default=[1, 2, 4])
a = ap.parse_args()

def jl(p): return pd.DataFrame([json.loads(l) for l in open(p) if l.strip()]) if os.path.exists(p) else pd.DataFrame()
sys.path.insert(0, ROOT)
from tools.stats import sign_test

cells, curves = [], {}
for d in a.datasets:
    for k in a.seeds:
        A = f'{d}v_s{k}'; S = f'student_{a.tag}_s{k}'
        gs, st = f'results/{A}_gate_summary.csv', f'results/{A}_{S}.jsonl'
        if not (os.path.exists(gs) and os.path.exists(st)): print(f'-- {A}: missing {"gate_summary" if not os.path.exists(gs) else "student"}; skipped'); continue
        Tall = pd.read_csv(gs); Tall = Tall[Tall.s_churn == 0]           # every churn-0 gate config (any sigma_max, rho)
        T = Tall[(Tall.sigma_max == 80) & (Tall.rho == 7)]
        floor = json.load(open(f'results/{A}_floor.json')) if os.path.exists(f'results/{A}_floor.json') else {}
        ref = T[(T.solver == 'heun') & (T.steps == 50)].iloc[0]
        tref = T[(T.solver == 'heun') & (T.steps.isin([20, 50]))].sort_values('shape_mean').iloc[0]   # E7.3: best of 39 / 99 NFE
        fchoice = json.load(open(f'results/{A}_fast_choice.json')) if os.path.exists(f'results/{A}_fast_choice.json') else {}
        Fx = jl(f'results/{A}_fast.jsonl'); Lx = jl(f'results/{A}_mle_log.jsonl')
        tol = {m: max({'shape': 0.25, 'trend': 0.5}[m], 3 * ref[f'{m}_sd']) for m in ('shape', 'trend')}
        ok = T[(T.shape_mean <= ref.shape_mean + tol['shape'])]
        nfe_star = int(ok.nfe.min()) if len(ok) else None
        tstar = T[T.nfe == nfe_star].sort_values('shape_mean').iloc[0] if nfe_star else None
        Sdf = jl(st); Tc = jl(f'results/{A}_teacher_c2st.jsonl'); Tx = jl(f'results/{A}_teacher_extra.jsonl'); Sx = jl(f'results/{A}_{S}_extra.jsonl')
        curves[(d, k)] = {'teacher': T[['solver', 'nfe', 'shape_mean', 'trend_mean']].sort_values('nfe'), 'student': Sdf.groupby('nfe')[['shape_err', 'trend_err']].mean().reset_index(), 'floor': floor}
        for nfe in a.nfes:
            s = Sdf[Sdf.nfe == nfe]
            if not len(s): continue
            tm = T[T.nfe == nfe].sort_values('shape_mean')
            row = {'dataset': d, 'seed': k, 'nfe': nfe, 'n_student_seeds': int(len(s)),
                   'student_shape': s.shape_err.mean(), 'student_shape_sd': s.shape_err.std(ddof=1), 'student_trend': s.trend_err.mean(),
                   'student_c2st': s.c2st.mean() if 'c2st' in s else np.nan,
                   'teacher_matched_cfg': f'{tm.iloc[0].solver}{int(tm.iloc[0].steps)}' if len(tm) else None,
                   'teacher_matched_shape': tm.iloc[0].shape_mean if len(tm) else np.nan, 'teacher_matched_trend': tm.iloc[0].trend_mean if len(tm) else np.nan,
                   'nfe_star': nfe_star, 'teacher_star_shape': tstar.shape_mean if tstar is not None else np.nan, 'teacher_star_trend': tstar.trend_mean if tstar is not None else np.nan,
                   'teacher99_shape': ref.shape_mean, 'teacher99_trend': ref.trend_mean, 'floor_shape': floor.get('shape_floor', np.nan), 'floor_trend': floor.get('trend_floor', np.nan),
                   'teacher_ref_cfg': f'heun{int(tref.steps)}', 'teacher_ref_shape': tref.shape_mean, 'teacher_ref_trend': tref.trend_mean}
            tb = Tall[Tall.nfe == nfe].sort_values('shape_mean')             # E7.4: strongest gate config at this NFE
            if len(tb):
                row.update({'teacher_best_cfg': f'{tb.iloc[0].solver}{int(tb.iloc[0].steps)}_smax{int(tb.iloc[0].sigma_max)}_rho{int(tb.iloc[0].rho)}',
                            'teacher_best_shape': tb.iloc[0].shape_mean, 'teacher_best_trend': tb.iloc[0].trend_mean})
            fc = fchoice.get(str(nfe))                                        # E1: fair few-step teacher F_k (val-selected)
            if fc and len(Fx):
                f = Fx[(Fx.solver == fc['solver']) & (Fx.steps == fc['steps']) & (Fx.sigma_max == fc['sigma_max']) & (Fx.rho == fc['rho'])]
                if len(f):
                    row.update({'teacher_fair_cfg': f"{fc['solver']}{fc['steps']}_smax{fc['sigma_max']}_rho{fc['rho']}", 'teacher_fair_n': int(len(f)),
                                'teacher_fair_shape': f.shape_err.mean(), 'teacher_fair_trend': f.trend_err.mean(),
                                'teacher_fair_c2st': f.c2st.mean() if 'c2st' in f else np.nan})
            if len(Lx) and 'mle_r2_log' in Lx:                                # E7.1: news MLE on the log1p target
                ls = Lx[(Lx.solver == 'cm') & (Lx.nfe == nfe)]
                if len(ls): row['student_mle_r2_log'] = ls.mle_r2_log.mean()
                if len(tm):
                    lt = Lx[(Lx.solver == tm.iloc[0].solver) & (Lx.steps == tm.iloc[0].steps) & (Lx.sigma_max == 80)]
                    if len(lt): row['teacher_matched_mle_r2_log'] = lt.mle_r2_log.mean()
                lt = Lx[(Lx.solver == 'heun') & (Lx.steps == 50) & (Lx.sigma_max == 80)]
                if len(lt): row['teacher99_mle_r2_log'] = lt.mle_r2_log.mean()
            if len(tm) and len(Tc):
                c = Tc[(Tc.solver == tm.iloc[0].solver) & (Tc.steps == tm.iloc[0].steps)]; row['teacher_matched_c2st'] = c.c2st.mean() if len(c) else np.nan
            if len(Tc): c = Tc[(Tc.solver == 'heun') & (Tc.steps == 50)]; row['teacher99_c2st'] = c.c2st.mean() if len(c) else np.nan
            for m in ('mle_roc_auc', 'mle_macro_f1', 'dcr_score'):             # mle_rmse dropped (E7.1: invalid on news)
                if len(Sx) and m in Sx: row[f'student_{m}'] = Sx[Sx.nfe == nfe][m].mean()
                if len(Tx) and m in Tx:
                    if len(tm): c = Tx[(Tx.solver == tm.iloc[0].solver) & (Tx.steps == tm.iloc[0].steps)]; row[f'teacher_matched_{m}'] = c[m].mean() if len(c) else np.nan
                    c = Tx[(Tx.solver == 'heun') & (Tx.steps == 50)]; row[f'teacher99_{m}'] = c[m].mean() if len(c) else np.nan
            for m in ('shape', 'trend', 'c2st', 'mle_roc_auc', 'mle_macro_f1', 'mle_r2_log', 'dcr_score'):
                for b, lab in (('teacher_matched', 'matched'), ('teacher_best', 'best'), ('teacher_fair', 'fair'), ('teacher_star', 'star'), ('teacher_ref', 'ref'), ('teacher99', '99')):
                    if f'student_{m}' in row and f'{b}_{m}' in row:
                        if m == 'dcr_score':   # DCR ideal is 0.5: compare distances from 0.5 (negative = student closer to ideal)
                            row[f'd_{m}_vs_{lab}'] = abs(row[f'student_{m}'] - 0.5) - abs(row[f'{b}_{m}'] - 0.5)
                        else: row[f'd_{m}_vs_{lab}'] = row[f'student_{m}'] - row[f'{b}_{m}']
            cells.append(row)

C = pd.DataFrame(cells)
if not len(C): sys.exit('no complete cells')
TREND_EXCLUDED = {'default'}   # trend floor too noisy on default (gate: bootstrap-floor sd > delta/3); pre-registered exclusion
for col in [c for c in C.columns if c.startswith('d_trend_')]: C.loc[C.dataset.isin(TREND_EXCLUDED), col] = np.nan
C.to_csv(f'results/study_{a.tag}_cells.csv', index=False)
lower_better = {'shape', 'trend', 'dcr_score'}   # dcr_score diffs are |x-0.5| deviations          # c2st: 1 = indistinguishable is best -> higher better; auc/f1 higher better; dcr: closer to 0.5
summ = []
for nfe, g in C.groupby('nfe'):
    for col in [c for c in C.columns if c.startswith('d_')]:
        m = col[2:].split('_vs_')[0]; v = g[col].dropna()
        if not len(v): continue
        wins = int((v < 0).sum()) if m in lower_better else int((v > 0).sum())
        summ.append({'nfe': nfe, 'comparison': col, 'cells': int(len(v)), 'mean_diff': v.mean(), 'sd_over_cells': v.std(ddof=1) if len(v) > 1 else np.nan, 'student_wins': wins, 'sign_p': sign_test(v)})
Su = pd.DataFrame(summ); Su.to_csv(f'results/study_{a.tag}_summary.csv', index=False)
with pd.option_context('display.width', 250, 'display.max_columns', 40, 'display.float_format', lambda x: f'{x:.3f}'):
    print('== per cell (shape pp; negative d_ = student better for shape/trend/rmse)')
    print(C[['dataset', 'seed', 'nfe', 'student_shape', 'teacher_matched_cfg', 'teacher_matched_shape', 'nfe_star', 'teacher_star_shape', 'teacher99_shape', 'floor_shape']].to_string(index=False))
    print('== paired summary over cells')
    print(Su.to_string(index=False))
print(f'written results/study_{a.tag}_cells.csv, results/study_{a.tag}_summary.csv  ({C.dataset.nunique()} datasets x {C.seed.nunique()} seeds)')

if a.fig:
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    ds = a.datasets; fig, axes = plt.subplots(1, len(ds), figsize=(4.2 * len(ds), 3.4), squeeze=False)
    for ax, d in zip(axes[0], ds):
        for (dd, k), cv in curves.items():
            if dd != d: continue
            t = cv['teacher']
            for sol, ls in (('heun', '-'), ('euler', '--')):
                tt = t[t.solver == sol]; ax.plot(tt.nfe, tt.shape_mean, ls, color='0.4', lw=1, alpha=0.8, label=f'teacher {sol}' if k == a.seeds[0] else None)
            s = cv['student']; ax.plot(s.nfe, s.shape_err, 'o', color='C3', ms=4, label='CD student' if k == a.seeds[0] else None)
            if cv['floor']: ax.axhline(cv['floor']['shape_floor'], color='0.7', lw=0.8, ls=':', label='floor' if k == a.seeds[0] else None)
        ax.set_xscale('log'); ax.set_xlabel('NFE'); ax.set_title(d); ax.grid(alpha=0.3)
    axes[0][0].set_ylabel('shape error (pp)'); axes[0][0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(f'results/study_{a.tag}_nfe.png', dpi=160); print(f'written results/study_{a.tag}_nfe.png')
