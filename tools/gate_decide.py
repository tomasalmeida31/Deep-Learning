"""Mechanical GO / STOP / GREY decision for the gate experiment (pre-registered rule).

  python tools/gate_decide.py --datanames adultv_s0 defaultv_s0 newsv_s0

Inputs per dataset: results/<d>_gate.jsonl (from eval_fast, reference = real.csv = train')
and results/<d>_floor.json (from floor_boot). Writes results/<d>_gate_summary.csv
(one row per config: mean/sd over sampling seeds) and prints the verdict.

Rule (fixed before any model was trained):
  E99      = mean shape/trend error of heun, 50 steps (99 NFE), s_churn = 0, over its seeds
  sigma_s  = sd of that same config over its seeds
  delta    = DELTA_SHAPE / DELTA_TREND (pp); tolerance per metric = max(delta, 3 * sigma_s)
  a config "matches" if, for every metric NOT excluded, mean error <= E99 + tolerance;
  a (dataset, metric) is EXCLUDED when its bootstrap-floor sd > delta / 3 (metric too noisy);
  the floor MEAN never excludes anything (errors are reported as excess over it)
  a dataset with NO surviving metric is ABSENT: it is dropped from the gate and the
  denominators shrink (rule below); with < MIN_DATASETS surviving the gate is INCONCLUSIVE
  and the next dataset in ADD_ORDER must be run before any verdict is read
  NFE*     = smallest NFE over ALL configs (any solver, churn, schedule) that matches
  1-step   = best (lowest excess) 1-NFE config; "posterior mean suffices" if it matches
  Verdict over the surviving datasets S (|S| >= MIN_DATASETS):
           STOP if NFE* <= 3 on all of S, or 1-step matches on all of S;
           GO   if NFE* >= 7 on at least ceil(2/3 * |S|) of S  (3 -> 2, 2 -> 2, 4 -> 3);
           GREY otherwise (student restricted to 1-2 NFE).
"""
import argparse, json, os, sys
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.chdir(ROOT)
DELTA = {'shape': 0.25, 'trend': 0.5}
MIN_DATASETS = 2
ADD_ORDER = ['shoppers', 'magic', 'beijing']   # next gate datasets if the base erodes

def analyse(d, min_seeds=10, ignore_floor_noise=False):
    rows = [json.loads(l) for l in open(f'results/{d}_gate.jsonl') if l.strip()]
    df = pd.DataFrame(rows)
    need = {'solver', 'steps', 's_churn', 'sigma_max', 'rho', 'seed', 'nfe', 'shape_err', 'trend_err'}
    assert need <= set(df.columns), f'missing columns in results/{d}_gate.jsonl: {need - set(df.columns)}'
    floor = json.load(open(f'results/{d}_floor.json'))
    key = ['solver', 'steps', 's_churn', 'sigma_max', 'rho', 'nfe']
    g = df.groupby(key).agg(n=('seed', 'count'), shape_mean=('shape_err', 'mean'), shape_sd=('shape_err', 'std'),
                            trend_mean=('trend_err', 'mean'), trend_sd=('trend_err', 'std')).reset_index()
    g['shape_excess'] = g['shape_mean'] - floor['shape_floor']; g['trend_excess'] = g['trend_mean'] - floor['trend_floor']
    ref = g[(g.solver == 'heun') & (g.steps == 50) & (g.s_churn == 0) & (g.sigma_max == 80) & (g.rho == 7)]
    assert len(ref) == 1 and ref.n.iloc[0] >= min_seeds, f'reference config heun/50/c0 missing or < {min_seeds} seeds'
    ref = ref.iloc[0]
    excluded = [] if ignore_floor_noise else [m for m in ('shape', 'trend') if floor[f'{m}_floor_sd'] > DELTA[m] / 3]
    tol = {m: max(DELTA[m], 3 * ref[f'{m}_sd']) for m in ('shape', 'trend')}
    used = [m for m in ('shape', 'trend') if m not in excluded]
    if not used:
        g.to_csv(f'results/{d}_gate_summary.csv', index=False)
        return {'dataname': d, 'absent': True, 'excluded_metrics': excluded, 'floor_sd_shape': floor['shape_floor_sd'], 'floor_sd_trend': floor['trend_floor_sd'], 'n_configs': int(len(g))}
    g['matches'] = np.all([g[f'{m}_mean'] <= ref[f'{m}_mean'] + tol[m] for m in used], axis=0)
    g = g.sort_values(['nfe', 'shape_mean']); g.to_csv(f'results/{d}_gate_summary.csv', index=False)
    nfe_star = int(g[g.matches].nfe.min()) if g.matches.any() else None
    one = g[g.nfe == 1].sort_values('shape_mean').iloc[0]
    sto = g[(g.solver == 'heun') & (g.steps == 50) & (g.s_churn == 1)]
    churn_gap = {m: float(sto[f'{m}_mean'].iloc[0] - ref[f'{m}_mean']) for m in ('shape', 'trend')} if len(sto) else None
    return {'dataname': d, 'absent': False, 'E99_shape': float(ref.shape_mean), 'E99_trend': float(ref.trend_mean), 'sigma_s_shape': float(ref.shape_sd), 'sigma_s_trend': float(ref.trend_sd),
            'floor_shape': floor['shape_floor'], 'floor_trend': floor['trend_floor'], 'floor_sd_shape': floor['shape_floor_sd'], 'floor_sd_trend': floor['trend_floor_sd'],
            'excluded_metrics': excluded, 'tolerance': tol, 'NFE_star': nfe_star, 'one_step_matches': bool(one.matches),
            'one_step_best': f"{one.solver} smax={one.sigma_max} shape={one.shape_mean:.2f} trend={one.trend_mean:.2f}",
            'churn99_minus_det99': churn_gap, 'n_configs': int(len(g))}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--datanames', nargs='+', required=True); ap.add_argument('--min_seeds', type=int, default=10, help='smoke tests only'); ap.add_argument('--ignore_floor_noise', action='store_true', help='smoke tests only'); ap.add_argument('--out', default='results/gate_verdict.json'); a = ap.parse_args()
    R = [analyse(d, a.min_seeds, a.ignore_floor_noise) for d in a.datanames]
    for r in R:
        if r['absent']:
            print(f"\n[{r['dataname']}] ABSENT: both metrics excluded (floor sd shape {r['floor_sd_shape']:.3f}, trend {r['floor_sd_trend']:.3f})"); continue
        print(f"\n[{r['dataname']}] E99 det: shape {r['E99_shape']:.3f} (floor {r['floor_shape']:.3f}, sigma_s {r['sigma_s_shape']:.3f})  trend {r['E99_trend']:.3f} (floor {r['floor_trend']:.3f}, sigma_s {r['sigma_s_trend']:.3f})")
        print(f"   tolerance shape {r['tolerance']['shape']:.3f} trend {r['tolerance']['trend']:.3f}; excluded metrics: {r['excluded_metrics'] or 'none'}")
        print(f"   churn(99) - det(99): {r['churn99_minus_det99']}")
        print(f"   NFE* = {r['NFE_star']}   1-step matches: {r['one_step_matches']} ({r['one_step_best']})   configs: {r['n_configs']}")
    S = [r for r in R if not r['absent']]
    absent = [r['dataname'] for r in R if r['absent']]
    stars = [r['NFE_star'] for r in S]
    import math
    if len(S) < MIN_DATASETS:
        verdict = 'INCONCLUSIVE'
        print(f"\nVERDICT: INCONCLUSIVE  (only {len(S)} dataset(s) survive; absent: {absent}). Run the next dataset in {ADD_ORDER} through the gate and re-run this script.")
    else:
        need_go = math.ceil(2 * len(S) / 3)
        if all(s is not None and s <= 3 for s in stars) or all(r['one_step_matches'] for r in S):
            verdict = 'STOP'
        elif sum(1 for s in stars if s is None or s >= 7) >= need_go:
            verdict = 'GO'
        else:
            verdict = 'GREY'
        print(f"\nVERDICT: {verdict}   (surviving datasets {[r['dataname'] for r in S]}, absent {absent}; NFE* {stars}; None = nothing matched even at 99 NFE, treated as >= 7; GO needs >= {need_go})")
    json.dump({'verdict': verdict, 'per_dataset': R, 'rule': open(__file__).read().split('"""')[1]}, open(a.out, 'w'), indent=1, default=str)
    print('written', a.out)

if __name__ == '__main__':
    main()
