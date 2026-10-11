"""E1 (ablation plan): choose the fair few-step TEACHER sampler per NFE budget on the VALIDATION carve.

  python tools/pick_fast.py --dataname adultv_s0
      reads  results/<dataname>_fast_valgrid.jsonl  (eval_fast --real val over sample_grid --grid fast_val, tag v)
      writes results/<dataname>_fast_choice.json and prints FAST_SPECS lines (solver:steps:0:smax:rho) for the sbatch

Rule (fixed before any result): for each budget k in {1, 2, 4, 8, 16}, F_k = the candidate with the lowest mean
val shape error among all candidates with nfe <= k (so heun at 3 NFE may win k = 4); ties broken by val trend.
Same protocol as the student's tau choice (pick_taus.py): 5 seeds x 8192 rows scored against val.csv, relative
choice only; reported numbers come from real.csv (train') with 20 seeds.
"""
import argparse, json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.chdir(ROOT)
import pandas as pd

BUDGETS = (1, 2, 4, 8, 16)
KEY = ['solver', 'steps', 'sigma_max', 'rho', 'nfe']

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--dataname', required=True); ap.add_argument('--seeds', type=int, default=5)
    a = ap.parse_args()
    df = pd.DataFrame([json.loads(l) for l in open(f'results/{a.dataname}_fast_valgrid.jsonl') if l.strip()])
    need = set(KEY) | {'seed', 'shape_err', 'trend_err'}
    assert need <= set(df.columns), f'missing columns {need - set(df.columns)}'
    df = df.drop_duplicates(subset=KEY + ['seed'], keep='last')
    g = df.groupby(KEY).agg(n=('seed', 'count'), shape_m=('shape_err', 'mean'), shape_sd=('shape_err', 'std'),
                            trend_m=('trend_err', 'mean')).reset_index()
    bad = g[g.n != a.seeds]
    assert not len(bad), f'groups without exactly {a.seeds} seeds:\n{bad}'
    chosen, specs = {}, []
    for k in BUDGETS:
        sub = g[g.nfe <= k].sort_values(['shape_m', 'trend_m'])
        assert len(sub), f'no candidate with nfe <= {k}'
        print(f'budget {k} NFE: {len(sub)} candidates; top 5')
        for _, r in sub.head(5).iterrows():
            print(f'   {r.solver:8s} N={int(r.steps):2d} nfe={int(r.nfe):2d} smax={r.sigma_max:>3} rho={r.rho}  shape {r.shape_m:6.2f}±{r.shape_sd:4.2f}  trend {r.trend_m:6.2f}')
        b = sub.iloc[0]
        chosen[k] = {'solver': b.solver, 'steps': int(b.steps), 'sigma_max': int(b.sigma_max), 'rho': int(b.rho), 'nfe': int(b.nfe),
                     'val_shape': float(b.shape_m), 'val_trend': float(b.trend_m), 'n_candidates': int(len(sub))}
        specs.append(f'{b.solver}:{int(b.steps)}:0:{int(b.sigma_max)}:{int(b.rho)}')
    json.dump(chosen, open(f'results/{a.dataname}_fast_choice.json', 'w'), indent=1)
    print('CHOSEN', json.dumps(chosen))
    print('FAST_SPECS', ' '.join(sorted(set(specs))))

if __name__ == '__main__':
    main()
