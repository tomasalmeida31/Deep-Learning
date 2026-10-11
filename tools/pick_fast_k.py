"""Measured F_k for an extra budget k (1 Oct 2026; review finding 13: "≈ 6 NFE" was an interpolation).

  python tools/pick_fast_k.py --dataname adultv_s0 --k 6 --valgrid results/adultv_s0_fast_valgrid.jsonl results/adultv_s0_fast6_valgrid.jsonl
      writes results/<dataname>_fast<k>_choice.json and prints FASTK_SPECS solver:steps:0:smax:rho

Same rule as pick_fast.py (E1): F_k = the candidate with the lowest mean val shape error among ALL scored
candidates with nfe <= k (the original 196-config E1 grid plus the new k-NFE candidates), ties by val trend;
5 val seeds x 8192 rows, relative choice only. The chosen config is then sampled with 20 seeds on train'.
"""
import argparse, json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.chdir(ROOT)
import pandas as pd

KEY = ['solver', 'steps', 'sigma_max', 'rho', 'nfe']


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--dataname', required=True); ap.add_argument('--k', type=int, required=True)
    ap.add_argument('--valgrid', nargs='+', required=True); ap.add_argument('--seeds', type=int, default=5)
    a = ap.parse_args()
    rows = []
    for p in a.valgrid:
        rows += [json.loads(l) for l in open(p) if l.strip()]
    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=KEY + ['seed'], keep='last')
    g = df.groupby(KEY).agg(n=('seed', 'count'), shape_m=('shape_err', 'mean'), shape_sd=('shape_err', 'std'), trend_m=('trend_err', 'mean')).reset_index()
    bad = g[g.n != a.seeds]
    assert not len(bad), f'groups without exactly {a.seeds} seeds:\n{bad}'
    sub = g[g.nfe <= a.k].sort_values(['shape_m', 'trend_m'])
    assert len(sub), f'no candidate with nfe <= {a.k}'
    print(f'budget {a.k} NFE: {len(sub)} candidates; top 5')
    for _, r in sub.head(5).iterrows():
        print(f'   {r.solver:8s} N={int(r.steps):2d} nfe={int(r.nfe):2d} smax={r.sigma_max:>4} rho={r.rho}  shape {r.shape_m:6.2f}±{r.shape_sd:4.2f}  trend {r.trend_m:6.2f}')
    b = sub.iloc[0]
    sm = int(b.sigma_max) if float(b.sigma_max) == int(float(b.sigma_max)) else float(b.sigma_max)
    chosen = {str(a.k): {'solver': b.solver, 'steps': int(b.steps), 'sigma_max': sm, 'rho': int(b.rho), 'nfe': int(b.nfe),
                         'val_shape': float(b.shape_m), 'val_trend': float(b.trend_m), 'n_candidates': int(len(sub)), 'valgrid': a.valgrid}}
    json.dump(chosen, open(f'results/{a.dataname}_fast{a.k}_choice.json', 'w'), indent=1)
    print('CHOSEN', json.dumps(chosen))
    print('FASTK_SPECS', f'{b.solver}:{int(b.steps)}:0:{sm}:{int(b.rho)}')


if __name__ == '__main__':
    main()
