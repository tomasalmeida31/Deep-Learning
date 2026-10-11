"""Verdict v2 §6.1: does any sigma_max < 1 beat F_1 / F_2 (chosen at sigma_max = 1, the grid edge) on the val carve?

  python tools/pick_fast_lo.py --dataname adultv_s0
      reads  results/<A>_fastlo_valgrid.jsonl  (eval_fast --real val over sample_grid --grid fast_lo, tag v)
             results/<A>_fast_choice.json      (the E1 choice, incl. val_shape of F_1, F_2)
      writes results/<A>_fastlo_choice.json and prints FASTLO_SPECS (configs that beat the E1 choice on val; may be empty)
Same selection rule as pick_fast.py (lowest mean val shape over 5 seeds, ties by trend), applied per budget k in {1, 2}
to the UNION of the E1 candidates with nfe <= k and the new sigma_max < 1 candidates.
"""
import argparse, json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.chdir(ROOT)
import pandas as pd

KEY = ['solver', 'steps', 'sigma_max', 'rho', 'nfe']

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--dataname', required=True); a = ap.parse_args()
    A = a.dataname
    lo = pd.DataFrame([json.loads(l) for l in open(f'results/{A}_fastlo_valgrid.jsonl') if l.strip()])
    hi = pd.DataFrame([json.loads(l) for l in open(f'results/{A}_fast_valgrid.jsonl') if l.strip()])
    df = pd.concat([hi, lo], ignore_index=True).drop_duplicates(subset=KEY + ['seed'], keep='last')
    g = df.groupby(KEY).agg(n=('seed', 'count'), shape_m=('shape_err', 'mean'), shape_sd=('shape_err', 'std'), trend_m=('trend_err', 'mean')).reset_index()
    assert (g.n == 5).all(), g[g.n != 5]
    old = json.load(open(f'results/{A}_fast_choice.json')); out, specs = {}, []
    for k in (1, 2):
        sub = g[g.nfe <= k].sort_values(['shape_m', 'trend_m'])
        print(f'budget {k} NFE (E1 choice: {old[str(k)]["solver"]} N{old[str(k)]["steps"]} smax {old[str(k)]["sigma_max"]} val shape {old[str(k)]["val_shape"]:.2f}); top 6 incl. sigma_max < 1:')
        for _, r in sub.head(6).iterrows():
            print(f'   {r.solver:8s} N={int(r.steps)} smax={r.sigma_max:<4} shape {r.shape_m:6.2f}±{r.shape_sd:4.2f} trend {r.trend_m:6.2f}')
        b = sub.iloc[0]; improved = bool(b.shape_m < old[str(k)]['val_shape'] - 1e-9)
        curve = {str(r.sigma_max): round(float(r.shape_m), 3) for _, r in g[(g.nfe == 1) if k == 1 else ((g.nfe == 2) & (g.solver == 'euler'))].sort_values('sigma_max').iterrows()}
        out[k] = {'solver': b.solver, 'steps': int(b.steps), 'sigma_max': float(b.sigma_max), 'rho': int(b.rho), 'nfe': int(b.nfe), 'val_shape': float(b.shape_m),
                  'val_trend': float(b.trend_m), 'beats_E1_choice': improved, 'E1_val_shape': old[str(k)]['val_shape'], 'val_shape_vs_sigma_max': curve}
        if improved: specs.append(f'{b.solver}:{int(b.steps)}:0:{b.sigma_max}:{int(b.rho)}')
    json.dump(out, open(f'results/{A}_fastlo_choice.json', 'w'), indent=1)
    print('CHOSEN', json.dumps(out)); print('FASTLO_SPECS', ' '.join(sorted(set(specs))))

if __name__ == '__main__':
    main()
