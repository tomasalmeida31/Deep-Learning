"""Perfect-generator noise floor of the shipped Shape/Trend metrics: bootstrap resamples of
real.csv (train') scored against real.csv itself, n = |train'|. Writes results/<dataname>_floor.json
with mean and sd over --n_boot resamples. Used by gate_decide.py to report excess-over-floor
and to flag (dataset, metric) cells whose floor sd is too large to support the margin."""
import argparse, json, os, subprocess, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.chdir(ROOT)
import numpy as np, pandas as pd
ap = argparse.ArgumentParser(); ap.add_argument('--dataname', required=True); ap.add_argument('--n_boot', type=int, default=5); ap.add_argument('--workers', type=int, default=4)
a = ap.parse_args()
out = f'results/{a.dataname}_floor.json'
if os.path.exists(out):
    print('SKIP floor exists:', open(out).read()); sys.exit(0)
os.makedirs('results', exist_ok=True); tmp = f'synthetic/{a.dataname}/floor'; os.makedirs(tmp, exist_ok=True)
real = pd.read_csv(f'synthetic/{a.dataname}/real.csv')
for b in range(a.n_boot):
    real.sample(n=len(real), replace=True, random_state=b).reset_index(drop=True).to_csv(f'{tmp}/boot{b}.csv', index=False, float_format='%.6g')
jl = f'{tmp}/floor.jsonl'
if os.path.exists(jl): os.remove(jl)
subprocess.check_call([sys.executable, 'tools/eval_fast.py', '--dataname', a.dataname, '--real', 'real', '--glob', f'{tmp}/boot*.csv', '--out', jl, '--workers', str(a.workers), '--delete_csv'])
rows = [json.loads(l) for l in open(jl)]
res = {'dataname': a.dataname, 'n_boot': a.n_boot, 'n': int(len(real)),
       'shape_floor': float(np.mean([r['shape_err'] for r in rows])), 'shape_floor_sd': float(np.std([r['shape_err'] for r in rows], ddof=1)),
       'trend_floor': float(np.mean([r['trend_err'] for r in rows])), 'trend_floor_sd': float(np.std([r['trend_err'] for r in rows], ddof=1))}
json.dump(res, open(out, 'w'), indent=1); print('OK floor_boot', json.dumps(res))
