"""Teacher seed spread: E99 shape and trend error for each training seed of each table, plus the
leaky-setup teacher (seed 0) for comparison.
Usage: python tools/seed_spread_report.py > results/seed_spread_report.txt"""
import json, os
import numpy as np

def load(path):
    if not os.path.exists(path): return []
    return [json.loads(l) for l in open(path) if l.strip()]

def e99(records):
    return [r for r in records if r.get('solver')=='heun' and int(r.get('steps',-1))==50 and int(r.get('sigma_max',-1))==80 and int(r.get('rho',-1))==7 and float(r.get('s_churn',0))==0]

def mean(records, key):
    v=[r[key] for r in records if key in r and isinstance(r[key],(int,float)) and not (isinstance(r[key],float) and np.isnan(r[key]))]
    return float(np.mean(v)) if v else float('nan')

for d in ['adult','default','news']:
    print(f'\n{d}: leak-free seed spread at E99 (shape, trend)')
    for k in (0,1,2):
        A=f'{d}v_s{k}'
        g=e99(load(f'results/{A}_gate.jsonl'))
        print(f'  s{k}: shape={mean(g,"shape_err"):.3f}  trend={mean(g,"trend_err"):.3f}  (n={len(g)})')
    A=f'{d}L_s0'
    g=e99(load(f'results/{A}_gate.jsonl'))
    print(f'  Lk0: shape={mean(g,"shape_err"):.3f}  trend={mean(g,"trend_err"):.3f}  (n={len(g)})  <- leaky protocol, seed 0')
