"""Tests for the 1 Oct 2026 evaluation changes (review findings 5, 15, 17). CPU, smoke alias by default.

  python tools/test_eval.py [--dataname adultsmoke_s0]
"""
import argparse, json, os, sys, tempfile
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.chdir(ROOT)
import numpy as np, pandas as pd
from tools import eval_extra, eval_fast, eval_pairs_common

ap = argparse.ArgumentParser(); ap.add_argument('--dataname', default='adultsmoke_s0'); a = ap.parse_args()
info, num, cat, tgt = eval_extra.load_info(a.dataname)
real = pd.read_csv(f'synthetic/{a.dataname}/real.csv'); real.columns = range(real.shape[1])
tf = eval_extra.Transform(real, num, cat, tgt, info['task_type'])

# 1. DCR record space: X omits the target; X_full appends it (1 column for regression, one-hot for classification)
X, Xf = tf.X(real), tf.X_full(real)
extra = 1 if info['task_type'] == 'regression' else len(tf.labels)
assert Xf.shape == (X.shape[0], X.shape[1] + extra), (X.shape, Xf.shape)
assert np.array_equal(Xf[:, :X.shape[1]], X), 'X_full must extend X, not change it'
# changing ONLY the target changes X_full but not X (the review's reproduction of the omission)
alt = real.copy()
alt[tgt[0]] = alt[tgt[0]].iloc[::-1].to_numpy()
assert np.array_equal(tf.X(alt), X) and not np.array_equal(tf.X_full(alt), Xf), 'target must enter X_full only'
print(f'1. DCR space: features {X.shape[1]} cols, full {Xf.shape[1]} cols (target adds {extra}); target-only change invisible to X, visible to X_full')

# 2. DCR diagnostics: exact copies and ties are counted; score convention (tie counts against "closer to train")
R = Xf[:60]; T = Xf[60:120]; S = np.concatenate([R[:10], T[:10], (R[:5] + T[:5]) / 2])   # 10 copies of train, 10 of test, 5 midpoints
d = eval_extra.dcr(S, R, T, rows=1000, seed=0)
assert d['dcr_exact_copies'] == 10 / 25, d
assert d['dcr_ties'] >= 0 and abs(d['dcr_score_ties_split'] - (d['dcr_score'] + 0.5 * d['dcr_ties'])) < 1e-12, d
assert d['dcr_score'] + d['dcr_ties'] >= 10 / 25 - 1e-12, d
print(f'2. dcr(): exact copies {d["dcr_exact_copies"]:.2f}, ties {d["dcr_ties"]:.2f}, score {d["dcr_score"]:.3f}, ties-split {d["dcr_score_ties_split"]:.3f}')

# 3. seeded C2ST: same file -> identical scores across calls; repeats give a finite sd; different files -> different seeds
tmp = tempfile.mkdtemp(); p1 = f'{tmp}/syn_a.csv'; p2 = f'{tmp}/syn_b.csv'
v = pd.read_csv(f'synthetic/{a.dataname}/val.csv'); v.to_csv(p1, index=False); v.iloc[::-1].to_csv(p2, index=False)
r1 = eval_fast.one((p1, f'synthetic/{a.dataname}/real.csv', info, True, False, 0, 2))
r2 = eval_fast.one((p1, f'synthetic/{a.dataname}/real.csv', info, True, False, 0, 2))
r3 = eval_fast.one((p2, f'synthetic/{a.dataname}/real.csv', info, True, False, 0, 2))
assert r1['c2st'] == r2['c2st'] and r1['c2st_mean'] == r2['c2st_mean'], 'seeded C2ST must be reproducible'
assert r1['c2st_seed_used'] != r3['c2st_seed_used'] and r1['c2st_repeats'] == 2 and np.isfinite(r1['c2st_sd'])
assert abs(r1['shape_err'] - r3['shape_err']) < 1e-9, 'shape must not depend on row order'
print(f'3. C2ST seeded: {r1["c2st"]:.4f} reproducible; 2 splits mean {r1["c2st_mean"]:.4f} sd {r1["c2st_sd"]:.4f}; per-file seeds differ')

# 4. common-metric pairs: C(n,2) rows, one metric for every pair type, zero error for real-vs-real, bounded errors
rp = f'synthetic/{a.dataname}/real.csv'
cols, pairs, _ = eval_pairs_common.one((rp, rp, info, 10))
n = real.shape[1]
assert len(cols) == n and len(pairs) == n * (n - 1) // 2, (len(cols), len(pairs))
assert max(p['err'] for p in pairs) < 1e-9 and max(p['err_dep'] for p in pairs) < 1e-6, 'real vs real must score 0 error'
assert max(c['err_tv'] for c in cols) < 1e-9
types = {p['pair_type'] for p in pairs}; assert types <= {'num-num', 'num-cat', 'cat-cat'}, types
cols2, pairs2, _ = eval_pairs_common.one((p1, rp, info, 10))
assert all(0 <= p['err'] <= 100 and 0 <= p['err_dep'] <= 100 for p in pairs2)
numc = [c for c in cols2 if c['col_type'] == 'numerical']
assert all('mass_hi99' in c and 'qerr_0.99' in c for c in numc) and all('mass_hi99' not in c for c in cols2 if c['col_type'] == 'categorical')
print(f'4. pairs_common: {len(pairs)} pairs over types {sorted(types)}, real-vs-real error 0; val-vs-real mean err {np.mean([p["err"] for p in pairs2]):.2f}, err_dep {np.mean([p["err_dep"] for p in pairs2]):.2f}')
print('OK test_eval')
