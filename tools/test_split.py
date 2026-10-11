"""Regression tests for tools/make_val_split.py (review finding 1): the unseen-category repair must not treat a
REGRESSION target as a category, the historical rule must stay reproducible, and the alias guard must refuse a
silently different split. Runs in a temporary directory with synthetic data; touches nothing under data/.

  python tools/test_split.py
"""
import json, os, shutil, subprocess, sys, tempfile
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
SCRIPT = os.path.join(ROOT, 'tools', 'make_val_split.py')
from tools.make_val_split import carve


def make_dataset(d, task, n=400, seed=0):
    rng = np.random.RandomState(seed)
    xn = rng.randn(n, 2); xc = np.stack([rng.choice(['a', 'b', 'c'], n), rng.choice(['x', 'y'], n)], 1).astype(object)
    xc[0, 0] = 'rare'                                                 # one category that appears once
    if task == 'regression':
        y = rng.choice([100., 200., 300., 500., 1000.], n)               # a few repeated values ...
        val0, _ = carve(n, 1234, 0.25)
        y[val0[:5]] = [1e6, 2e6, 3e6, 4e6, 5e6]                          # ... plus 5 unique tail values that land in val
    else:
        y = rng.choice([0, 1], n)
    os.makedirs(d, exist_ok=True)
    np.save(f'{d}/X_num_train.npy', xn); np.save(f'{d}/X_cat_train.npy', xc); np.save(f'{d}/y_train.npy', y)
    np.save(f'{d}/X_num_test.npy', xn[:50]); np.save(f'{d}/X_cat_test.npy', xc[:50]); np.save(f'{d}/y_test.npy', y[:50])
    df = pd.DataFrame({'n0': xn[:, 0], 'n1': xn[:, 1], 'c0': xc[:, 0], 'c1': xc[:, 1], 'target': y})
    df.to_csv(f'{d}/train.csv', index=False); df.iloc[:50].to_csv(f'{d}/test.csv', index=False)
    json.dump({'task_type': task, 'num_col_idx': [0, 1], 'cat_col_idx': [2, 3], 'target_col_idx': [4], 'train_num': n, 'test_num': 50}, open(f'{d}/info.json', 'w'))


def run(args, cwd):
    r = subprocess.run([sys.executable, SCRIPT] + args, cwd=cwd, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    return r.stdout


tmp = tempfile.mkdtemp(prefix='split_test_')
try:
    for name, task in (('fakereg', 'regression'), ('fakecls', 'binclass')):
        make_dataset(f'{tmp}/data/{name}', task)
        out = run(['--dataname', name, '--audit', '--val_frac', '0.25'], tmp)
        aud = json.load(open(f'{tmp}/results/{name}_split_audit.json'))
        L, T = aud['legacy_target_in_fix'], aud['task_aware']
        if task == 'regression':
            assert L['n_moved'] > T['n_moved'] and L['n_moved_only_because_of_target'] == L['n_moved'] - T['n_moved'], (L, T)
            assert T['moved_by_column'].get('target', 0) == 0, 'task-aware rule must never move a row because of a regression target'
            assert L['moved_by_column'].get('target', 0) == 5 and set(L['moved_target_values']) >= {1e6, 2e6, 3e6, 4e6, 5e6}, L
        else:
            assert L['n_moved'] == T['n_moved'] and L['moved_rows'] == T['moved_rows'], 'for a categorical target both rules must agree'
        # create aliases under the new default (task-aware), then verify the legacy rule is refused on an existing alias when it differs
        run(['--dataname', name, '--seeds', '0', '--val_frac', '0.25'], tmp)
        info = json.load(open(f'{tmp}/data/{name}v_s0/info.json')); assert info['_split_fix'] == 'task_aware', info
        assert os.path.exists(f'{tmp}/data/{name}v_s0/split_indices.npz'), 'split indices must be archived'
        idx = np.load(f'{tmp}/data/{name}v_s0/split_indices.npz'); assert len(idx['train_idx']) == info['train_num'] and len(idx['val_idx']) == info['test_num']
        assert set(idx['train_idx']).isdisjoint(idx['val_idx']), 'train and val indices overlap'
        r = subprocess.run([sys.executable, SCRIPT, '--dataname', name, '--seeds', '0', '--val_frac', '0.25', '--legacy_target_in_fix'], cwd=tmp, capture_output=True, text=True)
        if task == 'regression':
            assert r.returncode != 0 and 'exists with different content' in (r.stdout + r.stderr), 'the alias guard must refuse a different split'
        else:
            assert r.returncode == 0 and 'exists' in r.stdout, 'identical split under both rules must be accepted as existing'
        print(f'{name} ({task}): legacy moved {L["n_moved"]} ({L["moved_by_column"]}), task-aware moved {T["n_moved"]} ({T["moved_by_column"]}); alias guard ok')
    print('OK test_split')
finally:
    shutil.rmtree(tmp, ignore_errors=True)
