"""Create derived dataset aliases for the leak-free protocol.

For a processed dataset NAME (data/NAME/{X_num,X_cat,y}_{train,test}.npy, info.json)
this creates, for each requested seed k:

  data/NAMEv_s{k}   train slot = 90% of the shipped train split ("train'")
                    test  slot = the remaining 10% ("val")  -> the VAE's selection
                                 signal (vae/main.py uses X_test) becomes validation
  data/NAMEL_s{k}   train slot = the same train'
                    test  slot = the shipped test split      -> the paper's (leaky)
                                 protocol on identical training data
  synthetic/NAMEv_s{k}/real.csv = train'   test.csv = shipped test   val.csv = val
  synthetic/NAMEL_s{k}/real.csv = train'   test.csv = shipped test   val.csv = val

The val carve uses --split_seed (default 1234) and is identical for every k,
so the aliases of one NAME differ only in the model seed used later.
--smoke additionally creates NAMEsmoke_s0 with --smoke_rows train' rows for CPU tests.
Idempotent: existing aliases are verified and left alone.

Unseen-category repair (fix_unseen). The authors' encoder maps a category unseen in the
train slot to index max+1, outside the VAE's embedding table, which crashes vae/main.py's
eval block; val rows holding such a category are therefore moved into train'. The target
column takes part in this repair ONLY when it is categorical (binclass / multiclass).
  * Historical behaviour (every alias created before 1 Oct 2026): the target was always
    included, so for a REGRESSION dataset val rows with a target value unseen in train'
    (i.e. unique values, which for news are tail values of `shares`) were moved too.
    `--legacy_target_in_fix` reproduces that split bit-for-bit; aliases record which
    rule made them in info.json['_split_fix'] (absent = legacy).
  * `--audit` reproduces the carve without writing anything and reports, per moved row,
    which column(s) triggered the move under both rules, plus the effect on the val
    target distribution; writes results/<NAME>_split_audit.json.
Split indices are also archived to data/<alias>/split_indices.npz for provenance.
"""
import argparse, json, os, shutil, sys
import numpy as np
import pandas as pd

def load_split(d, split):
    y = np.load(f'{d}/y_{split}.npy', allow_pickle=True)
    xn = np.load(f'{d}/X_num_{split}.npy', allow_pickle=True) if os.path.exists(f'{d}/X_num_{split}.npy') else None
    xc = np.load(f'{d}/X_cat_{split}.npy', allow_pickle=True) if os.path.exists(f'{d}/X_cat_{split}.npy') else None
    return xn, xc, y

def save_split(d, split, xn, xc, y):
    os.makedirs(d, exist_ok=True)
    np.save(f'{d}/y_{split}.npy', y)
    if xn is not None: np.save(f'{d}/X_num_{split}.npy', xn)
    if xc is not None: np.save(f'{d}/X_cat_{split}.npy', xc)

def sub(a, idx):
    return None if a is None else a[idx]

def repair_columns(d_src, task_type, include_target):
    """Columns whose val values must be a subset of train' values: categorical features, plus the
    target when it is categorical (or when include_target forces the historical rule)."""
    _, xc, y = load_split(d_src, 'train')
    cols = [] if xc is None else [(f'cat{j}', xc[:, j]) for j in range(xc.shape[1])]
    if include_target or task_type != 'regression':
        cols.append(('target', y[:, 0] if y.ndim == 2 else y))
    return cols

def fix_unseen(d_src, val_idx, tr_idx, task_type='binclass', include_target=False, trace=None):
    cols = repair_columns(d_src, task_type, include_target)
    moved = 0
    for it in range(10):
        bad = np.zeros(len(val_idx), dtype=bool)
        for name, c in cols:
            seen = set(np.unique(c[tr_idx]).tolist())
            hit = np.array([v not in seen for v in c[val_idx]], dtype=bool)
            if trace is not None:
                for i in np.flatnonzero(hit): trace.setdefault(int(val_idx[i]), []).append((it, name))
            bad |= hit
        if not bad.any(): break
        tr_idx = np.sort(np.concatenate([tr_idx, val_idx[bad]])); val_idx = val_idx[~bad]; moved += int(bad.sum())
    assert not bad.any(), 'could not make val categories a subset of train categories'
    if moved: print(f'moved {moved} val rows with unseen categories into train (target included: {include_target or task_type != "regression"})')
    return val_idx, tr_idx

def make_alias(src_name, alias, tr_idx, te_source, info, train_csv, test_csv, val_csv, split_fix):
    """te_source: ('val', idx) or ('test', None)"""
    d_src = f'data/{src_name}'; d = f'data/{alias}'
    if os.path.exists(f'{d}/info.json'):
        chk = json.load(open(f'{d}/info.json'))
        assert chk.get('_alias_of') == src_name and chk['train_num'] == len(tr_idx), \
            f'{alias} exists with different content (existing train_num {chk["train_num"]}, requested {len(tr_idx)}; ' \
            f'existing _split_fix={chk.get("_split_fix", "legacy")}, requested {split_fix})'
        if not os.path.exists(f'{d}/split_indices.npz'):
            np.savez(f'{d}/split_indices.npz', train_idx=tr_idx, val_idx=te_source[1] if te_source[0] == 'val' else np.array([], dtype=int))
        return 'exists'
    xn, xc, y = load_split(d_src, 'train')
    save_split(d, 'train', sub(xn, tr_idx), sub(xc, tr_idx), y[tr_idx])
    if te_source[0] == 'val':
        vi = te_source[1]
        save_split(d, 'test', sub(xn, vi), sub(xc, vi), y[vi])
        n_test = len(vi)
    else:
        xn_t, xc_t, y_t = load_split(d_src, 'test')
        save_split(d, 'test', xn_t, xc_t, y_t)
        n_test = len(y_t)
    inf = dict(info); inf['train_num'] = int(len(tr_idx)); inf['test_num'] = int(n_test); inf['_alias_of'] = src_name; inf['_split_fix'] = split_fix
    json.dump(inf, open(f'{d}/info.json', 'w'), indent=2)
    np.savez(f'{d}/split_indices.npz', train_idx=tr_idx, val_idx=te_source[1] if te_source[0] == 'val' else np.array([], dtype=int))
    s = f'synthetic/{alias}'; os.makedirs(s, exist_ok=True)
    train_csv.to_csv(f'{s}/real.csv', index=False)
    test_csv.to_csv(f'{s}/test.csv', index=False)
    val_csv.to_csv(f'{s}/val.csv', index=False)
    train_csv.to_csv(f'{d}/train.csv', index=False)   # not read by the model; kept for symmetry
    return 'created'

def carve(n, split_seed, val_frac):
    rng = np.random.RandomState(split_seed)
    perm = rng.permutation(n)
    n_val = int(round(val_frac * n))
    return np.sort(perm[:n_val]), np.sort(perm[n_val:])

def audit(name, d_src, info, split_seed, val_frac):
    """Reproduce the carve under the legacy rule (target always in the repair) and the task-aware
    rule, and report what differs. Writes results/<name>_split_audit.json. Nothing else is touched."""
    task = info['task_type']
    _, _, y = load_split(d_src, 'train'); n = len(y)
    y1 = (y[:, 0] if y.ndim == 2 else y).astype(float) if task == 'regression' else None
    val0, tr0 = carve(n, split_seed, val_frac)
    out = {'dataset': name, 'task_type': task, 'n_train_shipped': int(n), 'n_val_initial': int(len(val0)), 'split_seed': split_seed}
    for rule, inc in (('legacy_target_in_fix', True), ('task_aware', False)):
        trace = {}
        v, t = fix_unseen(d_src, val0.copy(), tr0.copy(), task, include_target=inc, trace=trace)
        moved = sorted(set(val0.tolist()) - set(v.tolist()))
        reasons = {str(i): sorted(set(c for _, c in trace.get(i, []))) for i in moved}
        by_col = {}
        for i in moved:
            for c in reasons[str(i)]: by_col[c] = by_col.get(c, 0) + 1
        only_target = [i for i in moved if reasons[str(i)] == ['target']]
        rec = {'n_moved': len(moved), 'n_val_final': int(len(v)), 'n_train_final': int(len(t)), 'moved_rows': moved,
               'reasons': reasons, 'moved_by_column': by_col, 'n_moved_only_because_of_target': len(only_target)}
        if y1 is not None:
            qs = [0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0]
            rec['val_target_quantiles_before'] = np.quantile(y1[val0], qs).tolist()
            rec['val_target_quantiles_after'] = np.quantile(y1[v], qs).tolist()
            rec['moved_target_values'] = sorted(y1[moved].tolist()) if moved else []
            rec['moved_target_quantile_rank_in_shipped_train'] = [float((y1 < y1[i]).mean()) for i in moved]
        out[rule] = rec
    # does a matching alias exist, and which rule does it correspond to?
    for k in range(3):
        d = f'data/{name}v_s{k}'
        if os.path.exists(f'{d}/info.json'):
            chk = json.load(open(f'{d}/info.json'))
            match = [r for r in ('legacy_target_in_fix', 'task_aware') if out[r]['n_train_final'] == chk['train_num']]
            yv = np.load(f'{d}/y_test.npy', allow_pickle=True)
            out[f'{name}v_s{k}'] = {'train_num': chk['train_num'], 'test_num': chk['test_num'], '_split_fix': chk.get('_split_fix', 'absent (legacy)'),
                                     'train_num_matches_rule': match, 'val_rows_on_disk': int(len(yv))}
    os.makedirs('results', exist_ok=True)
    p = f'results/{name}_split_audit.json'; json.dump(out, open(p, 'w'), indent=1)
    L, T = out['legacy_target_in_fix'], out['task_aware']
    print(f'AUDIT {name} ({task}): legacy rule moved {L["n_moved"]} rows ({L["moved_by_column"]}; {L["n_moved_only_because_of_target"]} only because of the target); '
          f'task-aware rule moves {T["n_moved"]} rows ({T["moved_by_column"]}). Val size {L["n_val_final"]} vs {T["n_val_final"]}.')
    if y1 is not None and L['moved_target_values']:
        r = L['moved_target_quantile_rank_in_shipped_train']
        print(f'  moved target values: min {min(L["moved_target_values"]):.4g} max {max(L["moved_target_values"]):.4g}; quantile ranks in shipped train: min {min(r):.3f} median {float(np.median(r)):.3f} max {max(r):.3f}')
        print(f'  val target quantiles (0,1,5,25,50,75,95,99,100 %) before: {[round(q, 1) for q in L["val_target_quantiles_before"]]}')
        print(f'                                              after : {[round(q, 1) for q in L["val_target_quantiles_after"]]}')
    for k in range(3):
        if f'{name}v_s{k}' in out: print(f'  {name}v_s{k}: ' + json.dumps(out[f'{name}v_s{k}']))
    print('wrote', p)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataname', required=True)
    ap.add_argument('--seeds', type=int, nargs='+', default=[0])
    ap.add_argument('--split_seed', type=int, default=1234)
    ap.add_argument('--val_frac', type=float, default=0.1)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--smoke_rows', type=int, default=512)
    ap.add_argument('--legacy_target_in_fix', action='store_true', help='reproduce the pre-Oct-2026 repair rule (target always included, even for regression)')
    ap.add_argument('--audit', action='store_true', help='report what the repair moved under both rules; writes nothing but results/<name>_split_audit.json')
    a = ap.parse_args()
    name = a.dataname
    d_src = f'data/{name}'
    assert os.path.exists(f'{d_src}/info.json'), f'run process_dataset.py first: {d_src}/info.json missing'
    info = json.load(open(f'{d_src}/info.json'))
    if a.audit:
        audit(name, d_src, info, a.split_seed, a.val_frac); return
    task = info['task_type']
    split_fix = 'legacy_target_in_fix' if a.legacy_target_in_fix else 'task_aware'
    _, _, y = load_split(d_src, 'train')
    n = len(y)
    val_idx, tr_idx = carve(n, a.split_seed, a.val_frac)
    val_idx, tr_idx = fix_unseen(d_src, val_idx, tr_idx, task, include_target=a.legacy_target_in_fix)
    train_df = pd.read_csv(f'{d_src}/train.csv'); test_df = pd.read_csv(f'{d_src}/test.csv')
    assert len(train_df) == n, 'train.csv and y_train.npy disagree'
    tr_csv, val_csv = train_df.iloc[tr_idx].reset_index(drop=True), train_df.iloc[val_idx].reset_index(drop=True)
    for k in a.seeds:
        r1 = make_alias(name, f'{name}v_s{k}', tr_idx, ('val', val_idx), info, tr_csv, test_df, val_csv, split_fix)
        r2 = make_alias(name, f'{name}L_s{k}', tr_idx, ('test', None), info, tr_csv, test_df, val_csv, split_fix)
        print(f'{name}v_s{k}: {r1} (train {len(tr_idx)}, test-slot=val {len(val_idx)}); {name}L_s{k}: {r2} (train {len(tr_idx)}, test-slot=test {len(test_df)}); split_fix={split_fix}')
    if a.smoke:
        sm_tr = tr_idx[:a.smoke_rows]; sm_val = val_idx[:max(32, a.smoke_rows // 4)]
        sm_val, sm_tr = fix_unseen(d_src, sm_val, sm_tr, task, include_target=a.legacy_target_in_fix)
        r = make_alias(name, f'{name}smoke_s0', sm_tr, ('val', sm_val), info,
                       train_df.iloc[sm_tr].reset_index(drop=True), test_df.iloc[:256], train_df.iloc[sm_val].reset_index(drop=True), split_fix)
        print(f'{name}smoke_s0: {r} (train {len(sm_tr)}, val {len(sm_val)})')
    print('OK make_val_split', name)

if __name__ == '__main__':
    main()
