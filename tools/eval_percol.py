"""Ablation plan E5: per-COLUMN and per-PAIR fidelity for many synthetic CSVs (eval_fast keeps only the aggregates).

  python tools/eval_percol.py --dataname adultv_s0 --real real --glob 'synthetic/adultv_s0/percol/*.csv' \
      --out results/adultv_s0_percol.jsonl --out_pairs results/adultv_s0_percol_pairs.jsonl --workers 5

Same metadata construction as tools/eval_fast.py (reorder = the authors' eval_density logic) and the same
SDMetrics QualityReport; one JSON line per column (Column Shapes details) and per column pair (Column Pair Trends
details). err = (1 - score) * 100, the same units as eval_fast's shape_err / trend_err; the mean of the per-column
scores equals eval_fast's shape_score (this was checked by tools/test_percol.py, which fell out of date and is not shipped). Idempotent on `file`.
"""
import argparse, glob, json, os, sys, time, warnings
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.chdir(ROOT)
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from multiprocessing import Pool
from tools.eval_fast import reorder


def column_table(info, real):
    """reordered index -> (name, col_type, is_target, cardinality / n_unique in real)"""
    num, cat, tgt = list(info['num_col_idx']), list(info['cat_col_idx']), list(info['target_col_idx'])
    if info['task_type'] == 'regression': num = num + tgt
    else: cat = cat + tgt
    order = num + cat; names = {int(k): v for k, v in info['idx_name_mapping'].items()}
    out = {}
    for i, orig in enumerate(order):
        out[i] = {'column': names.get(orig, str(orig)), 'orig_idx': int(orig), 'col_type': 'numerical' if i < len(num) else 'categorical',
                  'is_target': bool(orig in tgt), 'n_unique_real': int(real[orig].nunique())}
    return out


def _score_col(df):
    return 'Score' if 'Score' in df.columns else 'Quality Score'


def one(job):
    syn_path, real_path, info = job
    from sdmetrics.reports.single_table import QualityReport
    t0 = time.time()
    real = pd.read_csv(real_path); syn = pd.read_csv(syn_path)
    real.columns = range(len(real.columns)); syn.columns = range(len(syn.columns))
    ct = column_table(info, real)
    r, s, meta = reorder(real, syn, info)
    q = QualityReport(); q.generate(r, s, meta, verbose=False)
    props = q.get_properties(); shape = float(props['Score'][0])
    side = syn_path[:-4] + '.json'; sc = json.load(open(side)) if os.path.exists(side) else {}
    keep = {k: sc[k] for k in ('solver', 'steps', 'nfe', 'seed', 'sigma_max', 'rho', 'student', 'tag') if k in sc}
    base = {'file': os.path.basename(syn_path), 'real': os.path.basename(real_path), **keep}
    cols, pairs = [], []
    d = q.get_details('Column Shapes'); scol = _score_col(d)
    for _, row in d.iterrows():
        i = int(row['Column']); c = ct[i]; v = float(row[scol])
        cols.append({**base, **c, 'metric': row['Metric'], 'score': v, 'err': (1 - v) * 100 if v == v else float('nan'), 'file_shape_score': shape})
    d = q.get_details('Column Pair Trends'); scol = _score_col(d)
    for _, row in d.iterrows():
        i, j = int(row['Column 1']), int(row['Column 2']); a_, b_ = ct[i], ct[j]; v = float(row[scol])
        tps = sorted([a_['col_type'][:3], b_['col_type'][:3]], key=lambda t: (t != 'num', t))
        pairs.append({**base, 'col_a': a_['column'], 'col_b': b_['column'], 'pair_type': '-'.join(tps),
                      'involves_target': a_['is_target'] or b_['is_target'], 'metric': row['Metric'], 'score': v,
                      'err': (1 - v) * 100 if v == v else float('nan')})
    return cols, pairs, round(time.time() - t0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataname', required=True); ap.add_argument('--real', default='real')
    ap.add_argument('--glob', required=True); ap.add_argument('--out', required=True); ap.add_argument('--out_pairs', required=True)
    ap.add_argument('--workers', type=int, default=4)
    a = ap.parse_args()
    real_path = a.real if a.real.endswith('.csv') else f'synthetic/{a.dataname}/{a.real}.csv'
    info = json.load(open(f'data/{a.dataname}/info.json'))
    done = set()
    if os.path.exists(a.out):
        for line in open(a.out):
            if line.strip(): done.add(json.loads(line)['file'])
    files = sorted(f for f in glob.glob(a.glob) if os.path.basename(f) not in done)
    print(f'{len(done)} files already evaluated, {len(files)} to do, workers={a.workers}')
    t0 = time.time(); n = 0
    with open(a.out, 'a') as fc, open(a.out_pairs, 'a') as fp, Pool(a.workers) as pool:
        for cols, pairs, sec in pool.imap_unordered(one, [(f, real_path, info) for f in files], chunksize=1):
            for p in pairs: fp.write(json.dumps(p) + '\n')
            fp.flush()
            for c in cols: fc.write(json.dumps(c) + '\n')           # columns last: `file` in --out marks the file done
            fc.flush(); n += 1
            if n % 10 == 0 or n == len(files): print(f'  {n}/{len(files)} in {time.time()-t0:.0f}s (last {sec}s)')
    print('OK eval_percol', a.out)


if __name__ == '__main__':
    main()
