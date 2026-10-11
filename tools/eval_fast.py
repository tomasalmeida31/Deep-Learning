"""Shape / Trend (and optionally C2ST) for many synthetic CSVs, against a chosen real CSV,
using exactly the metadata construction of the authors' eval/eval_density.py and
eval/eval_detection.py (QualityReport; LogisticDetection). DiagnosticReport is skipped.

  python tools/eval_fast.py --dataname adultv_s0 --real real --glob 'synthetic/adultv_s0/grid/*.csv' \
      --out results/adultv_s0_gate.jsonl --workers 10 [--c2st] [--delete_csv]

--real real  = synthetic/<d>/real.csv (train', the reference used for all decisions);
--real val   = the 10% validation carve (n ~ 3k: its finite-sample floor is 1.2-1.4 pp shape,
               so it is NOT used for equivalence decisions; C2ST vs val can also fail on
               categories absent from val).

Writes one JSON line per CSV: {file, real, shape_err, trend_err, shape_score, trend_score,
c2st, nfe, ...sidecar}. shape_err = (1 - Column Shapes score) * 100 (percent), same for trend.
Idempotent: files already present in --out are skipped.

C2ST = SDMetrics LogisticDetection = 1 - mean_folds(max(AUC, 0.5) * 2 - 1): a transformed
discrimination AUC of a logistic classifier (1 = indistinguishable, 0 = perfectly separable). SDMetrics
0.11.1 scores it with a shuffled 3-fold CV split that takes NO random_state, so the value depends on
numpy's global RNG at call time. Results written before 1 Oct 2026 therefore carry an unseeded CV
split (the shape/trend scores are unaffected: they are deterministic). Now: numpy is seeded from
crc32(file name) + --c2st_seed before each call (recorded as `c2st_seed_used`), and --c2st_repeats k
scores k seeded splits and records c2st_mean / c2st_sd so the split noise is measured, not assumed.
"""
import argparse, copy, glob, json, os, sys, time, warnings, zlib
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.chdir(ROOT)
warnings.filterwarnings('ignore')
import pandas as pd
from multiprocessing import Pool

def reorder(real_data, syn_data, info):
    """verbatim logic of eval/eval_density.py::reorder, on a deep copy of info"""
    info = copy.deepcopy(info)
    num_col_idx = info['num_col_idx']; cat_col_idx = info['cat_col_idx']; target_col_idx = info['target_col_idx']
    if info['task_type'] == 'regression': num_col_idx += target_col_idx
    else: cat_col_idx += target_col_idx
    new_real = pd.concat([real_data[num_col_idx], real_data[cat_col_idx]], axis=1); new_real.columns = range(len(new_real.columns))
    new_syn = pd.concat([syn_data[num_col_idx], syn_data[cat_col_idx]], axis=1); new_syn.columns = range(len(new_syn.columns))
    metadata = info['metadata']
    columns = {int(k): v for k, v in metadata['columns'].items()}
    metadata['columns'] = {}
    for i in range(len(new_real.columns)):
        metadata['columns'][i] = columns[num_col_idx[i]] if i < len(num_col_idx) else columns[cat_col_idx[i - len(num_col_idx)]]
    return new_real, new_syn, metadata

def one(job):
    syn_path, real_path, info, want_c2st, delete_csv, c2st_seed, c2st_repeats = job
    import numpy as np
    from sdmetrics.reports.single_table import QualityReport
    t0 = time.time()
    real = pd.read_csv(real_path); syn = pd.read_csv(syn_path)
    real.columns = range(len(real.columns)); syn.columns = range(len(syn.columns))
    r, s, meta = reorder(real, syn, info)
    q = QualityReport(); q.generate(r, s, meta, verbose=False)
    props = q.get_properties()
    shape, trend = float(props['Score'][0]), float(props['Score'][1])
    out = {'file': os.path.basename(syn_path), 'real': os.path.basename(real_path), 'shape_score': shape, 'trend_score': trend,
           'shape_err': (1 - shape) * 100, 'trend_err': (1 - trend) * 100, 'n_syn': int(len(syn)), 'n_real': int(len(real))}
    if want_c2st:
        from sdmetrics.single_table import LogisticDetection
        base = (zlib.crc32(os.path.basename(syn_path).encode()) + int(c2st_seed)) % (2 ** 31)
        vals = []
        for rep in range(max(1, c2st_repeats)):
            np.random.seed((base + 7919 * rep) % (2 ** 31))     # the only RNG SDMetrics' KFold(shuffle=True) consumes
            vals.append(float(LogisticDetection.compute(real_data=r, synthetic_data=s, metadata=meta)))
        out['c2st'] = vals[0]; out['c2st_seed_used'] = base
        if c2st_repeats > 1:
            out['c2st_mean'] = float(np.mean(vals)); out['c2st_sd'] = float(np.std(vals, ddof=1)); out['c2st_repeats'] = len(vals)
    side = syn_path[:-4] + '.json'
    if os.path.exists(side): out.update(json.load(open(side)))
    out['eval_seconds'] = round(time.time() - t0, 1)
    if delete_csv: os.remove(syn_path)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataname', required=True)
    ap.add_argument('--real', default='val', help="'val' | 'real' | path to a csv")
    ap.add_argument('--glob', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--c2st', action='store_true')
    ap.add_argument('--delete_csv', action='store_true')
    ap.add_argument('--c2st_seed', type=int, default=0, help='added to crc32(file name) to seed the C2ST CV split')
    ap.add_argument('--c2st_repeats', type=int, default=1, help='>1: score this many seeded CV splits, record c2st_mean/c2st_sd')
    a = ap.parse_args()
    real_path = a.real if a.real.endswith('.csv') else f'synthetic/{a.dataname}/{a.real}.csv'
    assert os.path.exists(real_path), real_path
    info = json.load(open(f'data/{a.dataname}/info.json'))
    done = set()
    if os.path.exists(a.out):
        for line in open(a.out):
            if line.strip(): done.add(json.loads(line)['file'])
    files = sorted(f for f in glob.glob(a.glob) if os.path.basename(f) not in done)
    print(f'{len(done)} already evaluated, {len(files)} to do, real={real_path}, workers={a.workers}')
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    jobs = [(f, real_path, info, a.c2st, a.delete_csv, a.c2st_seed, a.c2st_repeats) for f in files]
    t0 = time.time(); n = 0
    with open(a.out, 'a') as fh, Pool(a.workers) as pool:
        for res in pool.imap_unordered(one, jobs, chunksize=1):
            fh.write(json.dumps(res) + '\n'); fh.flush(); n += 1
            if n % 20 == 0 or n == len(jobs):
                print(f'  {n}/{len(jobs)} evaluated, {time.time()-t0:.0f}s elapsed, last={res["file"]} shape_err={res["shape_err"]:.2f} trend_err={res["trend_err"]:.2f} ({res["eval_seconds"]}s)')
    print('OK eval_fast', a.out)

if __name__ == '__main__':
    main()
