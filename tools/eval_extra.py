"""Downstream utility (MLE) and memorisation (DCR) for many synthetic CSVs — the C4 metrics.

  python tools/eval_extra.py --dataname adultv_s0 --glob 'synthetic/adultv_s0/student_cdA_s0/*_sel_s*.csv' \
      --out results/adultv_s0_student_cdA_s0_extra.jsonl --workers 5 [--dcr_rows 5000] [--seed 0]
  python tools/eval_extra.py --dataname adultv_s0 --glob synthetic/adultv_s0/real.csv --out results/adultv_s0_real_extra.jsonl
      (the real-data reference: MLE ceiling and the DCR floor of real-vs-real)

MLE follows TabSyn's protocol (eval/eval_mle.py): a classifier/regressor is trained ON THE SYNTHETIC
table and tested on the real, held-out `synthetic/<d>/test.csv` (the shipped test split, never train'
or val). Implementation differs from the authors' in three deliberate ways, all documented here:
(1) features are built with a clean transform (per-column min/max from real train' for numerics,
one-hot fitted on real train' for categoricals) instead of eval/mle/mle.py::feat_transform, which
reuses the first numeric column's min/max for every column and uses a removed sklearn kwarg;
(2) the XGBoost grid is n_estimators {50, 100} x max_depth {4, 8} on CPU 'hist', chosen on a 1/9
held-out slice of the synthetic table, fixed seed; (3) reported: binary -> roc_auc, accuracy;
multiclass -> macro_f1, accuracy; regression -> rmse, r2. Same protocol for every file, so
comparisons are fair; absolute values are not the paper's.

DCR follows eval/eval_dcr.py: L1 distance in a space of range-normalised numerics + one-hot
categoricals; score = fraction of synthetic rows closer to the training set than to the test
set; 0.5 is ideal, higher means memorisation. Two deliberate differences: the training reference is
subsampled to the test-set size with a fixed seed (the authors compare against sets of unequal
size, which biases the score toward the larger set) and only --dcr_rows synthetic rows are used.
Run the same on real.csv to get the real-vs-real floor. Idempotent: files already in --out are skipped.

DCR record space (--dcr_space). 'features' (the historical default; every result file written
before 1 Oct 2026) measures distances in the PREDICTOR-feature space only: the transform above is
the MLE feature transform and omits the target column. 'full' appends the target (range-normalised
numeric for regression, one-hot for classification) so the distance is over the whole record.
Keys are suffixed `_full` in that mode so the two are never confused. Both modes also report
`dcr_ties` (fraction of synthetic rows exactly equidistant to train and test; the score counts a tie
as "not closer to train") and `dcr_exact_copies` (fraction with L1 distance 0 to some train' row).
"""
import argparse, glob, json, os, sys, time, warnings
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.chdir(ROOT)
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from multiprocessing import Pool


def load_info(dataname):
    info = json.load(open(f'data/{dataname}/info.json'))
    num, cat, tgt = list(info['num_col_idx']), list(info['cat_col_idx']), list(info['target_col_idx'])
    return info, num, cat, tgt


class Transform:
    """Fitted on real train' only; applied identically to synthetic, real and test."""
    def __init__(self, real, num, cat, tgt, task):
        self.num, self.cat, self.tgt, self.task = num, cat, tgt, task
        self.lo = real[num].min().to_numpy(float); self.hi = real[num].max().to_numpy(float)
        self.rng = np.where(self.hi > self.lo, self.hi - self.lo, 1.0)
        self.cats = {c: sorted(real[c].astype(str).unique().tolist()) for c in cat}
        self.labels = None if task == 'regression' else sorted(real[tgt[0]].astype(str).unique().tolist())
        if task == 'regression':
            self.tgt_lo, self.tgt_hi = float(real[tgt[0]].min()), float(real[tgt[0]].max())

    def X(self, df):
        parts = []
        if self.num: parts.append((df[self.num].to_numpy(float) - self.lo) / self.rng)
        for c in self.cat:
            v = df[c].astype(str).to_numpy(); parts.append(np.stack([(v == k).astype(float) for k in self.cats[c]], 1))
        return np.concatenate(parts, 1) if parts else np.zeros((len(df), 0))

    def X_full(self, df):
        """Predictor features plus the target column: the whole record, for DCR."""
        X = self.X(df)
        col = df[self.tgt[0]]
        if self.task == 'regression':
            t = col.to_numpy(float)[:, None]
            lo, hi = self.tgt_lo, self.tgt_hi
            t = (t - lo) / (hi - lo if hi > lo else 1.0)
        else:
            v = col.astype(str).to_numpy(); t = np.stack([(v == k).astype(float) for k in self.labels], 1)
        return np.concatenate([X, t], 1)

    def y(self, df):
        col = df[self.tgt[0]]
        if self.task == 'regression': return col.to_numpy(float)
        idx = {k: i for i, k in enumerate(self.labels)}
        return np.array([idx.get(str(v), -1) for v in col])


def mle(Xs, ys, Xt, yt, task, seed):
    from xgboost import XGBClassifier, XGBRegressor
    from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, mean_squared_error, r2_score
    rng = np.random.RandomState(seed); n = len(Xs); perm = rng.permutation(n); nv = max(1, n // 9)
    va, tr = perm[:nv], perm[nv:]
    keep = ys >= 0 if task != 'regression' else np.isfinite(ys)
    tr = tr[keep[tr]]; va = va[keep[va]]
    best, best_score = None, -np.inf
    for ne in (50, 100):
        for md in (4, 8):
            kw = dict(n_estimators=ne, max_depth=md, tree_method='hist', random_state=seed, n_jobs=1, verbosity=0)
            if task == 'regression':
                m = XGBRegressor(**kw).fit(Xs[tr], ys[tr]); sc = -mean_squared_error(ys[va], m.predict(Xs[va]))
            else:
                if len(np.unique(ys[tr])) < 2: continue
                m = XGBClassifier(**kw).fit(Xs[tr], ys[tr]); p = m.predict(Xs[va]); sc = accuracy_score(ys[va], p)
            if sc > best_score: best, best_score = m, sc
    if best is None: return {'mle_error': 'degenerate synthetic labels'}
    if task == 'regression':
        pr = best.predict(Xt); return {'mle_rmse': float(np.sqrt(mean_squared_error(yt, pr))), 'mle_r2': float(r2_score(yt, pr))}
    pr = best.predict(Xt); out = {'mle_accuracy': float(accuracy_score(yt, pr))}
    if task == 'binclass':
        prob = best.predict_proba(Xt); k = list(best.classes_).index(1) if 1 in best.classes_ else 0
        out['mle_roc_auc'] = float(roc_auc_score(yt, prob[:, k])) if len(best.classes_) == 2 else float('nan')
    else:
        out['mle_macro_f1'] = float(f1_score(yt, pr, average='macro'))
    return out


def dcr(S, R, T, rows, seed):
    rng = np.random.RandomState(seed)
    if len(S) > rows: S = S[rng.choice(len(S), rows, replace=False)]
    m = min(len(R), len(T))
    R = R[rng.choice(len(R), m, replace=False)] if len(R) > m else R
    T = T[rng.choice(len(T), m, replace=False)] if len(T) > m else T
    def mind(A, B):            # L1 nearest-neighbour distance; scipy cdist (C) in row chunks: identical values to the
        from scipy.spatial.distance import cdist   # numpy-broadcast version used before 1 Oct 2026, ~30x faster
        out = np.empty(len(A))
        for i in range(0, len(A), 500):
            out[i:i + 500] = cdist(A[i:i + 500], B, metric="cityblock").min(1)
        return out
    dr, dt = mind(S, R), mind(S, T)
    return {'dcr_score': float((dr < dt).mean()), 'dcr_median_to_train': float(np.median(dr)), 'dcr_median_to_test': float(np.median(dt)), 'dcr_rows': int(len(S)),
            'dcr_ties': float((dr == dt).mean()), 'dcr_exact_copies': float((dr == 0).mean()),
            'dcr_score_ties_split': float(((dr < dt) + 0.5 * (dr == dt)).mean())}


G = {}
def init(dataname, dcr_rows, seed, target_transform='none', no_dcr=False, dcr_space='features', no_mle=False):
    info, num, cat, tgt = load_info(dataname)
    real = pd.read_csv(f'synthetic/{dataname}/real.csv'); test = pd.read_csv(f'synthetic/{dataname}/test.csv')
    real.columns = range(real.shape[1]); test.columns = range(test.shape[1])
    tf = Transform(real, num, cat, tgt, info['task_type'])
    G.update(tf=tf, task=info['task_type'], Xr=tf.X(real), Xt=tf.X(test), yt=tf.y(test), dcr_rows=dcr_rows, seed=seed,
             log=(target_transform == 'log1p' and info['task_type'] == 'regression'), no_dcr=no_dcr, dcr_space=dcr_space, no_mle=no_mle)
    if dcr_space == 'full' and not no_dcr:
        G.update(Xr_full=tf.X_full(real), Xt_full=tf.X_full(test))


def one(path):
    t0 = time.time(); syn = pd.read_csv(path); syn.columns = range(syn.shape[1]); tf = G['tf']
    Xs, ys = tf.X(syn), tf.y(syn)
    out = {'file': os.path.basename(path), 'n_syn': int(len(syn))}
    if G['no_mle']:
        pass
    elif G['log']:   # E7.1: regression target on log1p scale (synthetic negatives clipped to 0 first, count recorded)
        out['mle_log_clipped'] = int((ys < 0).sum())
        m = mle(Xs, np.log1p(np.clip(ys, 0, None)), G['Xt'], np.log1p(np.clip(G['yt'], 0, None)), G['task'], G['seed'])
        out.update({(k + '_log' if k in ('mle_rmse', 'mle_r2') else k): v for k, v in m.items()})
    else:
        out.update(mle(Xs, ys, G['Xt'], G['yt'], G['task'], G['seed']))
    if not G['no_dcr']:
        if G['dcr_space'] == 'full':
            out.update({k + '_full': v for k, v in dcr(tf.X_full(syn), G['Xr_full'], G['Xt_full'], G['dcr_rows'], G['seed']).items()})
        else:
            out.update(dcr(Xs, G['Xr'], G['Xt'], G['dcr_rows'], G['seed']))
    out['dcr_space'] = G['dcr_space']
    side = path[:-4] + '.json'
    if os.path.exists(side): out.update({k: v for k, v in json.load(open(side)).items() if k not in out})
    out['extra_seconds'] = round(time.time() - t0, 1); return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataname', required=True); ap.add_argument('--glob', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--workers', type=int, default=4); ap.add_argument('--dcr_rows', type=int, default=5000); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--target_transform', choices=('none', 'log1p'), default='none', help="regression only: log1p -> keys mle_rmse_log / mle_r2_log")
    ap.add_argument('--no_dcr', action='store_true', help='skip the DCR block (~5x faster)')
    ap.add_argument('--no_mle', action='store_true', help='skip the MLE block (DCR only)')
    ap.add_argument('--dcr_space', choices=('features', 'full'), default='features', help="'features' = predictor columns only (historical); 'full' = whole record incl. target (keys suffixed _full)")
    a = ap.parse_args()
    done = set()
    if os.path.exists(a.out):
        for line in open(a.out):
            if line.strip(): done.add(json.loads(line)['file'])
    files = sorted(f for f in glob.glob(a.glob) if os.path.basename(f) not in done)
    print(f'{len(done)} already evaluated, {len(files)} to do, workers={a.workers}')
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    t0 = time.time(); n = 0
    with open(a.out, 'a') as fh, Pool(a.workers, initializer=init, initargs=(a.dataname, a.dcr_rows, a.seed, a.target_transform, a.no_dcr, a.dcr_space, a.no_mle)) as pool:
        for res in pool.imap_unordered(one, files, chunksize=1):
            fh.write(json.dumps(res) + '\n'); fh.flush(); n += 1
            if n % 10 == 0 or n == len(files):
                print(f'  {n}/{len(files)} in {time.time()-t0:.0f}s, last={res["file"]} ' + ' '.join(f'{k}={v:.4f}' for k, v in res.items() if k.startswith(('mle_', 'dcr_score', 'dcr_ties', 'dcr_exact')) and isinstance(v, float)))
    print('OK eval_extra', a.out)


if __name__ == '__main__':
    main()
