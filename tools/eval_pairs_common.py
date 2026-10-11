"""Common-metric pair fidelity and tail diagnostics (1 Oct 2026; answers review findings 15 and 22).

  python tools/eval_pairs_common.py --dataname adultv_s0 --real real --glob 'synthetic/adultv_s0/grid/*.csv' \
      --out_pairs results/adultv_s0_pairs_common.jsonl --out_cols results/adultv_s0_cols_common.jsonl --workers 5

Why. SDMetrics' Column Pair Trends scores num-num pairs by CorrelationSimilarity (|rho_real - rho_syn| / 2) and
any pair involving a categorical by ContingencySimilarity (total-variation distance of the joint table, numerics
binned), so the per-type pair excesses of E5 are measured on two metrics with different sensitivities. This tool
scores EVERY pair with ONE metric: numerics are discretised into --bins quantile bins whose edges come from the
real table only (applied identically to real and synthetic), categoricals are used as-is, and
  score = 1 - TVD(joint_real, joint_syn),  err = (1 - score) * 100   [= ContingencySimilarity's definition]
for all three pair types. For each pair it also reports `err_dep` = the same error after the synthetic joint has
been re-fitted to the REAL marginals by iterative proportional fitting (IPF, 50 sweeps): the error of a synthetic
joint whose 1-D marginals are forced to match the real ones. It is a diagnostic of how much of a pair error
survives marginal correction, NOT an additive decomposition (IPF minimises KL to the synthetic table, so err_dep
can occasionally exceed err); compare the two by pair type, not pair by pair.

Per column it reports the same common 1-D metric (TVD on the discretised marginal; `err_tv`) and, for numerical
columns, tail diagnostics: quantile errors |q_syn - q_real| / IQR_real at q in Q, and the synthetic mass below the
real 1st / above the real 99th percentile (`mass_lo1`, `mass_hi99`; 0.01 each for a perfect generator).
One JSON line per pair / per column, idempotent on `file`. Same metadata construction as eval_fast (reorder).

Fix of 11 Oct 2026 (found by the independent audit): synthetic values outside the real categories/bins were dropped
and the pair table renormalised, and the per-column error added that outside mass twice. Both now count it once, as
total-variation distance requires, and `syn_outside` records it. In every file scored for the report the outside
mass was exactly 0 (per-column err_tv equals the pair files' err_marg for all 13,050 column x file combinations), so
no reported number changes.
"""
import argparse, glob, json, os, sys, time, warnings
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.chdir(ROOT)
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from multiprocessing import Pool
from tools.eval_fast import reorder
from tools.eval_percol import column_table

Q = (0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999)


def edges_from_real(x, bins):
    e = np.unique(np.quantile(x[np.isfinite(x)], np.linspace(0, 1, bins + 1)[1:-1]))
    return e


def discretise(df, ct, edges):
    out = {}
    for i in df.columns:
        if ct[i]['col_type'] == 'numerical':
            out[i] = np.digitize(df[i].to_numpy(float), edges[i]).astype(np.int64)
        else:
            out[i] = df[i].astype(str).to_numpy()
    return out


def joint(a, b, ka, kb):
    """joint table over the real category sets ka, kb (index maps), as a share of ALL rows, plus the share of rows
    that fall outside them (a category or bin the real table does not have). Fixed 11 Oct 2026: the table used to be
    renormalised over the rows inside the real support, which hides that outside mass (it was 0 in every scored file)."""
    t = np.zeros((len(ka), len(kb)))
    ia = np.array([ka.get(v, -1) for v in a]); ib = np.array([kb.get(v, -1) for v in b])
    m = (ia >= 0) & (ib >= 0)
    np.add.at(t, (ia[m], ib[m]), 1.0)
    n = max(1, len(ia))
    return t / n, 1.0 - m.sum() / n


def ipf(P, row_t, col_t, sweeps=50):
    """rescale P so its margins equal row_t / col_t (zero-safe); returns the fitted table"""
    P = P.copy() + 1e-12
    for _ in range(sweeps):
        P *= (row_t / P.sum(1))[:, None]
        P *= (col_t / P.sum(0))[None, :]
    return P / P.sum()


def tvd(P, S):
    return 0.5 * float(np.abs(P - S).sum())


def one(job):
    syn_path, real_path, info, bins = job
    t0 = time.time()
    real = pd.read_csv(real_path); syn = pd.read_csv(syn_path)
    real.columns = range(len(real.columns)); syn.columns = range(len(syn.columns))
    ct = column_table(info, real)
    r, s, _ = reorder(real, syn, info)
    edges = {i: edges_from_real(r[i].to_numpy(float), bins) for i in r.columns if ct[i]['col_type'] == 'numerical'}
    dr, ds = discretise(r, ct, edges), discretise(s, ct, edges)
    keys = {i: {v: k for k, v in enumerate(sorted(set(dr[i].tolist())))} for i in r.columns}   # real categories/bins only
    side = syn_path[:-4] + '.json'; sc = json.load(open(side)) if os.path.exists(side) else {}
    keep = {k: sc[k] for k in ('solver', 'steps', 'nfe', 'seed', 'sigma_max', 'rho', 'student', 'tag') if k in sc}
    base = {'file': os.path.basename(syn_path), 'real': os.path.basename(real_path), 'bins': bins, **keep}
    cols, pairs = [], []
    marg = {}
    for i in r.columns:
        k = keys[i]
        pr = np.bincount([k[v] for v in dr[i]], minlength=len(k)) / len(dr[i])
        idx = [k.get(v, -1) for v in ds[i]]; idx = [j for j in idx if j >= 0]
        ps = np.bincount(idx, minlength=len(k)) / max(1, len(ds[i]))      # unseen synthetic values count as mass lost
        out_i = max(0.0, 1 - ps.sum())                                    # synthetic share outside the real bins
        marg[i] = (tvd(pr, ps) + 0.5 * out_i) * 100                       # same quantity as err_tv, reused in the pair records
        rec = {**base, **ct[i], 'err_tv': marg[i], 'syn_outside': out_i}   # 11 Oct 2026: was + out_i (double count)
        if ct[i]['col_type'] == 'numerical':
            xr, xs = r[i].to_numpy(float), s[i].to_numpy(float)
            iqr = float(np.quantile(xr, 0.75) - np.quantile(xr, 0.25)) or float(np.std(xr)) or 1.0
            qr, qs = np.quantile(xr, Q), np.quantile(xs, Q)
            rec.update({f'qerr_{q}': float(abs(a - b) / iqr) for q, a, b in zip(Q, qr, qs)})
            p1, p99 = np.quantile(xr, 0.01), np.quantile(xr, 0.99)
            rec.update({'mass_lo1': float((xs < p1).mean()), 'mass_hi99': float((xs > p99).mean()),
                        'max_ratio': float(xs.max() / xr.max()) if xr.max() else float('nan')})
        cols.append(rec)
    cidx = list(r.columns)
    for a in range(len(cidx)):
        for b in range(a + 1, len(cidx)):
            i, j = cidx[a], cidx[b]
            P, _ = joint(dr[i], dr[j], keys[i], keys[j]); S, out_s = joint(ds[i], ds[j], keys[i], keys[j])
            e = tvd(P, S) + 0.5 * out_s                                        # TVD including the mass outside the real support
            S_in = S / S.sum() if S.sum() > 0 else S                          # IPF works on the part inside the real support
            e_dep = tvd(P, ipf(S_in, P.sum(1), P.sum(0)))
            tps = sorted([ct[i]['col_type'][:3], ct[j]['col_type'][:3]], key=lambda t: (t != 'num', t))
            pairs.append({**base, 'col_a': ct[i]['column'], 'col_b': ct[j]['column'], 'pair_type': '-'.join(tps),
                          'involves_target': ct[i]['is_target'] or ct[j]['is_target'], 'err': e * 100, 'err_dep': e_dep * 100, 'syn_outside': out_s,
                          'err_marg_a': marg[i], 'err_marg_b': marg[j]})   # 11 Oct 2026: now include the outside mass, like err_tv
    return cols, pairs, round(time.time() - t0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataname', required=True); ap.add_argument('--real', default='real')
    ap.add_argument('--glob', required=True); ap.add_argument('--out_pairs', required=True); ap.add_argument('--out_cols', required=True)
    ap.add_argument('--workers', type=int, default=4); ap.add_argument('--bins', type=int, default=10)
    a = ap.parse_args()
    real_path = a.real if a.real.endswith('.csv') else f'synthetic/{a.dataname}/{a.real}.csv'
    info = json.load(open(f'data/{a.dataname}/info.json'))
    done = set()
    if os.path.exists(a.out_cols):
        for line in open(a.out_cols):
            if line.strip(): done.add(json.loads(line)['file'])
    files = sorted(f for f in glob.glob(a.glob) if os.path.basename(f) not in done)
    print(f'{len(done)} files already evaluated, {len(files)} to do, workers={a.workers}, bins={a.bins}')
    t0 = time.time(); n = 0
    with open(a.out_cols, 'a') as fc, open(a.out_pairs, 'a') as fp, Pool(a.workers) as pool:
        for cols, pairs, sec in pool.imap_unordered(one, [(f, real_path, info, a.bins) for f in files], chunksize=1):
            for p in pairs: fp.write(json.dumps(p) + '\n')
            fp.flush()
            for c in cols: fc.write(json.dumps(c) + '\n')
            fc.flush(); n += 1
            if n % 10 == 0 or n == len(files): print(f'  {n}/{len(files)} in {time.time()-t0:.0f}s (last {sec}s)')
    print('OK eval_pairs_common', a.out_pairs)


if __name__ == '__main__':
    main()
