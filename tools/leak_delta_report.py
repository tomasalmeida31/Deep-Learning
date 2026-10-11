"""Leak-delta table: the teacher trained with the authors' original setup (test split steering
the VAE, alias <d>L_s0) against our leak-free setup (validation carve, alias <d>v_s0), training
seed 0, same train'. Prints E99 shape, trend, C2ST, DCR and MLE for both, and the difference.
Usage: python tools/leak_delta_report.py > results/leak_delta_report.txt"""
import json, os
import numpy as np

DATASETS = ['adult', 'default', 'news']

def load(path):
    if not os.path.exists(path): return []
    return [json.loads(l) for l in open(path) if l.strip()]

def e99(records):
    return [r for r in records if r.get('solver') == 'heun' and int(r.get('steps', -1)) == 50 and int(r.get('sigma_max', -1)) == 80 and int(r.get('rho', -1)) == 7 and float(r.get('s_churn', 0)) == 0]

def mean(records, key):
    v = [r[key] for r in records if key in r and r[key] is not None and isinstance(r[key], (int, float)) and not (isinstance(r[key], float) and np.isnan(r[key]))]
    return (float(np.mean(v)), len(v)) if v else (float('nan'), 0)

for d in DATASETS:
    print(f'\n=== {d} ===')
    free, leak = f'{d}v_s0', f'{d}L_s0'
    row = {}
    for tag, A in [('free', free), ('leak', leak)]:
        gate = e99(load(f'results/{A}_gate.jsonl'))
        c2st = e99(load(f'results/{A}_teacher_c2st.jsonl'))
        extra = e99(load(f'results/{A}_teacher_extra.jsonl'))
        shape_m, shape_n = mean(gate, 'shape_err')
        trend_m, trend_n = mean(gate, 'trend_err')
        c2st_m, c2st_n = mean(c2st, 'c2st')
        dcr_m, dcr_n = mean(extra, 'dcr_score')
        row[tag] = dict(shape=shape_m, shape_n=shape_n, trend=trend_m, trend_n=trend_n, c2st=c2st_m, c2st_n=c2st_n, dcr=dcr_m, dcr_n=dcr_n)
        if d == 'news':
            rmse_m, rmse_n = mean(extra, 'mle_rmse'); r2_m, r2_n = mean(extra, 'mle_r2')
            row[tag]['mle_rmse'] = rmse_m; row[tag]['mle_r2'] = r2_m; row[tag]['mle_n'] = r2_n
            mlog = e99(load(f'results/{A}_mle_log.jsonl'))
            rmse_lm, _ = mean(mlog, 'mle_rmse_log'); r2_lm, r2_ln = mean(mlog, 'mle_r2_log')
            row[tag]['mle_rmse_log'] = rmse_lm; row[tag]['mle_r2_log'] = r2_lm; row[tag]['mle_log_n'] = r2_ln
        else:
            auc_m, auc_n = mean(extra, 'mle_roc_auc')
            row[tag]['mle_roc_auc'] = auc_m; row[tag]['mle_n'] = auc_n
        ref = load(f'results/{A}_ref_extra.jsonl')
        real_ref = next((r for r in ref if r.get('file') == 'real.csv'), {})
        row[tag]['ref_dcr'] = real_ref.get('dcr_score')
        if d == 'news':
            row[tag]['ref_r2'] = real_ref.get('mle_r2')
        else:
            row[tag]['ref_auc'] = real_ref.get('mle_roc_auc')
    for tag in ('free', 'leak'):
        r = row[tag]
        extra_keys = ' '.join(f'{k}={v:.4f}' for k, v in r.items() if k not in ('shape_n','trend_n','c2st_n','dcr_n','mle_n','mle_log_n') and isinstance(v, float))
        print(f'  {tag:5s} (n shape={r["shape_n"]} trend={r["trend_n"]} c2st={r["c2st_n"]} dcr={r["dcr_n"]} mle={r.get("mle_n")}): {extra_keys}')
    print('  delta (leak - free):')
    for k in row['free']:
        fv, lv = row['free'][k], row['leak'][k]
        if isinstance(fv, float) and isinstance(lv, float) and not (np.isnan(fv) or np.isnan(lv)):
            print(f'    {k}: {lv - fv:+.4f}  (free {fv:.4f} -> leak {lv:.4f})')
