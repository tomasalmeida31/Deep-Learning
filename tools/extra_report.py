"""Aggregate C4 metrics (MLE, DCR) per configuration: references, teacher at each NFE, student at each NFE.

  python tools/extra_report.py --dataname adultv_s0 --student student_cdA_s0
Reads results/<d>_ref_extra.jsonl (val.csv and real.csv), results/<d>_teacher_extra.jsonl,
results/<d>_<student>_extra.jsonl (if present). Writes results/<d>_<student>_extra_report.json.
"""
import argparse, json, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.chdir(ROOT)
import numpy as np, pandas as pd

ap = argparse.ArgumentParser(); ap.add_argument('--dataname', required=True); ap.add_argument('--student', default=None); a = ap.parse_args()
def load(p): return pd.DataFrame([json.loads(l) for l in open(p) if l.strip()]) if os.path.exists(p) else pd.DataFrame()
ref, tea, stu = load(f'results/{a.dataname}_ref_extra.jsonl'), load(f'results/{a.dataname}_teacher_extra.jsonl'), (load(f'results/{a.dataname}_{a.student}_extra.jsonl') if a.student else pd.DataFrame())
metrics = [c for c in ['mle_roc_auc', 'mle_accuracy', 'mle_macro_f1', 'mle_rmse', 'mle_r2', 'dcr_score'] if any(c in d.columns for d in (ref, tea, stu) if len(d))]
rows = []
def add(label, nfe, d):
    r = {'group': label, 'nfe': nfe, 'n': int(len(d))}
    for m in metrics:
        if m in d: r[m] = float(d[m].mean()); r[m + '_sd'] = float(d[m].std(ddof=1)) if len(d) > 1 else float('nan')
    rows.append(r)
for _, r in ref.iterrows(): add(f"reference {r['file']}", None, ref[ref.file == r['file']])
if len(tea):
    for (sol, st, nfe), d in tea.groupby(['solver', 'steps', 'nfe']): add(f'teacher {sol} {int(st)}', int(nfe), d)
if len(stu):
    for nfe, d in stu.groupby('nfe'): add(f'student {a.student}', int(nfe), d)
out = pd.DataFrame(rows).sort_values(['group', 'nfe'], key=lambda s: s.map(lambda v: (0 if str(v).startswith('reference') else 1 if str(v).startswith('teacher') else 2) if isinstance(v, str) else (v if v == v else -1)))
cols = ['group', 'nfe', 'n'] + [c for m in metrics for c in (m, m + '_sd')]
with pd.option_context('display.width', 200, 'display.float_format', lambda x: f'{x:.4f}'):
    print(out[cols].to_string(index=False))
name = f'results/{a.dataname}_{a.student or "teacher"}_extra_report.json'
json.dump(rows, open(name, 'w'), indent=1); print('written', name)
