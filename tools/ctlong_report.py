"""Follow-up (11 Oct 2026): teacher-initialised consistency training (CT) continued from 20k to 40k updates,
seed-0 cell of each table (arm `ctlong`, slurm/study_lib.sh::stage_ctlong).

  python tools/ctlong_report.py > results/ctlong_report.txt          (also writes results/ctlong_cells.csv)

For each cell: shape error at 1 / 2 / 4 evaluations of the CD study student (20k), CT at 20k (`ct`) and CT at 40k
(`ctlong`), their differences, and the held-out self-consistency diagnostics of both CT runs from log.jsonl
(same definitions as tools/ablation_report.py). Descriptive only: 3 cells, no test, no reading rule.
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.chdir(ROOT)
import numpy as np, pandas as pd
from tools.ablation_report import student, convergence

rows = []
for d in ['adult', 'default', 'news']:
    A = f'{d}v_s0'
    S, C20, C40 = student(A, 'study', 0), student(A, 'ct', 0), student(A, 'ctlong', 0)
    if S is None or C20 is None or C40 is None:
        print(f'-- {A}: missing study, ct or ctlong results; skipped'); continue
    row = {'dataset': d, 'seed': 0}
    for j in (1, 2, 4):
        s, c20, c40 = S.loc[j, 'shape'], C20.loc[j, 'shape'], C40.loc[j, 'shape']
        row.update({f'cd_{j}': s, f'ct20k_{j}': c20, f'ct40k_{j}': c40,
                    f'ct20k_minus_cd_{j}': c20 - s, f'ct40k_minus_cd_{j}': c40 - s, f'ct40k_minus_ct20k_{j}': c40 - c20})
    for tag, name in (('ct', 'ct20k'), ('ctlong', 'ct40k')):
        row.update({f'{name}_{kk}': v for kk, v in convergence(A, tag, 0).items()})
    rows.append(row)

T = pd.DataFrame(rows)
if len(T):
    T.to_csv('results/ctlong_cells.csv', index=False)
    with pd.option_context('display.width', 250, 'display.max_columns', 60, 'display.float_format', lambda x: f'{x:.3f}'):
        print(T.to_string(index=False))
    for j in (1, 2, 4):
        for c in (f'ct20k_minus_cd_{j}', f'ct40k_minus_cd_{j}', f'ct40k_minus_ct20k_{j}'):
            v = T[c].to_numpy()
            print(f'{c:22s} mean {np.mean(v):+.3f}  range {np.min(v):+.3f} .. {np.max(v):+.3f}  (n = {len(v)})')
    print('written results/ctlong_cells.csv')
else:
    print('no complete cells')
