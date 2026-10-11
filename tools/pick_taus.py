"""Choose the multistep tau schedule per NFE on the VALIDATION carve (never on test or train').

  python tools/pick_taus.py --dataname adultv_s0 --student student_cd_s0
      reads  results/<dataname>_<student>_valgrid.jsonl   (eval_fast --real val over the --tau_grid files)
      writes results/<dataname>_<student>_taus.json and prints the --taus arguments for sample_student.py

Selection: lowest mean shape error over the grid seeds; ties broken by trend error. The val carve
(~10% of train) has a finite-sample floor of ~1.2-1.4 pp shape, so it is used only for this
relative choice between schedules, never for the reported numbers (those use real.csv = train').
"""
import argparse, json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.chdir(ROOT)
import pandas as pd

ap = argparse.ArgumentParser(); ap.add_argument('--dataname', required=True); ap.add_argument('--student', required=True); a = ap.parse_args()
path = f'results/{a.dataname}_{a.student}_valgrid.jsonl'
df = pd.DataFrame([json.loads(l) for l in open(path) if l.strip()])
df['taus_key'] = df['taus'].apply(lambda t: json.dumps([round(float(x), 4) for x in t]))
g = df.groupby(['nfe', 'taus_key']).agg(n=('seed', 'count'), shape_m=('shape_err', 'mean'), shape_sd=('shape_err', 'std'), trend_m=('trend_err', 'mean')).reset_index()
chosen, args = {}, []
for k, sub in g.groupby('nfe'):
    sub = sub.sort_values(['shape_m', 'trend_m'])
    print(f'NFE {k}:')
    for _, r in sub.iterrows():
        print(f"   taus={r.taus_key:<32} shape {r.shape_m:6.2f}±{r.shape_sd:4.2f}  trend {r.trend_m:6.2f}  (n={r.n})")
    best = sub.iloc[0]; taus = json.loads(best.taus_key)
    chosen[int(k)] = {'taus': taus, 'val_shape': float(best.shape_m), 'val_trend': float(best.trend_m)}
    if k > 1: args.append(f"{int(k)}=" + ','.join(str(t) for t in taus))
json.dump(chosen, open(f'results/{a.dataname}_{a.student}_taus.json', 'w'), indent=1)
print('CHOSEN', json.dumps(chosen))
print('TAUS_ARGS', ' '.join(args))
