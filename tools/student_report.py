"""Student vs teacher at matched NFE, plus the pre-registered C3 pilot pass/fail.

  python tools/student_report.py --dataname adultv_s0 --student student_cd_s0

Reads results/<dataname>_gate_summary.csv (teacher, from gate_decide) and
results/<dataname>_<student>.jsonl (student, from eval_fast). Writes results/<dataname>_<student>_report.json.
Pilot rule (claude/tabsyn-confirmation-plan.md, C3): PASS if the 4-NFE student's mean shape error <= the
teacher's at 15 NFE (heun 8) AND the 1-NFE student's mean shape error <= 10 pp. This tests the machinery,
not the project's result: failing to reach E99 at 4 NFE is a reportable outcome, not a pilot failure.
"""
import argparse, json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.chdir(ROOT)
import pandas as pd

ap = argparse.ArgumentParser(); ap.add_argument('--dataname', required=True); ap.add_argument('--student', required=True); a = ap.parse_args()
T = pd.read_csv(f'results/{a.dataname}_gate_summary.csv')
S = pd.DataFrame([json.loads(l) for l in open(f'results/{a.dataname}_{a.student}.jsonl') if l.strip()])
s = S.groupby('nfe').agg(n=('seed', 'count'), shape_mean=('shape_err', 'mean'), shape_sd=('shape_err', 'std'),
                         trend_mean=('trend_err', 'mean'), trend_sd=('trend_err', 'std')).reset_index()
def teacher_at(nfe):
    r = T[(T.nfe == nfe) & (T.sigma_max == 80) & (T.rho == 7) & (T.s_churn == 0)]
    return r.iloc[0] if len(r) else None
e99 = teacher_at(99); t15 = teacher_at(15)
done = json.load(open(f'tabsyn/ckpt/{a.dataname}/{a.student}/DONE')) if os.path.exists(f'tabsyn/ckpt/{a.dataname}/{a.student}/DONE') else {}
taus_by_nfe = S.groupby('nfe')['taus'].first().to_dict() if 'taus' in S else {}
print(f'{a.dataname} / {a.student}   (teacher E99: shape {e99.shape_mean:.2f} trend {e99.trend_mean:.2f}; teacher@15 NFE: shape {t15.shape_mean:.2f} trend {t15.trend_mean:.2f})')
print(f"   recipe: steps={done.get('steps')} N={done.get('N')} mu={done.get('mu')} lr={done.get('lr')} lr_decay={done.get('lr_decay','none')} loss={done.get('loss')} ode={done.get('ode')}; best ckpt step {done.get('best',{}).get('step')} (ema_selfcons {done.get('best',{}).get('ema_selfcons')}); train {done.get('seconds')}s")
for k, t in taus_by_nfe.items(): print(f"   NFE {k} taus: {[round(float(x),3) for x in t]}")
print(f'{"NFE":>4} {"student shape":>15} {"student trend":>15} {"teacher shape@NFE":>18} {"teacher trend@NFE":>18}')
rows = []
for _, r in s.iterrows():
    t = teacher_at(int(r.nfe)); ts = f'{t.shape_mean:.2f}' if t is not None else '-'; tt = f'{t.trend_mean:.2f}' if t is not None else '-'
    print(f'{int(r.nfe):>4} {r.shape_mean:>9.2f}±{r.shape_sd:<5.2f} {r.trend_mean:>9.2f}±{r.trend_sd:<5.2f} {ts:>18} {tt:>18}')
    rows.append({'nfe': int(r.nfe), 'n': int(r.n), 'shape_mean': float(r.shape_mean), 'shape_sd': float(r.shape_sd), 'trend_mean': float(r.trend_mean), 'trend_sd': float(r.trend_sd),
                 'teacher_shape_at_nfe': None if t is None else float(t.shape_mean), 'teacher_trend_at_nfe': None if t is None else float(t.trend_mean)})
by = {r['nfe']: r for r in rows}
pass4 = 4 in by and by[4]['shape_mean'] <= t15.shape_mean
pass1 = 1 in by and by[1]['shape_mean'] <= 10.0
verdict = 'PASS' if (pass4 and pass1) else 'FAIL'
print(f'\nC3 pilot: 4-NFE shape {by.get(4, {}).get("shape_mean", float("nan")):.2f} <= teacher@15 {t15.shape_mean:.2f}: {pass4};  1-NFE shape {by.get(1, {}).get("shape_mean", float("nan")):.2f} <= 10: {pass1}  ->  {verdict}')
matches = [r['nfe'] for r in rows if r['shape_mean'] <= e99.shape_mean + 0.25]
print('student NFE within 0.25 pp shape of E99:', matches or 'none')
json.dump({'dataname': a.dataname, 'student': a.student, 'rows': rows, 'E99_shape': float(e99.shape_mean), 'E99_trend': float(e99.trend_mean),
           'teacher15_shape': float(t15.shape_mean), 'pilot_pass4': bool(pass4), 'pilot_pass1': bool(pass1), 'verdict': verdict, 'matches_E99_at_nfe': matches},
          open(f'results/{a.dataname}_{a.student}_report.json', 'w'), indent=1)
print('written', f'results/{a.dataname}_{a.student}_report.json')
