"""Wall-clock generation benchmark (follow-up, 11 Oct 2026): how long it takes to produce a full synthetic table
with the one-step student and with the teacher samplers it is compared against, on the same GPU, same table size.

  python tools/bench_speed.py --dataname adultv_s0 --seed 0 [--repeats 5] [--gpu 0]
  python tools/bench_speed.py --report            (summary over every results/*_bench_speed.json)

For one cell it times, for each generator, the same three phases the pipeline runs (sample_grid.py / sample_student.py):
  net     latent generation: the network evaluations from noise to a latent  (the part NFE counts)
  decode  VAE decoder + inverse transforms + assembling the table (pandas DataFrame); writing CSV is excluded
  total   net + decode
Generators: the study student S1 / S2 / S4 (best.pt, tau schedules from results/<A>_student_study_s<k>_taus.json);
the validation-selected teacher samplers F4 / F6 / F8 / F16 (results/<A>_fast_choice.json, _fast6_choice.json);
and E99 (the authors' 50-step Heun schedule, churn 0). Each generator makes |train'| rows in one batch, as in the
pipeline; one untimed warm-up run, then --repeats timed runs; medians are reported. Timings use torch.cuda.synchronize.
Training cost: 'seconds' from the study student's DONE record (wall-clock of the 20k-step run, including its periodic
diagnostics). Break-even = training seconds / (time per row of a baseline - time per row of S1), i.e. how many rows
must be generated before the student's training time is paid back; 'never' if S1 is not faster per row.
Writes results/<A>_bench_speed.json. The --report mode prints medians over cells and writes results/bench_speed_summary.csv.
"""
import argparse, glob, json, os, statistics, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.chdir(ROOT)


def timed(fn, device):
    import torch
    if device.startswith('cuda'): torch.cuda.synchronize(device)          # the measured GPU, not the current default one
    t = time.perf_counter(); out = fn()
    if device.startswith('cuda'): torch.cuda.synchronize(device)
    return out, time.perf_counter() - t


def bench(a):
    import torch
    from tabsyn.latent_utils import recover_data
    from tools.distill import load_teacher, make_student, sample_consistency
    from tools.samplers import sample_ode
    from tools.sample_grid import decode_on_device
    A, k = a.dataname, a.seed
    device = f'cuda:{a.gpu}' if (a.gpu != -1 and torch.cuda.is_available()) else 'cpu'
    teacher_D, train_z, mean, dim, info, num_inverse, cat_inverse, _ = load_teacher(A, device)
    mean = mean.to(device); n = a.num_samples or train_z.shape[0]
    sdir = f'tabsyn/ckpt/{A}/student_study_s{k}'
    f = make_student(teacher_D, dim, device); f.load_state_dict(torch.load(f'{sdir}/best.pt', map_location=device)); f.eval()
    taus = {int(j): v['taus'] for j, v in json.load(open(f'results/{A}_student_study_s{k}_taus.json')).items()}
    fast = json.load(open(f'results/{A}_fast_choice.json')); fast.update(json.load(open(f'results/{A}_fast6_choice.json')))
    gens = [(f'S{j}', 'student', {'taus': taus[j]}) for j in (1, 2, 4)]
    gens += [(f'F{j}', 'teacher', fast[str(j)]) for j in (4, 6, 8, 16)]
    gens += [('E99', 'teacher', {'solver': 'heun', 'steps': 50, 'sigma_max': 80, 'rho': 7})]

    def make(kind, cfg):
        if kind == 'student':
            return lambda: sample_consistency(f, n, dim, cfg['taus'], device)
        return lambda: sample_ode(teacher_D, n, dim, num_steps=int(cfg['steps']), s_churn=0.0, solver=cfg['solver'],
                                  sigma_max=cfg['sigma_max'], rho=int(cfg['rho']), device=device)

    def decode(x):
        x = x * 2 + mean
        syn_num, syn_cat, syn_target = decode_on_device(x, info, num_inverse, cat_inverse, device)
        return recover_data(syn_num, syn_cat, syn_target, info)

    out = {'dataname': A, 'seed': k, 'rows': int(n), 'device': device, 'repeats': a.repeats, 'torch': torch.__version__,
           'gpu_name': torch.cuda.get_device_name(device) if device.startswith('cuda') else 'cpu', 'generators': {}}
    with torch.no_grad():
        for label, kind, cfg in gens:
            torch.manual_seed(1000003 * 0 + 17)
            (x, nfe), _ = timed(make(kind, cfg), device); decode(x)                     # warm-up, untimed
            net_t, dec_t = [], []
            for r in range(a.repeats):
                torch.manual_seed(1000003 * (r + 1) + 17)
                (x, nfe), t1 = timed(make(kind, cfg), device)
                df, t2 = timed(lambda: decode(x), device)
                assert len(df) == n
                net_t.append(t1); dec_t.append(t2)
            tot = [p + q for p, q in zip(net_t, dec_t)]
            out['generators'][label] = {'nfe': int(nfe), 'config': cfg, 'net_s': statistics.median(net_t),
                                        'decode_s': statistics.median(dec_t), 'total_s': statistics.median(tot),
                                        'net_s_all': net_t, 'decode_s_all': dec_t}
            g = out['generators'][label]
            print(f"{A} {label:4s} nfe={g['nfe']:3d}  net {g['net_s']*1e3:8.1f} ms  decode {g['decode_s']*1e3:8.1f} ms  total {g['total_s']*1e3:8.1f} ms")
    train_s = json.load(open(f'{sdir}/DONE')).get('seconds')
    out['student_train_seconds'] = train_s
    s1 = out['generators']['S1']['total_s'] / n
    out['break_even_rows'] = {}
    for b in ('F4', 'F6', 'F8', 'F16', 'E99'):
        gain = out['generators'][b]['total_s'] / n - s1
        out['break_even_rows'][b] = (train_s / gain) if (train_s and gain > 0) else None
    json.dump(out, open(f'results/{A}_bench_speed.json', 'w'), indent=1)
    print('break-even rows (student training time / per-row saving vs baseline):',
          {b: (round(v) if v else 'never') for b, v in out['break_even_rows'].items()})
    print(f'written results/{A}_bench_speed.json')


def report():
    import pandas as pd
    rows = []
    for p in sorted(glob.glob('results/*_bench_speed.json')):
        r = json.load(open(p))
        for label, g in r['generators'].items():
            rows.append({'dataname': r['dataname'], 'gpu': r['gpu_name'], 'rows': r['rows'], 'generator': label, 'nfe': g['nfe'],
                         'net_ms': g['net_s'] * 1e3, 'decode_ms': g['decode_s'] * 1e3, 'total_ms': g['total_s'] * 1e3,
                         'break_even_rows': r['break_even_rows'].get(label) if label != 'S1' else None,
                         'student_train_s': r['student_train_seconds']})
    if not rows: print('no results/*_bench_speed.json files'); return
    T = pd.DataFrame(rows)
    s1 = T[T.generator == 'S1'].set_index('dataname')
    T['total_vs_S1'] = T.apply(lambda x: x.total_ms / s1.loc[x.dataname, 'total_ms'], axis=1)
    T['net_vs_S1'] = T.apply(lambda x: x.net_ms / s1.loc[x.dataname, 'net_ms'], axis=1)
    T.to_csv('results/bench_speed_cells.csv', index=False)
    S = T.groupby('generator').agg(nfe=('nfe', 'first'), cells=('dataname', 'count'), net_ms=('net_ms', 'median'),
                                   decode_ms=('decode_ms', 'median'), total_ms=('total_ms', 'median'),
                                   total_vs_S1=('total_vs_S1', 'median'), net_vs_S1=('net_vs_S1', 'median'),
                                   break_even_rows=('break_even_rows', 'median'))
    S = S.reindex([g for g in ['S1', 'S2', 'S4', 'F4', 'F6', 'F8', 'F16', 'E99'] if g in S.index])
    S.to_csv('results/bench_speed_summary.csv')
    print(f"GPU(s): {sorted(T.gpu.unique())}; rows per table = |train'| of each cell")
    with pd.option_context('display.width', 200, 'display.float_format', lambda x: f'{x:.2f}'):
        print('per cell:'); print(T.drop(columns=['gpu']).to_string(index=False))
        print('\nmedian over cells (times in ms; *_vs_S1 = how many times longer than the one-step student):'); print(S.to_string())
    print('written results/bench_speed_cells.csv, results/bench_speed_summary.csv')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataname'); ap.add_argument('--seed', type=int); ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--repeats', type=int, default=5); ap.add_argument('--num_samples', type=int, default=None)
    ap.add_argument('--report', action='store_true')
    a = ap.parse_args()
    if a.report: report()
    else:
        assert a.dataname and a.seed is not None, '--dataname and --seed are required (or use --report)'
        bench(a)
