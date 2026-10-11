"""Sample a distilled student at 1/2/4/... NFE with multistep consistency sampling, decode to CSV
in the same layout as tools/sample_grid.py so tools/eval_fast.py and gate_decide.py can score it.

  python tools/sample_student.py --dataname adultv_s0 --student student_cd_s0 --nfe 1 2 4 --seeds 20 [--gpu 0]
       [--taus 2=<t1> 4=<t1>,<t2>,<t3>]   (override the default Karras-rank tau schedule)
       [--weights ema|student|best] [--tag sel]
       [--tau_grid]   sample a preset grid of tau schedules per NFE (for selection on the val carve)

Output: synthetic/<dataname>/<student>/cm_N<nfe>_c0_smax80_rho7_<tag>_s<seed>.csv + .json
        (solver='cm', steps=nfe, nfe=nfe, taus recorded). Skip-if-exists. num_samples = |train'|
        (or --num_samples). Grid files carry tag g<i>; selected runs carry --tag.
"""
import argparse, json, os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.chdir(ROOT)
import torch
from tabsyn.latent_utils import recover_data
from tools.distill import load_teacher, make_student, increasing_grid, sample_consistency, default_taus
from tools.sample_grid import decode_on_device
from tools.samplers import SIGMA_MAX, RHO


def tau_grid(grid, k):
    """Candidate tau schedules per NFE for selection on the validation carve. Geometric ladders from a
    top level tau_1 down to tau_bot, plus the default Karras-rank schedule."""
    if k == 1: return [[]]
    tops = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
    if k == 2: return [[t] for t in tops] + [default_taus(grid, 2)]
    out = []
    for top in [2.0, 4.0, 8.0, 16.0]:
        for bot in [0.05, 0.2, 0.5]:
            r = (bot / top) ** (1.0 / (k - 2)) if k > 2 else 1.0
            out.append([float(top * r ** i) for i in range(k - 1)])
    return out + [default_taus(grid, k)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataname', required=True); ap.add_argument('--student', required=True)
    ap.add_argument('--nfe', type=int, nargs='+', default=[1, 2, 4]); ap.add_argument('--seeds', type=int, default=20)
    ap.add_argument('--gpu', type=int, default=0); ap.add_argument('--weights', choices=['ema', 'student', 'best'], default='best')
    ap.add_argument('--taus', nargs='*', default=[], help='k=t1,t2,... overrides for a given NFE k')
    ap.add_argument('--num_samples', type=int, default=None)
    ap.add_argument('--tag', default='sel'); ap.add_argument('--tau_grid', action='store_true')
    a = ap.parse_args()
    device = f'cuda:{a.gpu}' if (a.gpu != -1 and torch.cuda.is_available()) else 'cpu'
    sdir = f'tabsyn/ckpt/{a.dataname}/{a.student}'
    assert os.path.exists(f'{sdir}/DONE'), f'{sdir}/DONE missing: distillation not finished'
    cfg = json.load(open(f'{sdir}/DONE'))
    teacher_D, train_z, mean, dim, info, num_inverse, cat_inverse, _ = load_teacher(a.dataname, device)
    f = make_student(teacher_D, dim, device)
    f.load_state_dict(torch.load(f'{sdir}/{a.weights}.pt', map_location=device)); f.eval()
    grid = increasing_grid(teacher_D, cfg['N'], device)
    overrides = {int(s.split('=')[0]): [float(x) for x in s.split('=')[1].split(',')] for s in a.taus}
    out_dir = f'synthetic/{a.dataname}/{a.student}'; os.makedirs(out_dir, exist_ok=True)
    n = a.num_samples or train_z.shape[0]
    idx_name_mapping = {int(k): v for k, v in info['idx_name_mapping'].items()}
    t0 = time.time(); done = 0
    jobs = []   # (nfe, taus, tag)
    for k in a.nfe:
        if a.tau_grid:
            for i, taus in enumerate(tau_grid(grid, k)): jobs.append((k, taus, f'g{i}'))
        else:
            taus = overrides.get(k, default_taus(grid, k)); jobs.append((k, taus, a.tag))
    for k, taus, tag in jobs:
        assert len(taus) == k - 1, (k, taus)
        for s in range(a.seeds):
            base = f'{out_dir}/cm_N{k}_c0_smax{SIGMA_MAX}_rho{RHO}_{tag}_s{s}'
            if os.path.exists(base + '.json'): continue
            torch.manual_seed(1000003 * s + 17)
            x, nfe = sample_consistency(f, n, dim, taus, device)
            assert nfe == k and torch.isfinite(x).all()
            x = x * 2 + mean.to(device)
            syn_num, syn_cat, syn_target = decode_on_device(x, info, num_inverse, cat_inverse, device)
            df = recover_data(syn_num, syn_cat, syn_target, info); df.rename(columns=idx_name_mapping, inplace=True)
            df.to_csv(base + '.csv', index=False, float_format='%.6g')
            json.dump({'solver': 'cm', 'steps': k, 's_churn': 0.0, 'sigma_max': SIGMA_MAX, 'rho': RHO, 'seed': s, 'nfe': nfe,
                       'taus': taus, 'tag': tag, 'weights': a.weights, 'student': a.student, 'num_samples': int(n), 'dataname': a.dataname},
                      open(base + '.json', 'w'))
            done += 1
    print(f'OK sample_student {a.dataname} {a.student}: {done} new files in {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
