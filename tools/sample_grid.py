"""Sample the trained latent EDM over a grid of (solver, num_steps, s_churn, sigma_max, rho,
sampling seed) and decode to CSV, loading the checkpoint once. Skip-if-exists.

  python tools/sample_grid.py --dataname adultv_s0 --grid gate [--gpu 0]
  python tools/sample_grid.py --dataname adultv_s0 --spec heun:50:1:80:7:0  (single config)

Output: synthetic/<dataname>/grid/<solver>_N<steps>_c<churn>_smax<smax>_rho<rho>_s<seed>.csv
plus a sidecar .json with the NFE actually used. num_samples = size of the train slot.
"""
import argparse, json, os, sys, time, itertools
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.chdir(ROOT)
import numpy as np, torch
from tabsyn.model import MLPDiffusion, Model
from tabsyn.latent_utils import get_input_generate, recover_data, split_num_cat_target
from tools.samplers import sample_ode

GATE_STEPS = [1, 2, 3, 4, 5, 6, 8, 10, 13, 20, 50]

def grid_specs(name):
    """(solver, steps, churn, smax, rho, seed)"""
    G = []
    if name in ('gate', 'shipped'):
        for N, c, s in itertools.product(GATE_STEPS, (0.0, 1.0), range(20)):
            G.append(('heun', N, c, 80, 7, s))
    if name in ('gate', 'euler'):
        for k, s in itertools.product((2, 4, 8, 16), range(20)):
            G.append(('euler', k, 0.0, 80, 7, s))
    if name in ('gate', 'sweep'):
        for smax, rho in itertools.product((80, 10, 3), (7, 3, 1)):
            if (smax, rho) == (80, 7): continue
            for s in range(5):
                for N in (2, 3, 4): G.append(('heun', N, 0.0, smax, rho, s))
                for k in (2, 4, 8): G.append(('euler', k, 0.0, smax, rho, s))
        for smax in (10, 3):
            for s in range(5): G.append(('heun', 1, 0.0, smax, 7, s))
    if name == 'fast_val':      # E1 (ablation plan): fair few-step teacher candidates, 5 val seeds each
        SMAX = (1, 2, 3, 5, 10, 20, 80)
        cands = set()
        for smax in SMAX: cands.add(('euler', 1, smax, 7))                       # k = 1: posterior mean
        for smax in SMAX:                                                          # k = 2: no interior grid point, rho irrelevant
            cands.add(('euler', 2, smax, 7)); cands.add(('dpmpp2m', 2, smax, 7))
        for k in (4, 8, 16):
            for smax, rho in itertools.product(SMAX, (1, 3, 7)):
                cands.add(('euler', k, smax, rho)); cands.add(('dpmpp2m', k, smax, rho))
                hN = (k + 1) // 2                                                  # heun NFE 2N-1 <= k
                cands.add(('heun', hN, smax, rho if hN > 2 else 7))               # N <= 2: rho irrelevant -> dedupe
        for (solver, N, smax, rho), s in itertools.product(sorted(cands), range(5)):
            G.append((solver, N, 0.0, smax, rho, s))
    if name == 'fast6':         # 1 Oct 2026 (review finding 13): measured F_6 candidates at exactly 5-6 NFE, same family as E1
        for smax, rho in itertools.product((1, 2, 3, 5, 10, 20, 80), (1, 3, 7)):
            for s in range(5):
                G.append(('euler', 6, 0.0, smax, rho, s)); G.append(('dpmpp2m', 6, 0.0, smax, rho, s)); G.append(('heun', 3, 0.0, smax, rho, s))
    if name == 'fast_lo':       # verdict v2 §6.1: is F_1's sigma_max = 1 a grid-edge artefact? 1-2 NFE below sigma_max 1
        for smax in (0.3, 0.5, 0.7):
            for s in range(5):
                G.append(('euler', 1, 0.0, smax, 7, s)); G.append(('euler', 2, 0.0, smax, 7, s)); G.append(('dpmpp2m', 2, 0.0, smax, 7, s))
    if name == 'smoke':
        G = [('heun', 50, 1.0, 80, 7, 0), ('heun', 1, 0.0, 80, 7, 0), ('heun', 2, 1.0, 80, 7, 0), ('euler', 4, 0.0, 80, 7, 0)]
    assert G, f'unknown grid {name}'
    return G

@torch.no_grad()
def decode_on_device(x, info, num_inverse, cat_inverse, device):
    """Same logic as tabsyn.latent_utils.split_num_cat_target, but runs the decoder on
    `device` instead of CPU (the authors' version builds a CPU tensor and decodes single-threaded,
    which dominates run time for ~800 configs). tools/test_decode.py asserts equality."""
    task_type = info['task_type']; target_col_idx = info['target_col_idx']
    pre_decoder = info['pre_decoder'].to(device); token_dim = info['token_dim']
    z = x.float().reshape(x.shape[0], -1, token_dim)
    x_hat_num, x_hat_cat = pre_decoder(z)
    syn_cat = torch.stack([pred.argmax(dim=-1) for pred in x_hat_cat]).t().cpu().numpy()
    syn_num = x_hat_num.cpu().numpy()
    syn_num = num_inverse(syn_num); syn_cat = cat_inverse(syn_cat)
    if task_type == 'regression':
        syn_target = syn_num[:, :len(target_col_idx)]; syn_num = syn_num[:, len(target_col_idx):]
    else:
        syn_target = syn_cat[:, :len(target_col_idx)]; syn_cat = syn_cat[:, len(target_col_idx):]
    return syn_num, syn_cat, syn_target

def _num(x):                     # 80 -> '80', 0.5 -> '0.5' (file names and sidecars)
    return int(x) if float(x) == int(float(x)) else float(x)

def fname(spec, tag=None):
    solver, N, c, smax, rho, s = spec
    return f'{solver}_N{N}_c{int(c)}_smax{_num(smax)}_rho{rho}' + (f'_{tag}' if tag else '') + f'_s{s}'

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataname', required=True)
    ap.add_argument('--grid', default=None)
    ap.add_argument('--spec', nargs='*', default=None, help='one or more solver:steps:churn:smax:rho:seed')
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--num_samples', type=int, default=None)
    ap.add_argument('--max_seeds', type=int, default=None, help='cap sampling seeds per config (smoke tests only)')
    ap.add_argument('--out_dir', default=None, help='default synthetic/<dataname>/grid')
    ap.add_argument('--tag', default=None, help='inserted before _s<seed> in file names (e.g. v = val grid, sel = final)')
    a = ap.parse_args()
    device = f'cuda:{a.gpu}' if (a.gpu != -1 and torch.cuda.is_available()) else 'cpu'
    specs = grid_specs(a.grid) if a.grid else []
    if a.max_seeds is not None:
        specs = [sp for sp in specs if sp[5] < a.max_seeds]
    for sp in (a.spec or []):
        f = sp.split(':'); specs.append((f[0], int(f[1]), float(f[2]), _num(f[3]), int(f[4]), int(f[5])))
    out_dir = a.out_dir or f'synthetic/{a.dataname}/grid'; os.makedirs(out_dir, exist_ok=True)
    todo = [sp for sp in specs if not os.path.exists(f'{out_dir}/{fname(sp, a.tag)}.json')]
    print(f'{len(specs)} configs, {len(todo)} to do, device={device}')
    if not todo: return
    args = argparse.Namespace(dataname=a.dataname, device=device)
    train_z, _, _, ckpt_path, info, num_inverse, cat_inverse = get_input_generate(args)
    in_dim = train_z.shape[1]; mean = train_z.mean(0)
    net = Model(denoise_fn=MLPDiffusion(in_dim, 1024), hid_dim=in_dim).to(device)
    net.load_state_dict(torch.load(f'{ckpt_path}/model.pt', map_location=device))
    net.eval()
    D = net.denoise_fn_D
    num_samples = a.num_samples or train_z.shape[0]
    idx_name_mapping = {int(k): v for k, v in info['idx_name_mapping'].items()}
    t0 = time.time()
    for i, sp in enumerate(todo):
        solver, N, c, smax, rho, s = sp
        torch.manual_seed(1000003 * s + 17)           # sampling seed: latents + churn noise
        x, nfe = sample_ode(D, num_samples, in_dim, num_steps=N, s_churn=c, solver=solver, sigma_max=smax, rho=rho, device=device)
        x = x * 2 + mean.to(device)
        assert torch.isfinite(x).all(), f'non-finite latents for {fname(sp, a.tag)}'
        syn_num, syn_cat, syn_target = decode_on_device(x, info, num_inverse, cat_inverse, device)
        df = recover_data(syn_num, syn_cat, syn_target, info)
        df.rename(columns=idx_name_mapping, inplace=True)
        base = f'{out_dir}/{fname(sp, a.tag)}'
        df.to_csv(base + '.csv', index=False, float_format='%.6g')
        json.dump({'solver': solver, 'steps': N, 's_churn': c, 'sigma_max': _num(smax), 'rho': rho, 'seed': s, 'nfe': nfe,
                   'num_samples': int(num_samples), 'dataname': a.dataname, 'tag': a.tag}, open(base + '.json', 'w'))
        if (i + 1) % 25 == 0 or i + 1 == len(todo):
            print(f'  {i+1}/{len(todo)} done, {time.time()-t0:.0f}s')
    print('OK sample_grid', a.dataname)

if __name__ == '__main__':
    main()
