"""Seeded, skip-if-done wrapper around the authors' training entry points.

  python tools/train_seeded.py --stage vae       --dataname adultv_s0 --seed 0
  python tools/train_seeded.py --stage diffusion --dataname adultv_s0 --seed 0

Calls tabsyn.vae.main.main / tabsyn.main.main unchanged; writes
tabsyn/vae/ckpt/<dataname>/DONE or tabsyn/ckpt/<dataname>/DONE on success and
skips when that marker exists. --vae_epochs / --diff_epochs override the authors'
hard-coded 4000 / 10001 (used for smoke tests only; the override is recorded in DONE).
"""
import argparse, os, random, sys, time, json
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['vae', 'diffusion'], required=True)
    ap.add_argument('--dataname', required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--vae_epochs', type=int, default=None)
    ap.add_argument('--diff_epochs', type=int, default=None)
    a = ap.parse_args()
    device = f'cuda:{a.gpu}' if (a.gpu != -1 and torch.cuda.is_available()) else 'cpu'
    if a.stage == 'vae':
        ckdir = f'{ROOT}/tabsyn/vae/ckpt/{a.dataname}'
        needed = ['train_z.npy', 'decoder.pt', 'encoder.pt']
    else:
        ckdir = f'{ROOT}/tabsyn/ckpt/{a.dataname}'
        needed = ['model.pt']
    done = f'{ckdir}/DONE'
    if os.path.exists(done) and all(os.path.exists(f'{ckdir}/{f}') for f in needed):
        print(f'SKIP {a.stage} {a.dataname}: {done} exists'); return
    assert os.path.exists(f'data/{a.dataname}/info.json'), f'data/{a.dataname}/info.json missing'
    set_seed(a.seed)
    t0 = time.time()
    if a.stage == 'vae':
        import tabsyn.vae.main as M
        if a.vae_epochs is not None:   # smoke tests only
            src = open(M.__file__).read()
            assert 'num_epochs = 4000' in src, 'vae/main.py changed; refusing to patch epochs'
            code = compile(src.replace('num_epochs = 4000', f'num_epochs = {a.vae_epochs}'), M.__file__, 'exec')
            ns = {'__name__': 'tabsyn_vae_patched', '__file__': M.__file__}; exec(code, ns); fn = ns['main']
        else:
            fn = M.main
        args = argparse.Namespace(dataname=a.dataname, gpu=a.gpu, device=device, max_beta=1e-2, min_beta=1e-5, lambd=0.7)
    else:
        assert os.path.exists(f'{ROOT}/tabsyn/vae/ckpt/{a.dataname}/train_z.npy'), 'VAE stage not done'
        import tabsyn.main as M
        if a.diff_epochs is not None:
            src = open(M.__file__).read()
            assert 'num_epochs = 10000 + 1' in src, 'tabsyn/main.py changed; refusing to patch epochs'
            code = compile(src.replace('num_epochs = 10000 + 1', f'num_epochs = {a.diff_epochs}'), M.__file__, 'exec')
            ns = {'__name__': 'tabsyn_diff_patched', '__file__': M.__file__}; exec(code, ns); fn = ns['main']
        else:
            fn = M.main
        args = argparse.Namespace(dataname=a.dataname, gpu=a.gpu, device=device)
    fn(args)
    for f in needed:
        assert os.path.exists(f'{ckdir}/{f}'), f'stage finished but {ckdir}/{f} missing'
    json.dump({'stage': a.stage, 'dataname': a.dataname, 'seed': a.seed, 'device': device,
               'seconds': round(time.time() - t0, 1), 'vae_epochs': a.vae_epochs, 'diff_epochs': a.diff_epochs,
               'torch': torch.__version__}, open(done, 'w'))
    print(f'DONE {a.stage} {a.dataname} seed={a.seed} in {time.time()-t0:.0f}s')

if __name__ == '__main__':
    main()
