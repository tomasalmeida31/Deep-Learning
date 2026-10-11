"""Asserts tools/sample_grid.decode_on_device equals the authors' split_num_cat_target
on a real checkpoint (uses the smoke alias by default) and times both on 29k rows."""
import os, sys, time, argparse
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.chdir(ROOT)
import numpy as np, torch
from tabsyn.latent_utils import get_input_generate, split_num_cat_target
from tools.sample_grid import decode_on_device
ap = argparse.ArgumentParser(); ap.add_argument('--dataname', default='adultsmoke_s0'); ap.add_argument('--gpu', type=int, default=-1); ap.add_argument('--rows', type=int, default=29000)
a = ap.parse_args()
device = f'cuda:{a.gpu}' if (a.gpu != -1 and torch.cuda.is_available()) else 'cpu'
args = argparse.Namespace(dataname=a.dataname, device=device)
train_z, _, _, _, info, num_inverse, cat_inverse = get_input_generate(args)
torch.manual_seed(0); x = (torch.randn(a.rows, train_z.shape[1]) * 0.5 + train_z.mean(0)).to(device)
t0 = time.time(); n1, c1, y1 = split_num_cat_target(x.cpu().numpy(), info, num_inverse, cat_inverse, device); ta = time.time() - t0
info['pre_decoder'] = info['pre_decoder'].to(device)
t0 = time.time(); n2, c2, y2 = decode_on_device(x, info, num_inverse, cat_inverse, device); tb = time.time() - t0
assert np.allclose(n1, n2, atol=1e-4) and (c1 == c2).all() and (y1 == y2).all(), 'decode mismatch'
print(f'decode equal on {a.rows} rows; authors CPU path {ta:.1f}s, ours on {device} {tb:.1f}s (torch threads={torch.get_num_threads()})')
print('OK test_decode')
