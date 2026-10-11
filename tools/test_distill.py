"""Correctness tests for tools/distill.py (C3 of the confirmation plan). CPU, uses the smoke
checkpoint by default (tabsyn/ckpt/adultsmoke_s0). Every assert names the bug it catches.

  python tools/test_distill.py [--dataname adultsmoke_s0]
"""
import argparse, math, os, sys, copy
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.chdir(ROOT)
import torch
from tools.distill import (load_teacher, make_student, increasing_grid, teacher_ode_step, cd_loss,
                           sample_consistency, default_taus, SIGMA_DATA)
from tools.samplers import sample_ode, SIGMA_MIN, SIGMA_MAX

ap = argparse.ArgumentParser(); ap.add_argument('--dataname', default='adultsmoke_s0'); a = ap.parse_args()
dev = 'cpu'
D, train_z, mean, dim, *_ = load_teacher(a.dataname, dev)
data = (train_z - mean) / 2
f = make_student(D, dim, dev).eval()
grid = increasing_grid(D, 18, dev)
torch.manual_seed(0); x = data[torch.randint(0, len(data), (256,))] + 0.3 * torch.randn(256, dim)

# 1. boundary condition and teacher-equivalence at initialisation (parametrisation bugs)
with torch.no_grad():
    y = f(x, torch.tensor([SIGMA_MIN]))
    assert torch.equal(y, x), 'f(x, sigma_min) must equal x exactly (c_skip(sigma_min) = 1)'
    xs = SIGMA_MAX * torch.randn(256, dim)
    dmax = (f(xs, torch.tensor([SIGMA_MAX])) - D(xs, torch.tensor([SIGMA_MAX]))).abs().max().item()
    bound = SIGMA_DATA ** 2 / ((SIGMA_MAX - SIGMA_MIN) ** 2 + SIGMA_DATA ** 2) * (xs - D(xs, torch.tensor([SIGMA_MAX]))).abs().max().item()
    assert dmax <= bound + 1e-5, f'at init f(x,80) must equal teacher D(x,80) up to c_skip(80)|x-D|: {dmax} > {bound}'
    # per-sample sigma broadcasting: f with a (B,1) sigma vector equals f applied row-wise
    sig = grid[torch.randint(0, 18, (256,))].reshape(-1, 1)
    yb = f(x, sig); yr = torch.cat([f(x[i:i+1], sig[i:i+1]) for i in range(8)])
    assert torch.allclose(yb[:8], yr, atol=1e-5), 'per-sample sigma broadcasting is wrong'
print(f'1. boundary f(x,s_min)=x exact; init |f-D|(80) = {dmax:.2e} <= c_skip bound {bound:.2e}; per-sample sigma ok')

# 2. grid and teacher ODE step: Heun step from t_{n+1} to t_n reproduces the authors' sampler step
assert abs(grid[0].item() - max(SIGMA_MIN, D.sigma_min)) < 1e-8 and abs(grid[-1].item() - SIGMA_MAX) < 1e-4 and (grid[1:] > grid[:-1]).all(), 'grid must be increasing s_min..s_max'
with torch.no_grad():
    t_hi, t_lo = grid[-1].reshape(1, 1).expand(256, 1), grid[-2].reshape(1, 1).expand(256, 1)
    ours = teacher_ode_step(D, xs, t_hi, t_lo, 'heun')
    # authors' Heun step (tools/samplers.py logic with s_churn = 0), one step of a 18-step grid
    d = (xs - D(xs, t_hi)) / t_hi; xn = xs + (t_lo - t_hi) * d; d2 = (xn - D(xn, t_lo)) / t_lo
    ref = xs + (t_lo - t_hi) * (0.5 * d + 0.5 * d2)
    assert torch.allclose(ours, ref, atol=1e-6), 'teacher_ode_step (heun) differs from the sampler''s Heun step'
    eu = teacher_ode_step(D, xs, t_hi, t_lo, 'euler'); assert torch.allclose(eu, xn, atol=1e-6)
print('2. increasing Karras grid ok; Heun/Euler teacher steps match the authors'' sampler step')

# 3. loss: zero when student == target and teacher step is exact identity in the limit; finite; grad flows only to student
class A: ode = 'heun'; weight = 'uniform'; loss = 'huber'
tgt = copy.deepcopy(f)
g = torch.Generator().manual_seed(1)
l = cd_loss(f, tgt, D, data[:256], grid, A, dim, gen=g)
assert torch.isfinite(l) and l.item() > 0, 'loss must be finite and positive for a non-consistent init'
l.backward()
assert all(p.grad is not None for p in f.parameters()) and all(p.grad is None for p in tgt.parameters()) and all(p.grad is None for p in D.parameters()), 'gradients must flow only into the student'
g = torch.Generator().manual_seed(1); l2 = cd_loss(f, tgt, D, data[:256], grid, A, dim, gen=g)
assert torch.allclose(l, l2), 'held-out loss with a fixed generator must be reproducible'
A.loss = 'mse'; A.weight = 'inv_dt'; lw = cd_loss(f, tgt, D, data[:256], grid, A, dim); assert torch.isfinite(lw)
print(f'3. loss finite ({l.item():.4f}), reproducible with a generator, grads only into the student; mse/inv_dt ok')

# 4. multistep sampler: NFE accounting, determinism, default tau schedules decreasing and inside (s_min, s_max)
for k in (1, 2, 4):
    taus = default_taus(grid, k); assert len(taus) == k - 1 and all(SIGMA_MIN < t < SIGMA_MAX for t in taus) and taus == sorted(taus, reverse=True), taus
    torch.manual_seed(3); s1, n1 = sample_consistency(f, 64, dim, taus, dev); torch.manual_seed(3); s2, _ = sample_consistency(f, 64, dim, taus, dev)
    assert n1 == k and torch.equal(s1, s2) and torch.isfinite(s1).all()
print('4. multistep sampler: NFE = 1/2/4 as declared, deterministic, taus', [round(t, 3) for t in default_taus(grid, 4)])

# 5. the untrained 1-NFE student is the gate's 1-NFE teacher (same seed -> same latents)
with torch.no_grad():
    torch.manual_seed(9); s_cm, _ = sample_consistency(f, 64, dim, [], dev)
    torch.manual_seed(9); s_te, nfe = sample_ode(D, 64, dim, num_steps=1, s_churn=0.0, solver='heun', device=dev)
    # equal up to the c_skip(80) ~ 4e-5 mixing term: |f - D| <= c_skip(80) |80 z - D| ~ 1e-2 on latents of scale ~1
    assert nfe == 1 and (s_cm - s_te).abs().max().item() < 2e-2, 'untrained student at 1 NFE must reproduce the gate''s 1-step teacher up to c_skip(80)|x-D|'
print('5. untrained 1-NFE student == gate 1-NFE teacher up to the c_skip(80) term (max diff %.1e)' % (s_cm - s_te).abs().max().item())
# 6. consistency TRAINING mode: zero teacher calls (CD: 2 per batch with heun, 1 with euler); zero loss when t_lo == t_hi
calls = {'n': 0}
hk = D.register_forward_hook(lambda m, i, o: calls.__setitem__('n', calls['n'] + 1))
for mode, ode, want in (('ct', 'heun', 0), ('cd', 'heun', 2), ('cd', 'euler', 1)):
    class B: pass
    B.mode, B.ode, B.weight, B.loss = mode, ode, 'uniform', 'huber'
    calls['n'] = 0; cd_loss(f, tgt, D, data[:64], grid, B, dim, gen=torch.Generator().manual_seed(2))
    assert calls['n'] == want, f'mode {mode}/{ode}: {calls["n"]} teacher calls, expected {want}'
hk.remove()
class B: mode = 'ct'; ode = 'heun'; weight = 'uniform'; loss = 'huber'
flat = torch.full_like(grid, grid[9].item())
l0 = cd_loss(f, copy.deepcopy(f), D, data[:64], flat, B, dim, gen=torch.Generator().manual_seed(2))
assert l0.abs().item() < 1e-6, f'ct loss with t_lo == t_hi and student == target must be 0, got {l0.item()}'
print('6. ct mode: 0 teacher calls (cd heun 2, euler 1); zero loss at t_lo == t_hi')

# 7. resume: 3 steps, then continue to 6 from last.pt; RESUME line printed, best record carried, log has both segments
import subprocess, shutil, json as _json
tag = '_testresume'; out = f'tabsyn/ckpt/{a.dataname}/student_{tag}_s0'; shutil.rmtree(out, ignore_errors=True)
base = [sys.executable, 'tools/distill.py', '--dataname', a.dataname, '--seed', '0', '--gpu', '-1', '--tag', tag, '--batch', '64',
        '--holdout', '256', '--log_every', '3', '--ckpt_every', '3', '--mu', '0.999', '--N', '18']
r1 = subprocess.run(base + ['--steps', '3'], capture_output=True, text=True); assert r1.returncode == 0, r1.stderr[-2000:]
os.remove(f'{out}/DONE')
r2 = subprocess.run(base + ['--steps', '6'], capture_output=True, text=True); assert r2.returncode == 0, r2.stderr[-2000:]
assert 'RESUME at step 3' in r2.stdout, r2.stdout[-1000:]
ck = torch.load(f'{out}/last.pt', map_location='cpu'); assert ck['step'] == 6 and ck['best']['step'] >= 0
steps = [_json.loads(l)['step'] for l in open(f'{out}/log.jsonl') if l.strip()]
assert steps == [0, 3, 6], steps
shutil.rmtree(out)
print('7. resume from last.pt: RESUME at step 3, best record carried, log steps', steps)

# 8. trajectory reproducibility (1 Oct 2026): (a) cached ref_stats.pt must not change the training stream;
#    (b) a resumed run (3 + 3 steps) must equal an uninterrupted 6-step run — on GPU this exercises the CUDA RNG
#    save/restore added after the review (checkpoints written before it hold only the CPU state).
dev8 = 'cuda' if torch.cuda.is_available() else 'cpu'; gpu8 = '0' if dev8 == 'cuda' else '-1'
def _run(tag, steps):
    out = f'tabsyn/ckpt/{a.dataname}/student_{tag}_s0'
    cmd = [sys.executable, 'tools/distill.py', '--dataname', a.dataname, '--seed', '0', '--gpu', gpu8, '--tag', tag, '--batch', '64',
           '--holdout', '256', '--log_every', '3', '--ckpt_every', '3', '--mu', '0.999', '--N', '18', '--steps', str(steps)]
    r = subprocess.run(cmd, capture_output=True, text=True); assert r.returncode == 0, r.stderr[-2000:]
    return out, r.stdout
def _sd(out): return torch.load(f'{out}/student.pt', map_location='cpu')
def _maxdiff(s1, s2): return max((s1[k].float() - s2[k].float()).abs().max().item() for k in s1)
for t in ('_t8a', '_t8b', '_t8c'): shutil.rmtree(f'tabsyn/ckpt/{a.dataname}/student_{t}_s0', ignore_errors=True)
oa, _ = _run('_t8a', 3)                                            # fresh, computes ref_stats.pt
ob = f'tabsyn/ckpt/{a.dataname}/student__t8b_s0'; os.makedirs(ob); shutil.copy(f'{oa}/ref_stats.pt', f'{ob}/ref_stats.pt')
_run('_t8b', 3)                                                    # same 3 steps with the reference cached
d_cache = _maxdiff(_sd(oa), _sd(ob))
assert d_cache < 1e-6, f'cached ref_stats.pt changed the training trajectory (max |dw| = {d_cache:.2e})'
os.remove(f'{oa}/DONE'); _, so = _run('_t8a', 6)                   # resume 3 -> 6
assert 'RESUME at step 3' in so and (dev8 == 'cpu' or 'cuda_rng_restored=True' in so), so[-500:]
oc, _ = _run('_t8c', 6)                                            # uninterrupted 6
d_res = _maxdiff(_sd(oa), _sd(oc))
tol = 1e-6 if dev8 == 'cpu' else 1e-4                              # GPU kernels need not be bit-deterministic
assert d_res < tol, f'resumed (3+3) and uninterrupted (6) runs differ: max |dw| = {d_res:.2e} on {dev8}'
ck = torch.load(f'{oc}/last.pt', map_location='cpu'); assert (dev8 == 'cpu') or ('cuda_rng' in ck), 'GPU checkpoints must store the CUDA RNG state'
for t in ('_t8a', '_t8b', '_t8c'): shutil.rmtree(f'tabsyn/ckpt/{a.dataname}/student_{t}_s0', ignore_errors=True)
print(f'8. trajectory reproducibility on {dev8}: cached-reference max |dw| = {d_cache:.1e}; resumed-vs-uninterrupted max |dw| = {d_res:.1e}')
print('OK test_distill (8 tests)')
