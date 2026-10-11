"""Consistency distillation (Song et al., ICML 2023, Alg. 2) of a trained TabSyn latent EDM.

  python tools/distill.py --dataname adultv_s0 --seed 0 --gpu 0 [--steps 100000] [--N 18]
         [--mu 0.95] [--loss huber|mse] [--ode heun|euler] [--weight uniform|inv_dt] [--tag cd]

Teacher   : the authors' EDM checkpoint tabsyn/ckpt/<dataname>/model.pt, frozen.
Student   : same MLPDiffusion architecture, initialised from the teacher, wrapped as
              f_theta(x, s) = c_skip(s) * x + (1 - c_skip(s)) * D_theta(x, s),
              c_skip(s) = sd^2 / ((s - s_min)^2 + sd^2),   sd = 0.5, s_min = 0.002.
            This satisfies the boundary condition f(x, s_min) = x exactly and, because
            c_skip(80) ~ 4e-5, f(x, 80) equals the teacher's posterior mean D(x, 80) at
            initialisation, so the untrained 1-NFE student is the 1-NFE teacher of the gate.
Grid      : Karras sigma grid t_1 = s_min < ... < t_N = 80 (rho = 7), N fixed (default 18).
Step      : x0 ~ train' latents (standardised as in tabsyn/main.py: (z - mean) / 2),
            n ~ U{1..N-1}, x_{n+1} = x0 + t_{n+1} z; teacher ODE step t_{n+1} -> t_n
            (Heun by default = 2 teacher NFE, or Euler = 1) gives x_hat_n;
            loss = d( f_theta(x_{n+1}, t_{n+1}), f_{theta^-}(x_hat_n, t_n) ), theta^- = EMA(mu).
            d = pseudo-Huber sqrt(|.|^2 + c^2) - c with c = 0.00054 sqrt(dim) (Song & Dhariwal
            2024) or plain MSE. lambda(t_n) = 1 (uniform) or 1/(t_{n+1} - t_n) (inv_dt).
Sampling  : multistep consistency sampling (Alg. 1): x = f(80 z, 80); for tau in taus:
            x = f(x + sqrt(tau^2 - s_min^2) z', tau).  NFE = 1 + len(taus).
Outputs   : tabsyn/ckpt/<dataname>/student_<tag>_s<seed>/{student.pt, target.pt, ema.pt,
            log.jsonl, ref_stats.pt, DONE}; resumable from the last checkpoint (every 5000 steps).
Diagnostics every --log_every steps: held-out loss on 4096 train' latents never used for
training; distance of the EMA student's 1-NFE latent mean/covariance to the teacher's 99-NFE
reference (8192 samples, computed once); these are the C3 correctness signals.
Checkpoint selection (v2): `ema_selfcons` = consistency loss of the EMA model against ITSELF
(target = ema) on the held-out latents, i.e. how far the EMA model is from being constant along
the teacher's ODE. The EMA weights with the lowest ema_selfcons are saved as best.pt (pilot v1
showed the last checkpoint can be far worse than the best one). --lr_decay cosine decays the
learning rate to 0 over --steps. --mu 0 makes the target the current student (stop-grad), the
iCT recipe of Song & Dhariwal 2024.

Reproducibility (1 Oct 2026 revision). Minibatch indices and noise are drawn on the training
device, i.e. from the CUDA generator on a GPU. Checkpoints now store the CUDA generator state(s)
as well as the CPU one, and a resume restores both, so a resumed run follows the same trajectory
as an uninterrupted one (given deterministic kernels). Checkpoints written before this revision
hold only the CPU state: a resume from them (`cuda_rng_restored: false` in the log and in DONE)
continues from the checkpointed weights with a fresh CUDA stream — the `long` (40k) arms of
24-25 Sep were all resumed this way from the 20k study checkpoints. Separately, the teacher
reference statistics are drawn after torch.manual_seed(10_000 + seed); when ref_stats.pt is
cached the same draws are replayed so the training stream no longer depends on the cache.
"""
import argparse, copy, json, os, sys, time, math
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.chdir(ROOT)
import numpy as np, torch, torch.nn as nn
if not torch.cuda.is_available():
    # the authors' latent_utils calls torch.load without map_location; GPU-saved checkpoints then fail on the
    # CPU-only login node. Default map_location to CPU there so tests/reports can run without a GPU.
    _torch_load = torch.load
    def _cpu_load(*args, **kw):
        kw.setdefault('map_location', 'cpu'); return _torch_load(*args, **kw)
    torch.load = _cpu_load
from tabsyn.model import MLPDiffusion, Model
from tabsyn.latent_utils import get_input_generate
from tools.samplers import sample_ode, karras_grid, SIGMA_MIN, SIGMA_MAX, RHO

SIGMA_DATA = 0.5


class ConsistencyStudent(nn.Module):
    """f(x, s) = c_skip(s) x + (1 - c_skip(s)) D(x, s), D = EDM-preconditioned denoiser."""
    def __init__(self, in_dim, sigma_min=SIGMA_MIN, sigma_data=SIGMA_DATA):
        super().__init__()
        self.D = Model(denoise_fn=MLPDiffusion(in_dim, 1024), hid_dim=in_dim).denoise_fn_D
        self.sigma_min, self.sigma_data = float(sigma_min), float(sigma_data)

    def c_skip(self, sigma):
        return self.sigma_data ** 2 / ((sigma - self.sigma_min) ** 2 + self.sigma_data ** 2)

    def forward(self, x, sigma):
        sigma = torch.as_tensor(sigma, dtype=torch.float32, device=x.device).reshape(-1, 1)
        if sigma.shape[0] == 1: sigma = sigma.expand(x.shape[0], 1)
        cs = self.c_skip(sigma)
        return cs * x + (1 - cs) * self.D(x, sigma)


def load_teacher(dataname, device):
    args = argparse.Namespace(dataname=dataname, device=device)
    train_z, _, _, ckpt_path, info, num_inverse, cat_inverse = get_input_generate(args)
    in_dim = train_z.shape[1]; mean = train_z.mean(0)
    net = Model(denoise_fn=MLPDiffusion(in_dim, 1024), hid_dim=in_dim).to(device)
    net.load_state_dict(torch.load(f'{ckpt_path}/model.pt', map_location=device))
    net.eval()
    for p in net.parameters(): p.requires_grad_(False)
    return net.denoise_fn_D, train_z, mean, in_dim, info, num_inverse, cat_inverse, ckpt_path


def make_student(teacher_D, in_dim, device):
    s = ConsistencyStudent(in_dim).to(device)
    s.D.load_state_dict(teacher_D.state_dict())
    return s


def increasing_grid(teacher_D, N, device):
    g = karras_grid(teacher_D, N, device=device)      # decreasing: [t_max ... t_min, 0]
    return g[:-1].flip(0)                              # increasing: t_1 = s_min ... t_N = s_max


@torch.no_grad()
def teacher_ode_step(D, x, t_hi, t_lo, ode='heun'):
    """Deterministic PF-ODE step from t_hi to t_lo (per-sample t as (B,1) tensors), no churn."""
    d = (x - D(x, t_hi)) / t_hi
    x_lo = x + (t_lo - t_hi) * d
    if ode == 'heun':
        d2 = (x_lo - D(x_lo, t_lo)) / t_lo
        x_lo = x + (t_lo - t_hi) * 0.5 * (d + d2)
    return x_lo


def distance(a, b, kind, dim):
    if kind == 'mse':
        return ((a - b) ** 2).sum(1)
    c = 0.00054 * math.sqrt(dim)
    return torch.sqrt(((a - b) ** 2).sum(1) + c * c) - c


@torch.no_grad()
def sample_consistency(f, n, dim, taus, device, sigma_max=SIGMA_MAX, sigma_min=SIGMA_MIN):
    z = torch.randn(n, dim, device=device)
    x = f(sigma_max * z, torch.tensor([sigma_max], device=device))
    nfe = 1
    for tau in taus:
        x = f(x + math.sqrt(tau ** 2 - sigma_min ** 2) * torch.randn_like(x), torch.tensor([float(tau)], device=device)); nfe += 1
    return x, nfe


def default_taus(grid, nfe):
    """tau schedule for multistep sampling: Karras-grid points at evenly spaced ranks below t_N."""
    if nfe <= 1: return []
    N = len(grid)
    idx = [int(round(N - 1 - (N - 1) * k / nfe)) for k in range(1, nfe)]
    return [float(grid[i]) for i in idx]


@torch.no_grad()
def latent_stats(x):
    mu = x.mean(0); c = torch.cov(x.T)
    return mu, c


@torch.no_grad()
def stats_distance(mu, cov, mu_ref, cov_ref):
    return float((mu - mu_ref).norm()), float((cov - cov_ref).norm() / cov_ref.norm())


def cd_loss(student, target, teacher_D, x0, grid, a, dim, gen=None):
    """gen: optional torch.Generator making n and z reproducible (used for the held-out loss)."""
    B = x0.shape[0]; N = len(grid)
    n = torch.randint(0, N - 1, (B,), device=x0.device, generator=gen)   # index of t_n, n in 1..N-1
    t_lo, t_hi = grid[n].reshape(-1, 1), grid[n + 1].reshape(-1, 1)
    z = torch.randn(x0.shape, device=x0.device, generator=gen)
    x_hi = x0 + t_hi * z
    with torch.no_grad():
        if getattr(a, 'mode', 'cd') == 'ct':      # consistency TRAINING (Song et al. 2023 Alg. 3): same z, no teacher call
            x_hat = x0 + t_lo * z
        else:                                     # consistency DISTILLATION: one teacher PF-ODE step t_{n+1} -> t_n
            x_hat = teacher_ode_step(teacher_D, x_hi, t_hi, t_lo, a.ode)
        tgt = target(x_hat, t_lo)
    pred = student(x_hi, t_hi)
    w = torch.ones(B, device=x0.device) if a.weight == 'uniform' else (1.0 / (t_hi - t_lo)).reshape(-1)
    return (w * distance(pred, tgt, a.loss, dim)).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataname', required=True); ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--gpu', type=int, default=0); ap.add_argument('--tag', default='cd')
    ap.add_argument('--steps', type=int, default=100000); ap.add_argument('--batch', type=int, default=4096)
    ap.add_argument('--N', type=int, default=18); ap.add_argument('--mu', type=float, default=0.95)
    ap.add_argument('--ema', type=float, default=0.999, help='EMA of the student used for sampling')
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--loss', choices=['huber', 'mse'], default='huber')
    ap.add_argument('--ode', choices=['heun', 'euler'], default='heun')
    ap.add_argument('--weight', choices=['uniform', 'inv_dt'], default='uniform')
    ap.add_argument('--log_every', type=int, default=1000); ap.add_argument('--ckpt_every', type=int, default=5000)
    ap.add_argument('--holdout', type=int, default=4096)
    ap.add_argument('--lr_decay', choices=['none', 'cosine'], default='none')
    ap.add_argument('--mode', choices=['cd', 'ct'], default='cd', help='cd = distillation (teacher ODE step); ct = consistency training (no teacher)')
    a = ap.parse_args()
    device = f'cuda:{a.gpu}' if (a.gpu != -1 and torch.cuda.is_available()) else 'cpu'
    out = f'tabsyn/ckpt/{a.dataname}/student_{a.tag}_s{a.seed}'; os.makedirs(out, exist_ok=True)
    if os.path.exists(f'{out}/DONE'):
        print(f'SKIP distill {a.dataname} {a.tag} s{a.seed}: DONE exists'); return
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    teacher_D, train_z, mean, dim, *_ = load_teacher(a.dataname, device)
    data = ((train_z - mean) / 2).to(device)                       # exactly tabsyn/main.py
    perm = torch.randperm(len(data), generator=torch.Generator().manual_seed(1234))
    hold, train = data[perm[:a.holdout]], data[perm[a.holdout:]]
    grid = increasing_grid(teacher_D, a.N, device)
    student = make_student(teacher_D, dim, device)
    target = copy.deepcopy(student).eval(); ema = copy.deepcopy(student).eval()
    for m in (target, ema):
        for p in m.parameters(): p.requires_grad_(False)
    opt = torch.optim.Adam(student.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.steps) if a.lr_decay == 'cosine' else None
    best = {'ema_selfcons': float('inf'), 'step': -1}
    # teacher reference statistics (99 NFE, 8192 samples), computed once. The draw is seeded with
    # 10_000 + seed; the training stream continues from the state it leaves behind, so when the file is
    # cached the identical draws are replayed (sample_ode heun/50/churn-0 consumes 1 + 50 randn(8192, dim)).
    torch.manual_seed(10_000 + a.seed)
    if os.path.exists(f'{out}/ref_stats.pt'):
        ref = torch.load(f'{out}/ref_stats.pt', map_location=device)
        for _ in range(1 + 50): torch.randn([8192, dim], device=device)
    else:
        xr, _ = sample_ode(teacher_D, 8192, dim, num_steps=50, s_churn=0.0, solver='heun', device=device)
        mu_r, cov_r = latent_stats(xr); ref = {'mu': mu_r, 'cov': cov_r}; torch.save(ref, f'{out}/ref_stats.pt')
    step = 0; resume_rec = None
    if os.path.exists(f'{out}/last.pt'):
        ck = torch.load(f'{out}/last.pt', map_location=device)
        student.load_state_dict(ck['student']); target.load_state_dict(ck['target']); ema.load_state_dict(ck['ema'])
        opt.load_state_dict(ck['opt']); step = ck['step']; torch.set_rng_state(ck['rng'].cpu())   # map_location moves it to the GPU; CPU RNG state must be a CPU ByteTensor
        cuda_ok = False
        if 'cuda_rng' in ck and device.startswith('cuda'):
            torch.cuda.set_rng_state_all([t.cpu() for t in ck['cuda_rng']]); cuda_ok = True
        if sched is not None and 'sched' in ck: sched.load_state_dict(ck['sched'])
        best = ck.get('best', best)
        resume_rec = {'resumed_at_step': int(step), 'cuda_rng_restored': cuda_ok, 'device': device}
        print(f'RESUME at step {step} (cuda_rng_restored={cuda_ok})')
    log = open(f'{out}/log.jsonl', 'a')
    if resume_rec is not None:          # own file: log.jsonl holds only diagnostic rows (read by the report scripts)
        with open(f'{out}/resume.jsonl', 'a') as rf: rf.write(json.dumps({'time': time.time(), **resume_rec}) + '\n')

    @torch.no_grad()
    def diagnostics(tag_step):
        student.eval()
        gh = torch.Generator(device=device).manual_seed(4242)          # same n, z every time -> comparable across steps
        hl = float(np.mean([cd_loss(student, target, teacher_D, hold[i:i + 1024], grid, a, dim, gen=gh).item() for i in range(0, len(hold), 1024)]))
        g = torch.Generator(device=device).manual_seed(777)
        z = torch.randn(8192, dim, device=device, generator=g)
        x1 = ema(SIGMA_MAX * z, torch.tensor([SIGMA_MAX], device=device))
        mu1, cov1 = latent_stats(x1); dmu, dcov = stats_distance(mu1, cov1, ref['mu'], ref['cov'])
        gh = torch.Generator(device=device).manual_seed(4242)
        sc = float(np.mean([cd_loss(ema, ema, teacher_D, hold[i:i + 1024], grid, a, dim, gen=gh).item() for i in range(0, len(hold), 1024)]))
        rec = {'step': tag_step, 'holdout_loss': hl, 'ema_selfcons': sc, 'ema_1nfe_mean_dist': dmu, 'ema_1nfe_cov_reldist': dcov,
               'lr': opt.param_groups[0]['lr'], 'time': time.time() - t0}
        if sc < best['ema_selfcons']:
            best.update(ema_selfcons=sc, step=tag_step); torch.save(ema.state_dict(), f'{out}/best.pt'); rec['best'] = True
        log.write(json.dumps(rec) + '\n'); log.flush(); student.train(); return rec

    t0 = time.time()
    if step == 0:
        r = diagnostics(0); print('init', json.dumps({k: round(v, 5) if isinstance(v, float) else v for k, v in r.items()}))
    student.train()
    while step < a.steps:
        idx = torch.randint(0, len(train), (a.batch,), device=device)
        loss = cd_loss(student, target, teacher_D, train[idx], grid, a, dim)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step(); step += 1
        if sched is not None: sched.step()
        with torch.no_grad():
            for pt, ps, pe in zip(target.parameters(), student.parameters(), ema.parameters()):
                pt.mul_(a.mu).add_(ps, alpha=1 - a.mu); pe.mul_(a.ema).add_(ps, alpha=1 - a.ema)
        if step % a.log_every == 0:
            r = diagnostics(step); r['train_loss'] = float(loss.detach())
            print(json.dumps({k: round(v, 5) if isinstance(v, float) else v for k, v in r.items()}), flush=True)
        if step % a.ckpt_every == 0 or step == a.steps:
            torch.save({'student': student.state_dict(), 'target': target.state_dict(), 'ema': ema.state_dict(),
                        'opt': opt.state_dict(), 'step': step, 'rng': torch.get_rng_state(), 'best': best,
                        **({'cuda_rng': torch.cuda.get_rng_state_all()} if device.startswith('cuda') else {}),
                        **({'sched': sched.state_dict()} if sched is not None else {})}, f'{out}/last.pt')
    torch.save(ema.state_dict(), f'{out}/ema.pt'); torch.save(student.state_dict(), f'{out}/student.pt')
    if not os.path.exists(f'{out}/best.pt'): torch.save(ema.state_dict(), f'{out}/best.pt')
    json.dump({**vars(a), 'device': device, 'seconds': round(time.time() - t0, 1), 'grid': [float(t) for t in grid], 'best': best,
               'torch': torch.__version__, 'resume': resume_rec, 'in_dim': int(dim), 'n_train_latents': int(len(train)), 'n_holdout_latents': int(len(hold))},
              open(f'{out}/DONE', 'w'), indent=1)
    print(f'DONE distill {a.dataname} {a.tag} s{a.seed} steps={step} in {time.time()-t0:.0f}s -> {out}')


if __name__ == '__main__':
    main()
