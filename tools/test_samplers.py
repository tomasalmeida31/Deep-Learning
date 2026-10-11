"""Asserts tools/samplers.py reproduces the authors' sampler exactly, and that the
1-step and Euler definitions behave as documented. Runs on CPU with a randomly
initialised denoiser (no checkpoint needed)."""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.chdir(ROOT)
import torch
from tabsyn.model import MLPDiffusion, Model
from tabsyn.diffusion_utils import sample as authors_sample
from tools.samplers import sample_ode, karras_grid

torch.manual_seed(0)
dim = 60
net = Model(denoise_fn=MLPDiffusion(dim, 1024), hid_dim=dim).denoise_fn_D.eval()
# 1. bit-for-bit equality with the authors' sampler at their defaults
for N in (50, 10, 2):
    torch.manual_seed(123); a = authors_sample(net, 256, dim, num_steps=N, device='cpu')
    torch.manual_seed(123); b, nfe = sample_ode(net, 256, dim, num_steps=N, s_churn=1.0, solver='heun', device='cpu')
    assert torch.equal(a, b), f'mismatch vs authors at N={N}'
    assert nfe == 2 * N - 1
print('authors sampler reproduced at N=50,10,2 (NFE 99,19,3)')
# 2. authors' N=1 is NaN; ours is one denoiser pass from pure noise
torch.manual_seed(1); a = authors_sample(net, 8, dim, num_steps=1, device='cpu'); assert torch.isnan(a).all()
torch.manual_seed(1); z = torch.randn([8, dim]) * 80.0
torch.manual_seed(1); b, nfe = sample_ode(net, 8, dim, num_steps=1, s_churn=0.0, solver='heun', device='cpu')
with torch.no_grad(): ref = net(z, torch.tensor(80.0))
md = (b - ref).abs().max().item()
assert nfe == 1 and md < 1e-3, f'1-step must equal D(80 z, 80) up to fp rounding of x-(x-D); max diff {md}'
print(f'   max |x_1step - D(80z,80)| = {md:.2e} (floating-point rounding of the 80*(x-D)/80 update)')
torch.manual_seed(1); e, nfe_e = sample_ode(net, 8, dim, num_steps=1, s_churn=0.0, solver='euler', device='cpu')
assert nfe_e == 1 and torch.equal(b, e)
print('N=1: authors NaN confirmed; ours = D(sigma_max z, sigma_max), 1 NFE, heun==euler')
# 3. Euler NFE accounting and deterministic reproducibility
torch.manual_seed(5); e1, n1 = sample_ode(net, 16, dim, num_steps=4, s_churn=0.0, solver='euler', device='cpu')
torch.manual_seed(5); e2, _ = sample_ode(net, 16, dim, num_steps=4, s_churn=0.0, solver='euler', device='cpu')
assert n1 == 4 and torch.equal(e1, e2)
g = karras_grid(net, 4, device='cpu'); assert abs(g[0].item() - 80) < 1e-3 and g[-1].item() == 0 and abs(g[-2].item() - 0.002) < 1e-6, g
print('euler N=4 -> 4 NFE, deterministic; grid endpoints ok')
# 4. DPM-Solver++(2M): 1 step == Euler 1 step (up to the fp rounding of Euler's x-(x-D) update);
#    NFE accounting; determinism; churn refused
torch.manual_seed(1); d1, nd = sample_ode(net, 8, dim, num_steps=1, s_churn=0.0, solver='dpmpp2m', device='cpu')
assert nd == 1 and (d1 - e).abs().max().item() < 1e-3
torch.manual_seed(7); p1, n1 = sample_ode(net, 16, dim, num_steps=6, s_churn=0.0, solver='dpmpp2m', device='cpu')
torch.manual_seed(7); p2, _ = sample_ode(net, 16, dim, num_steps=6, s_churn=0.0, solver='dpmpp2m', device='cpu')
assert n1 == 6 and torch.equal(p1, p2) and torch.isfinite(p1).all()
try:
    sample_ode(net, 4, dim, num_steps=4, s_churn=1.0, solver='dpmpp2m', device='cpu'); raise SystemExit('dpmpp2m accepted churn')
except AssertionError: pass
print('dpmpp2m: 1 step == euler 1 step, NFE = steps, deterministic, churn refused')
# 5. Gaussian toy with an exact denoiser: data N(0, sd^2 I), D(x, s) = x sd^2/(s^2+sd^2);
#    exact PF-ODE: x(s) = x_T sqrt((s^2+sd^2)/(T^2+sd^2)), so x(0) = x_T sd/sqrt(T^2+sd^2).
class Toy(torch.nn.Module):
    sigma_min, sigma_max, sd = 0.0, float('inf'), 0.5
    def round_sigma(self, s): return torch.as_tensor(s)
    def forward(self, x, s): s = torch.as_tensor(s, dtype=torch.float64); return x * self.sd ** 2 / (s ** 2 + self.sd ** 2)
toy = Toy()
def err(solver, N, T=80.0):
    torch.manual_seed(3); x, _ = sample_ode(toy, 512, 4, num_steps=N, s_churn=0.0, solver=solver, sigma_max=T, device='cpu')
    torch.manual_seed(3); xT = torch.randn([512, 4]) * torch.tensor(T)
    exact = xT * toy.sd / (T ** 2 + toy.sd ** 2) ** 0.5
    return ((x - exact).norm() / exact.norm()).item()
e_eu8, e_dp8, e_dp16 = err('euler', 8), err('dpmpp2m', 8), err('dpmpp2m', 16)
print(f'   toy rel. error at 8 steps: euler {e_eu8:.2e}  dpmpp2m {e_dp8:.2e};  dpmpp2m 16 steps {e_dp16:.2e}')
assert e_dp8 < e_eu8 and e_dp16 < e_dp8
print('dpmpp2m beats euler on the exact Gaussian toy and converges with steps')
# 6. dpmpp2m equals a verbatim transcription of k-diffusion's sample_dpmpp_2m (lambda-space form)
def kdiff_dpmpp_2m(model, x, sigmas):
    sigma_fn = lambda t: t.neg().exp(); t_fn = lambda sg: sg.log().neg(); old = None
    for i in range(len(sigmas) - 1):
        den = model(x, sigmas[i]); t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i + 1]); h = t_next - t
        if old is None or sigmas[i + 1] == 0: x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * den
        else:
            r = (t - t_fn(sigmas[i - 1])) / h; dd = (1 + 1 / (2 * r)) * den - (1 / (2 * r)) * old
            x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * dd
        old = den
    return x
with torch.no_grad():
    for N, sm, rho in [(4, 80, 7), (8, 3, 3), (16, 10, 1)]:
        torch.manual_seed(9); ours, _ = sample_ode(net, 64, dim, num_steps=N, s_churn=0.0, solver='dpmpp2m', sigma_max=sm, rho=rho)
        torch.manual_seed(9); gg = karras_grid(net, N, sm, rho); ref = kdiff_dpmpp_2m(net, torch.randn(64, dim) * gg[0], gg)
        assert (ours - ref).abs().max().item() < 1e-4, (N, sm, rho)
print('dpmpp2m == k-diffusion sample_dpmpp_2m (max diff < 1e-4) at (4,80,7), (8,3,3), (16,10,1)')
print('OK test_samplers')
