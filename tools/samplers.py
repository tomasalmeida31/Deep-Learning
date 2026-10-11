"""Our sampler for the TabSyn latent EDM. The authors' sampler (tabsyn/diffusion_utils.py)
is left untouched; tools/test_samplers.py asserts that this module reproduces it
bit-for-bit at num_steps=50, s_churn=1 with the same RNG seed.

Differences from the authors' sample():
  * num_steps, s_churn, sigma_max, rho are arguments (authors: hard-coded 50, 1, 80, 7);
  * solver='heun' (authors') or 'euler' (first-order, exactly num_steps NFE) or 'dpmpp2m'
    (DPM-Solver++(2M), data prediction, multistep; k-diffusion's sample_dpmpp_2m in EDM's
    sigma parametrisation; exactly num_steps NFE; churn must be 0);
  * num_steps == 1: the authors' Karras grid divides by (num_steps - 1) = 0 and yields
    NaN. We define the 1-step grid as [sigma_max, 0], so a Heun or Euler run with
    num_steps=1 is one denoiser pass from pure noise: x0 = D(sigma_max * z, sigma_max),
    the posterior-mean estimate. This is OUR definition of the 1-NFE teacher.
NFE accounting: heun -> 2*num_steps - 1 ; euler, dpmpp2m -> num_steps. Returned alongside samples.
"""
import numpy as np
import torch

SIGMA_MIN = 0.002   # authors' module constants
SIGMA_MAX = 80
RHO = 7
S_MIN, S_MAX, S_NOISE = 0, float('inf'), 1

def karras_grid(net, num_steps, sigma_max=SIGMA_MAX, rho=RHO, sigma_min=SIGMA_MIN, device='cpu'):
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)
    if num_steps == 1:
        t = torch.tensor([sigma_max], dtype=torch.float32, device=device)
    else:
        idx = torch.arange(num_steps, dtype=torch.float32, device=device)
        t = (sigma_max ** (1 / rho) + idx / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    return torch.cat([net.round_sigma(t), torch.zeros_like(t[:1])])

@torch.no_grad()
def sample_ode(net, num_samples, dim, num_steps=50, s_churn=1.0, solver='heun',
               sigma_max=SIGMA_MAX, rho=RHO, device='cpu', return_path=False):
    assert solver in ('heun', 'euler', 'dpmpp2m')
    t_steps = karras_grid(net, num_steps, sigma_max, rho, device=device)
    latents = torch.randn([num_samples, dim], device=device)
    x_next = latents.to(torch.float32) * t_steps[0]
    nfe = 0
    path = [x_next.clone()] if return_path else None
    if solver == 'dpmpp2m':
        return _dpmpp2m(net, x_next, t_steps, num_steps, s_churn, return_path, path)
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x_cur = x_next
        gamma = min(s_churn / num_steps, np.sqrt(2) - 1) if S_MIN <= t_cur <= S_MAX else 0
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_NOISE * torch.randn_like(x_cur)
        denoised = net(x_hat, t_hat).to(torch.float32); nfe += 1
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur
        if solver == 'heun' and i < num_steps - 1:
            denoised = net(x_next, t_next).to(torch.float32); nfe += 1
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
        if return_path: path.append(x_next.clone())
    expected = 2 * num_steps - 1 if solver == 'heun' else num_steps
    assert nfe == expected, (nfe, expected)
    return (x_next, nfe, path) if return_path else (x_next, nfe)


def _dpmpp2m(net, x, t_steps, num_steps, s_churn, return_path, path):
    """DPM-Solver++(2M) on the grid t_0 > ... > t_{N-1} > t_N = 0 (lambda = -log sigma).
    Step i (s = t_i, t = t_{i+1}): D_i = D(x_i, s); first-order (i == 0 or t == 0):
    x_{i+1} = (t/s) x_i + (1 - t/s) D_i  (= Euler on dx/dsigma = (x - D)/sigma);
    otherwise r = h_{i-1}/h_i with h_i = log(s/t), D~ = (1 + 1/(2r)) D_i - 1/(2r) D_{i-1},
    x_{i+1} = (t/s) x_i + (1 - t/s) D~."""
    assert s_churn == 0, 'dpmpp2m is deterministic: s_churn must be 0'
    nfe = 0; old = None
    for i in range(num_steps):
        s, t = t_steps[i], t_steps[i + 1]
        D = net(x, s).to(torch.float32); nfe += 1
        ratio = t / s
        if old is None or t == 0:
            x = ratio * x + (1 - ratio) * D
        else:
            h = torch.log(s / t); h_last = torch.log(t_steps[i - 1] / s)
            r = h_last / h
            Dt = (1 + 1 / (2 * r)) * D - (1 / (2 * r)) * old
            x = ratio * x + (1 - ratio) * Dt
        old = D
        if return_path: path.append(x.clone())
    assert nfe == num_steps, (nfe, num_steps)
    return (x, nfe, path) if return_path else (x, nfe)
