import numpy as np
import scipy.stats as stats
import torch


def BsCall(spot, vol, expiry, strike, r, nSteps, nPaths):
    device = torch.device("cpu")

    d1 = 1 / (vol * np.sqrt(expiry)) * (np.log(spot / strike) + (r + vol * vol * 0.5) * expiry)
    d2 = d1 - vol * np.sqrt(expiry)
    r_grad_ana = strike * expiry * np.exp(-r * expiry) * stats.norm.cdf(d2)
    spot_grad_ana = stats.norm.cdf(d1)
    vol_grad_ana = spot * stats.norm.pdf(d1) * np.sqrt(expiry)
    mat_grad_ana = -spot * stats.norm.pdf(d1) * vol / (2 * np.sqrt(expiry)) - r * strike * np.exp(
        -r * expiry) * stats.norm.cdf(d2)
    price_ana = stats.norm.cdf(d1) * spot - stats.norm.cdf(d2) * strike * np.exp(-r * expiry)

    r = torch.tensor(r, requires_grad=True)
    spot = torch.tensor(spot, requires_grad=True)
    expiry = torch.tensor(expiry, requires_grad=True)
    vol = torch.tensor(vol, requires_grad=True)
    strike = torch.tensor(strike, requires_grad=True)
    dt = torch.tensor(expiry / nSteps, device=device)
    S0 = spot

    for i in range(0, nSteps):
        rand_norm = torch.randn(nPaths)
        S0 = S0 * torch.exp((r - 0.5 * vol * vol) * dt + torch.sqrt(dt) * vol * rand_norm)
    unfloored_payoff = torch.tensor(S0 - strike)
    zeros = torch.tensor([0.0] * nPaths)
    payoff = torch.max(unfloored_payoff, zeros)

    price = torch.mean(torch.exp(-r * expiry) * payoff)
    price.backward()

    return np.array((price.item(), spot.grad.item(), r.grad.item(), -expiry.grad.item(), vol.grad.item(), price_ana,
                     spot_grad_ana, r_grad_ana, mat_grad_ana, vol_grad_ana)).reshape(2, 5);


x = BsCall(100.0, 0.2, 1.0, 100.0, 0.03, 252, 100000)

print(x)
