import time

import numpy as np
import torch


def BsCall(spot, vol, mat, strike, r, nSteps, nPaths, autograd):
    device = torch.device("cpu")

    r = torch.tensor(r, requires_grad=autograd, device=device)
    spot = torch.tensor(spot, requires_grad=autograd, device=device)
    mat = torch.tensor(mat, requires_grad=autograd, device=device)
    vol = torch.tensor(vol, requires_grad=autograd, device=device)
    strike = torch.tensor(strike, requires_grad=autograd, device=device)
    dt = torch.tensor(mat / nSteps, device=device)
    S0 = spot

    for i in range(0, nSteps):
        rand_norm = torch.randn(nPaths)
        S0 = S0 * torch.exp((r - 0.5 * vol * vol) * dt + torch.sqrt(dt) * vol * rand_norm)
    unfloored_payoff = torch.tensor(S0 - strike)
    zeros = torch.tensor([0.0] * nPaths)
    result = torch.max(unfloored_payoff, zeros)

    price = torch.mean(torch.exp(-r * mat) * result)
    if autograd:
        price.backward()

    return 0;


times = np.empty((3, 6))

paths = np.array((10, 100, 1000, 10000, 100000, 1000000))

for i in range(0, 6):
    t = time.time()
    BsCall(100.0, 0.2, 1.0, 100.0, 0.03, 252, paths[i], True)
    times[0, i] = time.time() - t
    t = time.time()
    BsCall(100.0, 0.2, 1.0, 100.0, 0.03, 252, paths[i], False)
    times[1, i] = time.time() - t
    times[2, i] = times[0, i] / times[1, i]
print(times)
