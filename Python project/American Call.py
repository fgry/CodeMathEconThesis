import torch
import numpy as np


def svdReg(x, y):
    pseudo = torch.pinverse(x)
    beta = torch.mv(torch.transpose(pseudo, 0, -1), y)
    return beta;


def simPathsNumpy(spot, vol, expiry, strike, r, nSteps, nPaths):
    dt = expiry / nSteps
    paths = np.array([nPaths * [spot]] * nSteps)

    for i in range(1, nSteps):
        randNorm = np.random.randn(nPaths)
        paths[i] = paths[i - 1] * np.exp((r - 0.5 * vol * vol) * dt + np.sqrt(dt) * vol * randNorm)
    return paths;


def simPathsTensor(spot, vol, expiry, strike, r, nSteps, nPaths):
    dt = torch.tensor(expiry / nSteps)
    paths = torch.tensor([nPaths * [spot]] * nSteps)

    for i in range(1, nSteps):
        randNorm = torch.randn(nPaths)
        paths[i] = paths[i - 1] * torch.exp((r - 0.5 * vol * vol) * dt + torch.sqrt(dt) * vol * randNorm)
    return paths;


def AmCall(spot, vol, expiry, strike, r, nSteps, nPaths, nExcer, device="cpu"):
    r = torch.tensor(r, requires_grad=True, device=device)
    spot = torch.tensor(spot, requires_grad=True, device=device)
    expiry = torch.tensor(expiry, requires_grad=True, device=device)
    vol = torch.tensor(vol, requires_grad=True, device=device)
    strike = torch.tensor(strike, requires_grad=True, device=device)
    dt = torch.tensor(expiry / nSteps)

    # ---------------- Simulating paths ----------------#
    S0 = simPathsTensor(spot, vol, expiry, strike, r, nSteps, nPaths)

    # ---------------- Defining variables for backpass ----------------#
    unflooredPayoff = torch.tensor(strike - S0[nSteps - 1])
    zeros = torch.tensor([0.] * nPaths)
    optVal = torch.max(unflooredPayoff, zeros)
    conVal = torch.tensor([0.] * nPaths)

    backstep = int(np.floor((nSteps - 2) / nExcer))
    rest = (nSteps - 2) % nExcer

    for i in range(nSteps - 2, rest - 1, -backstep):
        # ---------------- Selecting options in the money ----------------#
        inMoney = torch.max(strike - S0[i], zeros)
        locs = inMoney == 0
        m = torch.sum(locs == 0).item()
        # ---------------- Variables used for regression ----------------#
        rones = torch.tensor([1] * m, dtype=torch.float)
        s1 = S0[i][~locs]
        s2 = (S0[i] * S0[i])[~locs]
        x = torch.reshape(torch.cat((rones, s1, s2), 0), (3, m))
        y = optVal[~locs] * torch.exp(-r * dt) * backstep
        # ---------------- Performing regression and calculating continuation value ----------------#
        beta = svdReg(x, y)
        conVal[~locs] = torch.mv(torch.transpose(x, 0, -1), beta)
        conVal[locs] = optVal[locs] * torch.exp(-r * dt) * backstep
        optVal = torch.max((strike - S0[i]), conVal)

    optVal = optVal * torch.exp(-r * dt) * rest
    price = torch.mean(optVal)
    return price;


AmCall(36., 0.2, 1.0, 40.0, 0.06, 254, 1000, 50)
