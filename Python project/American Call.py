import numpy as np
import torch


def svdReg(x, y):
    pseudo = torch.pinverse(x)
    beta = torch.mv(torch.transpose(pseudo, 0, -1), y)
    return beta


def simPathsNumpy(spot, vol, expiry, strike, r, nSteps, nPaths):
    dt = expiry / nSteps
    paths = np.array([nPaths * [spot]] * nSteps)

    for i in range(1, nSteps):
        randNorm = np.random.randn(nPaths)
        paths[i] = paths[i - 1] * np.exp((r - 0.5 * vol * vol) * dt + np.sqrt(dt) * vol * randNorm)
    return paths


def simPathsTensor(spot, vol, expiry, r, nSteps, nPaths):
    dt = torch.tensor(expiry / nSteps)
    paths = torch.tensor(spot)
    randNorm = torch.randn(nPaths)
    paths = paths * torch.exp((r - 0.5 * vol * vol) * dt + torch.sqrt(dt) * vol * randNorm)
    paths = torch.cat([paths, paths * torch.exp((r - 0.5 * vol * vol) * dt + torch.sqrt(dt) * vol * randNorm)])
    paths = torch.reshape(paths, (2, nPaths))

    for i in range(2, nSteps):
        randNorm = torch.randn(nPaths)
        newPath = torch.reshape(
            paths[i - 1, :] * torch.exp((r - 0.5 * vol * vol) * dt + torch.sqrt(dt) * vol * randNorm), (1, nPaths))
        tmp = torch.cat((paths, newPath), 0)
        paths = torch.reshape(tmp, (i + 1, nPaths))
    return paths;


def AmCall(spot, vol, expiry, strike, r, nSteps, nPaths, nExcer, autograd=False, device="cpu"):
    r = torch.tensor(r, requires_grad=autograd, device=device)
    spot = torch.tensor(spot, requires_grad=autograd, device=device)
    expiry = torch.tensor(expiry, requires_grad=autograd, device=device)
    vol = torch.tensor(vol, requires_grad=autograd, device=device)
    strike = torch.tensor(strike, requires_grad=autograd, device=device)
    dt = torch.tensor(expiry / nSteps)

    # ---------------- Simulating paths ----------------#
    S0 = simPathsTensor(spot, vol, expiry, r, nSteps, nPaths)

    # ---------------- Defining variables for backpass ----------------#
    unflooredPayoff = torch.tensor(strike - S0[nSteps - 1])
    zeros = torch.tensor([0.] * nPaths)
    optVal = torch.max(unflooredPayoff, zeros)
    num = torch.FloatTensor(range(0, nPaths))

    backstep = int(np.floor((nSteps - 2) / nExcer))
    rest = (nSteps - 2) % nExcer

    for i in range(nSteps - 1 - backstep, rest, -backstep):
        # ---------------- Selecting options in the money ----------------#
        inMoney = torch.max(strike - S0[i], zeros)
        locs = inMoney == 0
        inMoneyRows = num[~locs]
        outMoneyRows = num[locs]
        m = torch.sum(locs == 0).item()

        if m == 0:
            optVal = optVal * torch.exp(-r * dt * backstep)
            continue;

        # ---------------- Variables used for regression ----------------#
        rones = torch.tensor([1] * m, dtype=torch.float)
        s1 = S0[i][~locs]
        s2 = (S0[i] * S0[i])[~locs]
        x = torch.reshape(torch.cat((rones, s1, s2), 0), (3, m))
        y = optVal[~locs] * torch.exp(-r * dt * backstep)
        # ---------------- Performing regression and calculating continuation value ----------------#
        beta = svdReg(x, y)
        conValIn = torch.mv(torch.transpose(x, 0, -1), beta)
        optvalOut = optVal[locs] * torch.exp(-r * dt * backstep)
        optValIn = torch.max((strike - S0[i][~locs]), conValIn)
        # ---------------- Restoring original indices by a permutation tensor  ----------------#
        if m == nPaths:
            optVal = optValIn;
        else:
            perm = torch.cat((inMoneyRows, outMoneyRows), 0).sort(0)[1]
            optVal = torch.cat((optValIn, optvalOut), 0)[perm];

    optVal = optVal * torch.exp(-r * dt * (rest + 1))
    price = torch.mean(optVal)

    # ---------------- Doing a backward sweep if autograd == True ----------------#

    if autograd == True:
        price.backward()
        output = (price, r.grad, spot.grad, strike.grad, vol.grad, expiry.grad)
    else:
        output = price

    return output;


x = AmCall(36.0, 0.2, 1.0, 40.0, 0.06, 52, 10000, 50, False)
print(x)
