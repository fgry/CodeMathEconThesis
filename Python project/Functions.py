import operator

import numpy as np
import scipy.stats as stats
import torch


def wlag1(x):
    return torch.exp(-x * 0.5)


def wlag2(x):
    return torch.exp(-x * 0.5) * (1 - x)


def wlag3(x):
    return torch.exp(-x * 0.5) * (1 - 2 * x + 0.5 * x ** 2)


def lag1(x):
    return -x + 1


def lag2(x):
    return 0.5 * (x ** 2 - 4 * x + 2)


def lag3(x):
    return (1 / 6) * (-x ** 3 + 9 * x ** 2 - 18 * x + 6)


def svdReg(x, y):
    pseudo = torch.pinverse(x)
    beta = torch.mv(pseudo, y)
    return beta


def simPathsNumpy(spot, vol, expiry, r, nSteps, nPaths):
    dt = expiry / nSteps
    paths = np.array([nPaths * [spot]] * nSteps)

    for i in range(1, nSteps):
        randNorm = np.random.randn(nPaths)
        paths[i] = paths[i - 1] * np.exp((r - 0.5 * vol * vol) * dt + np.sqrt(dt) * vol * randNorm)
    return paths


def simPathsTensor(spot, vol, expiry, r, n_steps, n_paths):
    dt = (expiry / n_steps)
    paths = torch.zeros((n_steps, n_paths))
    tmp = torch.zeros([n_steps, n_paths])
    tmp[0] = spot
    paths = paths + tmp
    tmp[0] = 0

    for i in range(1, n_steps):
        rand_norm = torch.randn(n_paths)
        tmp[i] = paths[i - 1] * torch.exp((r - 0.5 * vol * vol) * dt + torch.sqrt(dt) * vol * rand_norm)
        paths = paths + tmp
        tmp[i] = 0
    return paths;


def grid(vals, expiry, max_spot, min_spot):
    x_delta = torch.tensor((expiry) / (vals.size()[1]))
    end_x = float(expiry + x_delta)
    end_y = float(max_spot)

    x = np.linspace(0, end_x, vals.size()[1])
    y = np.linspace(min_spot, end_y, vals.size()[0])

    return x, y


def localVol(vals, time, space, x, y):
    n_paths = space.size()[0]
    dt = torch.tensor([time] * n_paths, dtype=torch.float).reshape(n_paths, 1)
    paths = torch.tensor(space, dtype=torch.float).reshape(n_paths, 1)
    z = torch.tensor(torch.cat([dt, paths], -1), dtype=torch.double)

    geqX = x.lt(torch.reshape(z[:, 0], (n_paths, 1)))
    geqY = y.lt(torch.reshape(z[:, 1], (n_paths, 1)))

    supx = x[0, geqX.sum(1)]
    supy = y[0, geqY.sum(1)]
    infx = x[0, geqX.sum(1) - 1];
    infy = y[0, geqY.sum(1) - 1];

    scal = torch.div(torch.tensor(1.0), torch.tensor((supx - infx) * (supy - infy), dtype=torch.float))[0]
    xvec = torch.tensor(torch.cat(((supx - z[:, 0]).reshape(n_paths, 1), (z[:, 0] - infx).reshape(n_paths, 1)), -1),
                        dtype=torch.float)
    yvec = torch.tensor(torch.cat(((supy - z[:, 1]).reshape(n_paths, 1), (z[:, 1] - infy).reshape(n_paths, 1)), -1),
                        dtype=torch.float)
    x2 = x.eq(torch.reshape(supx, (n_paths, -1))).nonzero()[:, 1]
    x1 = x2 - 1
    y1 = y.eq(torch.reshape(infy, (n_paths, -1))).nonzero()[:, 1]
    y2 = y1 + 1
    Q = torch.cat((vals[y1, x1].reshape(n_paths, 1, 1), vals[y1, x2].reshape(n_paths, 1, 1),
                   vals[y2, x1].reshape(n_paths, 1, 1), vals[y2, x2].reshape(n_paths, 1, 1)), 1).reshape(n_paths, 2, 2)

    yvec = yvec.unsqueeze(2)
    xvec = xvec.unsqueeze(1)

    tmp = Q.matmul(yvec)

    vol = (scal * xvec.matmul(tmp)).squeeze(2).squeeze(1)
    return vol;


def simPathsTensorLocal(spot, vols, expiry, r, n_steps, n_paths, max_spot, min_spot):
    end_x = float(expiry + 0.1)
    end_y = float(max_spot)

    x_grid = torch.tensor([np.linspace(0, end_x, vols.size()[1])] * n_paths)
    y_grid = torch.tensor([np.linspace(min_spot, end_y, vols.size()[0])] * n_paths)

    dt = (expiry / n_steps)
    paths = torch.zeros((n_steps, n_paths))
    tmp = torch.zeros([n_steps, n_paths])
    tmp[0] = spot
    paths = paths + tmp
    tmp[0] = 0

    for i in range(1, n_steps):
        rand_norm = torch.randn(n_paths)
        time = dt * i
        vol = localVol(vols, time, paths[i - 1, :], x_grid, y_grid)
        tmp[i] = paths[i - 1] * torch.exp((r - 0.5 * vol * vol) * dt + torch.sqrt(dt) * vol * rand_norm)
        paths = paths + tmp
        tmp[i] = 0
    return paths;


def localVolCeil(vals, time, space, x, y):
    n_paths = space.size()[0]
    dt = time.repeat(n_paths).reshape(n_paths, 1)
    paths = space.reshape(n_paths, 1)
    z = torch.cat([dt, paths], -1).double()

    geqX = x.lt(torch.reshape(z[:, 0], (n_paths, 1)))
    geqY = y.lt(torch.reshape(z[:, 1], (n_paths, 1)))
    supx = x[0, geqX.sum(1)]
    supy = y[0, geqY.sum(1)]
    infx = x[0, geqX.sum(1) - 1];
    infy = y[0, geqY.sum(1) - 1];

    x2 = x.eq(torch.reshape(supx, (n_paths, -1))).nonzero()[:, 1]
    x1 = x2 - 1
    y2 = y.eq(torch.reshape(supy, (n_paths, -1))).nonzero()[:, 1]
    y1 = y2 - 1

    R1 = ((supy - z[:, 1])) / (supy - infy) * vals[y1, x1] + ((z[:, 1] - infy) / (supy - infy)) * vals[y2, x1]
    R2 = ((supy - z[:, 1])) / (supy - infy) * vals[y1, x2] + ((z[:, 1] - infy) / (supy - infy)) * vals[y2, x2]

    vol = ((supx - z[:, 0])) / (supx - infx) * R1 + ((z[:, 0] - infx) / (supx - infx)) * R2

    return vol.float();


def lsSimPaths(spot, vols, expiry, r, n_steps, n_paths, x_grid, y_grid):
    # torch.manual_seed(123)

    one = torch.ones(n_paths)
    dt = torch.tensor(expiry / n_steps)
    paths = spot * one

    for i in range(1, n_steps):
        rand_norm = torch.randn(n_paths)
        time = dt * i
        vol = localVolCeil(vols, time, paths, x_grid, y_grid)
        paths = paths * torch.exp((r - 0.5 * vol ** 2) * dt + torch.sqrt(dt) * vol * rand_norm)
    return paths;


def LeasSquaresBS(vals, spot, vols, expiry, strikes, r, n_steps, n_paths):
    expiry = torch.tensor(expiry, dtype=torch.float)
    largest_expiry = max(expiry)
    min_spot = min(strikes + 1)
    max_spot = max(strikes - 1)

    n = int(len(vals))
    end_x = float(largest_expiry + 0.0001)

    ones = torch.ones((n, n_paths))
    r = torch.tensor(r)
    spot = torch.tensor(spot)
    vols = torch.tensor(vols, requires_grad=True)
    strikes = torch.tensor(strikes, dtype=torch.float).repeat(n_paths).reshape(n_paths, n).transpose(0, 1)
    vals = torch.tensor(vals, dtype=torch.float)

    x_grid = torch.tensor([np.linspace(0, end_x, vols.size()[1])] * n_paths)
    y_grid = torch.tensor(
        [np.append(np.append([18], np.linspace(min_spot, max_spot, vols.size()[0] - 2)), 60)] * n_paths)

    paths = lsSimPaths(spot, vols, largest_expiry, r, n_steps, n_paths, x_grid, y_grid) * ones
    unfloored_payoff = (paths - strikes)
    zeros = torch.tensor([[0.] * n_paths] * n)
    payoff = torch.max(unfloored_payoff, zeros)
    price = (torch.exp(-r * largest_expiry) * payoff).mean(1)
    sq_diffs = (vals - price) ** 2
    diff = torch.sum(sq_diffs).float()
    diff.backward()
    return diff, vols.grad


def Bachelier(spot, vol, expiry, strike, r):
    strike = np.exp(-r * expiry) * strike
    vega2 = (vol ** 2 / (2 * r)) * (1 - np.exp(-2 * r * expiry))
    price = (spot - strike) * stats.norm.cdf((spot - strike) / np.sqrt(vega2)) + np.sqrt(vega2) * stats.norm.pdf(
        (spot - strike) / (np.sqrt(vega2)))
    return price;


def BsCallAnalyticalTensor(spot, vol, expiry, strike, r):
    norm = torch.distributions.normal.Normal(torch.tensor(0.0), torch.tensor(1.0))
    d1 = 1 / (vol * torch.sqrt(expiry)) * (torch.log(spot / strike) + (r + vol * vol * 0.5) * expiry)
    d2 = d1 - vol * torch.sqrt(expiry)
    price_ana = norm.cdf(d1) * spot - norm.cdf(d2) * strike * torch.exp(-r * expiry)

    return price_ana;


def impliedVol(spot, price, expiry, strike, r):
    spot = torch.tensor(spot, dtype=torch.double)
    price = torch.tensor(price, dtype=torch.double)
    expiry = torch.tensor(expiry, dtype=torch.double)
    strike = torch.tensor(strike, dtype=torch.double)
    r = torch.tensor(r, dtype=torch.double)
    iv = torch.tensor([0.2], requires_grad=True, dtype=torch.double)

    current_val = BsCallAnalyticalTensor(spot, iv, expiry, strike, r)

    while abs(current_val - price) > 0.000001:
        current_val = BsCallAnalyticalTensor(spot, iv, expiry, strike, r)
        diff = price - current_val
        diff.backward()
        iv = (iv - diff / iv.grad).clone().detach().requires_grad_(True)
    return iv;


def DmodPV(spot, vols, expiry, strikes, r, n_steps, n_paths, largest_expiry, all_strike):
    expiry = torch.tensor(expiry, dtype=torch.float)
    min_spot = min(all_strike + 1)
    max_spot = max(all_strike - 1)
    torch.manual_seed(123)

    end_x = float(largest_expiry + 0.0001)

    r = torch.tensor(r)
    spot = torch.tensor(spot)
    expiry = torch.tensor(expiry)
    vols = torch.tensor(vols, requires_grad=True)
    strikes = torch.tensor(strikes)

    x_grid = torch.tensor([np.linspace(0, end_x, vols.size()[1])] * n_paths)
    y_grid = torch.tensor(
        [np.append(np.append([18], np.linspace(min_spot, max_spot, vols.size()[0] - 2)), 60)] * n_paths)

    paths = lsSimPaths(spot, vols, expiry, r, n_steps, n_paths, x_grid, y_grid)
    unfloored_payoff = (paths - strikes)
    zeros = torch.tensor([0.] * n_paths)
    payoff = torch.max(unfloored_payoff, zeros)
    price = (torch.exp(-r * expiry) * payoff).mean().double()
    price.backward()

    return vols.grad, price


def BsVega(spot, vol, expiry, strike, r):
    vol = vol.detach().double()
    d1 = 1 / (vol * np.sqrt(expiry)) * (np.log(spot / strike) + (r + vol * vol * 0.5) * expiry)
    vol_grad_ana = (spot * stats.norm.pdf(d1) * np.sqrt(expiry))
    return vol_grad_ana;


def BsCallAnalytical(spot, vol, expiry, strike, r):
    d1 = 1 / (vol * np.sqrt(expiry)) * (np.log(spot / strike) + (r + vol * vol * 0.5) * expiry)
    d2 = d1 - vol * np.sqrt(expiry)
    price_ana = stats.norm.cdf(d1) * spot - stats.norm.cdf(d2) * strike * np.exp(-r * expiry)
    return price_ana;


def AmCallLocal(spot, vol, expiry, strike, r, n_steps, n_paths, n_excer, autograd=False, tgrid=10, sgrid=10):
    device = "cpu"
    max_spot = spot * np.exp((r - 0.5 * vol * vol) * expiry + np.sqrt(expiry) * vol * 5.5)
    min_spot = spot * np.exp((r - 0.5 * vol * vol) * expiry - np.sqrt(expiry) * vol * 5.5)
    r = torch.tensor(r, requires_grad=autograd, device=device)
    spot = torch.tensor(spot, requires_grad=autograd, device=device)
    expiry = torch.tensor(expiry, requires_grad=autograd, device=device)
    vol = torch.tensor(vol, requires_grad=autograd, device=device)
    strike = torch.tensor(strike, requires_grad=autograd, device=device)
    dt = (expiry / n_steps).clone().detach().requires_grad_(True)
    vols = torch.tensor([[vol] * tgrid] * sgrid, requires_grad=autograd)

    # ---------------- Simulating paths ----------------#
    S0 = simPathsTensorLocal(spot, vols, expiry, r, n_steps, n_paths, max_spot, min_spot)

    # ---------------- Defining variables for backpass ----------------#
    unfloored_payoff = torch.tensor(strike - S0[n_steps - 1])
    zeros = torch.tensor([0.] * n_paths)
    opt_val = torch.max(unfloored_payoff, zeros)
    num = torch.FloatTensor(range(0, n_paths))

    backstep = int(np.floor((n_steps - 2) / n_excer))
    rest = (n_steps - 2) % n_excer

    for i in range(n_steps - 1 - backstep, rest, -backstep):
        # ---------------- Selecting options in the money ----------------#
        in_money = torch.max(strike - S0[i], zeros)
        locs = in_money == 0
        in_money_rows = num[~locs]
        out_money_rows = num[locs]
        m = torch.sum(locs == 0).item()

        if m == 0:
            opt_val = opt_val * torch.exp(-r * dt * backstep)
            continue;

        # ---------------- Variables used for regression ----------------#
        ones = torch.tensor([1.] * m)
        reg = S0[i][~locs] / strike
        s1 = lag1(reg)
        s2 = lag2(reg)
        s3 = lag3(reg)
        x = torch.reshape(torch.cat((ones, s1, s2, s3), 0), (4, m)).transpose(0, -1)
        y = (opt_val[~locs] * torch.exp(-r * dt * backstep)) / strike

        # ---------------- Performing regression and calculating continuation value ----------------#
        beta = svdReg(x, y)
        con_val_in = torch.mv(x, beta) * strike
        opt_val_out = opt_val[locs] * torch.exp(-r * dt * backstep)
        opt_val_in = torch.max((strike - S0[i][~locs]), con_val_in)

        # ---------------- Restoring original indices by a permutation tensor  ----------------#
        perm = torch.cat((in_money_rows, out_money_rows), 0).sort(0)[1]
        opt_val = torch.cat((opt_val_in, opt_val_out), 0)[perm]

    opt_val = opt_val * torch.exp(-r * dt * (rest + 1))
    square_root = np.sqrt(n_paths)
    std = opt_val.std() / square_root

    price = opt_val.mean()

    # ---------------- Doing a backward sweep if autograd == True ----------------#

    if autograd == True:
        price.backward()
        s_t_grid = grid(vols, expiry, max_spot, min_spot)
        output = (price.item(), r.grad.item(), spot.grad.item(), -expiry.grad.item(), std.item(), vols.grad.numpy())
    else:
        output = price

    return output;


def AmCallLocalOptim(val, spot, vol, expiry, strike, r, n_steps, n_paths, n_excer, autograd=False, tgrid=10, sgrid=10):
    device = "cpu"
    max_spot = spot * np.exp((r - 0.5 * 1 ** 2) * expiry + np.sqrt(expiry) * 1 * 5.5)
    min_spot = spot * np.exp((r - 0.5 * 1 ** 2) * expiry - np.sqrt(expiry) * 1 * 5.5)
    r = torch.tensor(r, requires_grad=autograd, device=device)
    spot = torch.tensor(spot, requires_grad=autograd, device=device)
    expiry = torch.tensor(expiry, requires_grad=autograd, device=device)
    vol = torch.tensor(vol, requires_grad=autograd, device=device)
    strike = torch.tensor(strike, requires_grad=autograd, device=device)
    dt = (expiry / n_steps).clone().detach().requires_grad_(True)

    # ---------------- Simulating paths ----------------#
    S0 = simPathsTensorLocal(spot, vol, expiry, r, n_steps, n_paths, max_spot, min_spot)

    # ---------------- Defining variables for backpass ----------------#
    unfloored_payoff = torch.tensor(strike - S0[n_steps - 1])
    zeros = torch.tensor([0.] * n_paths)
    opt_val = torch.max(unfloored_payoff, zeros)
    num = torch.FloatTensor(range(0, n_paths))

    backstep = int(np.floor((n_steps - 2) / n_excer))
    rest = (n_steps - 2) % n_excer

    for i in range(n_steps - 1 - backstep, rest, -backstep):
        # ---------------- Selecting options in the money ----------------#
        in_money = torch.max(strike - S0[i], zeros)
        locs = in_money == 0
        in_money_rows = num[~locs]
        out_money_rows = num[locs]
        m = torch.sum(locs == 0).item()

        if m == 0:
            opt_val = opt_val * torch.exp(-r * dt * backstep)
            continue;

        # ---------------- Variables used for regression ----------------#
        ones = torch.tensor([1.] * m)
        reg = S0[i][~locs] / strike
        s1 = lag1(reg)
        s2 = lag2(reg)
        s3 = lag3(reg)
        x = torch.reshape(torch.cat((ones, s1, s2, s3), 0), (4, m)).transpose(0, -1)
        y = (opt_val[~locs] * torch.exp(-r * dt * backstep)) / strike

        # ---------------- Performing regression and calculating continuation value ----------------#
        beta = svdReg(x, y)
        con_val_in = torch.mv(x, beta) * strike
        opt_val_out = opt_val[locs] * torch.exp(-r * dt * backstep)
        opt_val_in = torch.max((strike - S0[i][~locs]), con_val_in)

        # ---------------- Restoring original indices by a permutation tensor  ----------------#
        perm = torch.cat((in_money_rows, out_money_rows), 0).sort(0)[1]
        opt_val = torch.cat((opt_val_in, opt_val_out), 0)[perm]

    opt_val = opt_val * torch.exp(-r * dt * (rest + 1))
    square_root = np.sqrt(n_paths)
    std = opt_val.std() / square_root

    price = (val - opt_val.mean()) ** 2

    # ---------------- Doing a backward sweep if autograd == True ----------------#

    if autograd == True:
        price.backward()
        s_t_grid = grid(vol, expiry, max_spot, min_spot)
        output = (
            price.item(), r.grad.item(), spot.grad.item(), -expiry.grad.item(), std.item(), vol.grad, opt_val.mean())
    else:
        output = price

    return output;


def AmCallLocalBatch(val, spot, vol, expiry, strike, r, n_steps, n_paths, n_excer, autograd=False, tgrid=10, sgrid=10):
    reps = int(n_paths / 1000)
    result = AmCallLocalOptim(val, spot, vol, expiry, strike, r, n_steps, 1000, n_excer, autograd, tgrid, sgrid)
    for i in range(0, reps - 1):
        result = tuple(map(operator.add, result,
                           AmCallLocalOptim(val, spot, vol, expiry, strike, r, n_steps, 1000, n_excer, autograd, tgrid,
                                            sgrid)))
    return tuple([(1 / (n_paths / 1000)) * x for x in result])


def AmCall(spot, vol, expiry, strike, r, n_steps, n_paths, n_excer, autograd=False, device="cpu"):
    r = torch.tensor(r, requires_grad=autograd, device=device)
    spot = torch.tensor(spot, requires_grad=autograd, device=device)
    expiry = torch.tensor(expiry, requires_grad=autograd, device=device)
    vol = torch.tensor(vol, requires_grad=autograd, device=device)
    strike = torch.tensor(strike)
    dt = (expiry / n_steps)
    # ---------------- Simulating paths ----------------#
    S0 = simPathsTensor(spot, vol, expiry, r, n_steps, n_paths)

    # ---------------- Defining variables for backpass ----------------#
    unfloored_payoff = torch.tensor(strike - S0[n_steps - 1])
    zeros = torch.tensor([0.] * n_paths)
    opt_val = torch.max(unfloored_payoff, zeros)
    num = torch.FloatTensor(range(0, n_paths))

    backstep = int(np.floor((n_steps - 2) / n_excer))
    rest = (n_steps - 2) % n_excer

    for i in range(n_steps - 1 - backstep, rest, -backstep):
        # ---------------- Selecting options in the money ----------------#
        in_money = torch.max(strike - S0[i], zeros)
        locs = in_money == 0
        in_money_rows = num[~locs]
        out_money_rows = num[locs]
        m = torch.sum(locs == 0).item()

        if m == 0:
            opt_val = opt_val * torch.exp(-r * dt * backstep)
            continue;

        # ---------------- Variables used for regression ----------------#
        ones = torch.tensor([1.] * m)
        reg = S0[i][~locs] / strike
        s1 = lag1(reg)
        s2 = lag2(reg)
        s3 = lag3(reg)
        x = torch.reshape(torch.cat((ones, s1, s2, s3), 0), (4, m)).transpose(0, -1)
        y = (opt_val[~locs] * torch.exp(-r * dt * backstep)) / strike

        # ---------------- Performing regression and calculating continuation value ----------------#
        beta = svdReg(x, y)
        con_val_in = torch.mv(x, beta) * strike
        opt_val_out = opt_val[locs] * torch.exp(-r * dt * backstep)
        opt_val_in = torch.max((strike - S0[i][~locs]), con_val_in)

        # ---------------- Restoring original indices by a permutation tensor  ----------------#
        perm = torch.cat((in_money_rows, out_money_rows), 0).sort(0)[1]
        opt_val = torch.cat((opt_val_in, opt_val_out), 0)[perm]

    opt_val = opt_val * torch.exp(-r * dt * (rest + 1))
    square_root = np.sqrt(n_paths)
    std = opt_val.std() / square_root

    price = opt_val.mean()

    # ---------------- Doing a backward sweep if autograd == True ----------------#

    if autograd:
        price.backward()
        output = (price.item(), r.grad.item(), spot.grad.item(), vol.grad.item(), -expiry.grad.item(), std.item())
    else:
        output = price, std

    return output
