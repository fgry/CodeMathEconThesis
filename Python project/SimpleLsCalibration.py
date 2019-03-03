from sys import platform as sys_pf

if sys_pf == 'darwin':
    import matplotlib

    matplotlib.use("TkAgg")

import sys

reload(sys)
sys.setdefaultencoding('utf8')
import torch
import Functions as f
import numpy as np

###------------------------- Defining volatility matrix -------------------------###
n_paths = 30000
n_steps = 40
tdim = 5
sdim = 5
vol_floor = torch.zeros((sdim, tdim)).double()
vol = torch.tensor([[0.2] * tdim] * sdim).double()
m = tdim * sdim
n = 3
r = 0.02

###------------------------- Calculating Data -------------------------###

iv = torch.zeros(n)
spot = float(35.)
expiries = np.array([0.25] * n)
strike = np.linspace(34., 36., n).astype(float)
vals = torch.DoubleTensor(f.Bachelier(spot, 7., expiries, strike, r))

for i in range(0, vals.size()[0]):
    iv[i] = f.impliedVol(spot, vals[i], expiries[i], strike[i], r)

###------------------------- ADAM Parameters -------------------------###
expiries = torch.tensor(expiries)

beta1 = 0.9
beta2 = 0.999
mt = 0.0
vt = 0.0
vhat = torch.zeros((sdim, tdim)).double()
LearningRate = 0.005

#
# ###------------------------- Applying least squares by AMSGRAD -------------------------###
i = 1.
kor = f.LeasSquaresBS(vals, spot, vol, expiries, strike, r, n_steps, n_paths)
while i < 100:
    LearningRate = max(LearningRate / (1 + 0.0005 * i), 0.0005)
    print(LearningRate, kor[0])
    mt = beta1 * mt + (1 - beta1) * kor[1]
    vt = beta2 * vt + (1 - beta2) * kor[1] ** 2
    mhat = mt / (1 - beta1 ** i)
    vhat = torch.max(vt / (1 - beta2 ** i), vhat)
    vol = torch.max(vol - mhat * LearningRate / (torch.sqrt(vhat) + 10 ** -8), vol_floor)
    kor = f.LeasSquaresBS(vals, spot, vol, expiries, strike, r, n_steps, n_paths)
    print(mhat * LearningRate / (torch.sqrt(vhat) + 10 ** -8), vol)
    i += 1.
print(vol, kor)

###------------------------- Defining matrices for market risk computations -------------------------###


Dpmarf = torch.zeros((m, n)).double()
DpmarG = torch.zeros((1, n)).double()
Dpmodf = torch.zeros((m, m)).double()

DpmargDiag = torch.zeros((n, n)).double()
prices = torch.zeros(n).double()

DpmodPV = torch.zeros(n, m).double()

i = 0

###------------------------- Computing matrices for market risk -------------------------###

while i < len(expiries):
    kor = f.DmodPV(spot, vol, expiries[i], strike[i], r, n_steps, 100000, max(expiries), strike)
    DpmodPVi = kor[0]
    prices[i] = kor[1]
    print(DpmodPVi)
    DpmarG[0, i] = f.BsVega(spot, iv[i], expiries[i], strike[i], r)
    DpmargDiag[i, i] = DpmarG[0, i]
    print(DpmodPVi.reshape(m, 1))
    Dpmarf += torch.matmul(DpmodPVi.reshape(m, 1), DpmarG)
    Dpmodf += torch.matmul(DpmodPVi.reshape(m, 1), DpmodPVi.reshape(1, m))
    DpmarG[0, i] = 0.
    DpmodPV[i, :] = DpmodPVi.reshape(1, m)
    i += 1

Dpmodf = 2 * Dpmodf[1:4, 1:4]
Dpmarf = -2 * Dpmarf[1:4, :]
DpmodPV = DpmodPV[:, 5:20]

dpxi = -torch.pinverse(Dpmodf, 0.1).mm(Dpmarf)
tmp = torch.matmul(dpxi, DpmargDiag)
dpmarpv = torch.matmul(DpmodPV.reshape(n, 15), tmp)
np.set_printoptions(suppress=True)
print(dpmarpv.numpy())
print(prices)
