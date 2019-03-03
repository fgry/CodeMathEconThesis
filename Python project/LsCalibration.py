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
n_paths = 50000
n_steps = 40
tdim = 5
sdim = 10
vol_floor = torch.zeros((sdim, tdim)).double()
vol = torch.tensor([[0.2] * tdim] * sdim).double()
m = tdim * sdim
n = 11
r = 0.02

###------------------------- Calculating Data -------------------------###

iv = torch.zeros(n)
spot = float(35.)
expiries = np.array([0.25] * n)
strike = np.linspace(30., 40., n).astype(float)
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
while i < 150:
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


vol = torch.tensor([[0.2228, 0.2322, 0.2360, 0.2462, 0.2556],
                    [0.2236, 0.2245, 0.2272, 0.2328, 0.2393],
                    [0.2167, 0.2185, 0.2194, 0.2191, 0.2186],
                    [0.2136, 0.2148, 0.2139, 0.2110, 0.2074],
                    [0.2103, 0.2118, 0.2094, 0.2039, 0.1991],
                    [0.2036, 0.2019, 0.1990, 0.1967, 0.1946],
                    [0.1914, 0.1936, 0.1945, 0.1874, 0.1944],
                    [0.1921, 0.1783, 0.1792, 0.1868, 0.1893],
                    [0.1777, 0.1789, 0.1787, 0.1788, 0.1762],
                    [0.1527, 0.1758, 0.1734, 0.1740, 0.1673]]).double()

Dpmarf = torch.zeros((m, n)).double()
DpmarG = torch.zeros((1, n)).double()
Dpmodf = torch.zeros((m, m)).double()

DpmargDiag = torch.zeros((n, n)).double()
prices = torch.zeros(n).double()

DpmodPV = torch.zeros(n, m).double()

i = 0

###------------------------- Computing matrices for market risk -------------------------###

while i < len(expiries):
    kor = f.DmodPV(spot, vol, expiries[i], strike[i], r, n_steps, 500000, max(expiries), strike)
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

Dpmodf = 2 * Dpmodf
Dpmarf = -2 * Dpmarf

dpxi = -Dpmodf.pinverse(0.9).mm(Dpmarf)
tmp = torch.mm(dpxi, DpmargDiag)
dpmarpv = torch.mm(DpmodPV.reshape(n, m), tmp)
np.set_printoptions(suppress=True, precision=4)
print(dpmarpv)
print(prices)

dpxi2 = DpmodPV.pinverse(0.5).mm(DpmargDiag)
tmp2 = torch.mm(dpxi2, DpmargDiag)
dpmarpv2 = torch.mm(DpmodPV.reshape(n, m), tmp2)
print(dpmarpv2)
# x = np.linspace(0, 0.25 + 0.0001, 5)
# y = np.linspace(30, 40, 13 - 2)
# x, y = np.meshgrid(x, y)
# z = vol[1:12,:].numpy()
#
# fig = plt.figure()
# ax = fig.gca(projection='3d')
#
#
# # Plot the surface.
# surf = ax.plot_surface(x, y, z, cmap='inferno',
#                        linewidth=1, antialiased=True)
# # Customize the z axis.
# ax.zaxis.set_major_locator(LinearLocator(10))
# ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
# ax.set_title('Local volatility surface')
# ax.set_zlabel('Volatility')
#
# plt.ylabel('Space')
# plt.xlabel('Time')
# # Add a color bar which maps values to colors.
# fig.colorbar(surf, shrink=0.5, aspect=5)
#
# plt.show()
#
