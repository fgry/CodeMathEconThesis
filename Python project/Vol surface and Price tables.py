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
from matplotlib import pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import array2latex

vol = torch.tensor([[0.2228, 0.2322, 0.2360, 0.2462, 0.2556],
                    [0.2236, 0.2245, 0.2272, 0.2328, 0.2393],
                    [0.2167, 0.2185, 0.2194, 0.2191, 0.2186],
                    [0.2136, 0.2148, 0.2139, 0.2110, 0.2074],
                    [0.2103, 0.2118, 0.2094, 0.2039, 0.1991],
                    [0.2036, 0.2019, 0.1990, 0.1967, 0.1946],
                    [0.1914, 0.1936, 0.1945, 0.1874, 0.1944],
                    [0.1921, 0.1783, 0.1792, 0.1868, 0.1893],
                    [0.1777, 0.1789, 0.1787, 0.1788, 0.1762],
                    [0.1527, 0.1758, 0.1734, 0.1740, 0.1673]])

x = np.linspace(0, 0.25, 5)
y = np.linspace(30, 40, 10 - 2)
x, y = np.meshgrid(x, y)
z = vol[1:9, :].numpy()

fig = plt.figure()
ax = fig.gca(projection='3d')

# Plot the surface.
surf = ax.plot_surface(x, y, z, cmap='inferno',
                       linewidth=1, antialiased=True)
# Customize the z axis.
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
ax.set_title('Local volatility surface')
ax.set_zlabel('Volatility')
ax.autoscale(enable=True, axis='both', tight=None)

plt.ylabel('Space')
plt.xlabel('Time')
# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()

vol = vol.numpy()

print(vol)

table = array2latex.tolatex(np.round(vol.transpose(), 3),
                            header=('s_0', 's_1', 's_2', 's_3', 's_4', 's_5', '$s_6', 's_7', 's_8', 's_9', 's_10'))

print(table)

n = 11
r = 0.02

iv = torch.zeros(n)
spot = float(35.)
expiries = np.array([0.25] * n).astype(float)
strike = np.linspace(30., 40., n).astype(float)
vals = torch.DoubleTensor(f.Bachelier(spot, 7., expiries, strike, r))

for i in range(0, n):
    iv[i] = f.impliedVol(spot, vals[i], expiries[i], strike[i], r)

prices = torch.tensor([5.2601, 4.3583, 3.5103, 2.7362, 2.0551, 1.4815, 1.0209, 0.6703, 0.4168,
                       0.2445, 0.1361])

vals = torch.tensor(vals, dtype=torch.float)

table = vals.reshape(1, n).numpy()

table = array2latex.tolatex(np.round(table, 3),
                            header=('30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40'))
print(table)

table = torch.cat([vals, prices], -1).reshape(2, n).numpy()

table = array2latex.tolatex(np.round(table, 4),
                            header=('30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40'))
print(table)
