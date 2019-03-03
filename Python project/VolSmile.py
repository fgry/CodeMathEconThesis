from sys import platform as sys_pf

if sys_pf == 'darwin':
    import matplotlib

    matplotlib.use("TkAgg")

import sys

reload(sys)
sys.setdefaultencoding('utf8')

from ggplot import *
import pandas as pd
import torch
import numpy as np

import Functions as f

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
iv_calc = torch.zeros(prices.size())

for i in range(0, n):
    iv_calc[i] = f.impliedVol(spot, prices[i], expiries[i], strike[i], r)

iv = torch.FloatTensor(iv)

print(iv_calc, iv)

x = np.empty((n, 3), dtype='O')
x[0:11, 0] = iv.detach()
x[0:11, 1] = "Bachelier smile"
x[0:11, 2] = strike

x = pd.DataFrame(data=x, columns=("Implied Vol", "Type", "Strike"))

p = ggplot(aes(y='Implied Vol', x='Strike', colour="Type"), data=x) + geom_line(size=3, alpha=0.7) + ggtitle(
    'Bachelier volatility smile')
t = theme_gray()
t._rcParams['font.size'] = 20

p = p + t
p.show()

x = np.empty((11, 3), dtype='O')

x[0:11, 0] = iv_calc.detach() - iv.detach()
x[0:11, 1] = strike
x[0:11, 2] = "Error"

x = pd.DataFrame(data=x, columns=("Difference", "Strike", "Type"))
p = ggplot(aes(y='Difference', x='Strike', colour="Type"), data=x) + geom_line(size=3, alpha=0.7) + ggtitle(
    'Comparison of volatility smiles')
t = theme_gray()
t._rcParams['font.size'] = 20

p = p + t
p.show()

x = np.empty((n * 2, 3), dtype='O')
x[0:11, 0] = iv.detach()
x[0:11, 1] = "Bachelier Vol Smile"
x[11:22, 0] = iv_calc.detach()
x[11:22, 1] = "Calibrated Vol Smile"
x[0:11, 2] = strike
x[11:22, 2] = strike

x = pd.DataFrame(data=x, columns=("Implied Vol", "Type", "Strike"))

p = ggplot(aes(y='Implied Vol', x='Strike', colour='Type'), data=x) + geom_line(size=3, alpha=0.7) + ggtitle(
    'Comparison of volatility smiles')
t = theme_gray()
t._rcParams['font.size'] = 20

p = p + t
p.show()
