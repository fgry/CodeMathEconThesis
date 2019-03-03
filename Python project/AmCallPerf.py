from sys import platform as sys_pf

if sys_pf == 'darwin':
    import matplotlib

    matplotlib.use("TkAgg")

import sys

reload(sys)
sys.setdefaultencoding('utf8')

from ggplot import *
import pandas as pd
import numpy as np
from array2latex import tolatex

npaths = np.array([1000, 2000, 3000, 4000, 5000, 10000, 20000, 30000])
res = np.empty([4, 8], dtype=np.double)

res[0, :] = np.array([0.2, 0.27, 0.4, 0.5, 0.6, 1.2, 2.4, 4.4])
res[1, :] = np.array([0.5, 1.1, 2.1, 3.0, 4.7, 32.2, 105.6, 262.7])
res[2, :] = np.array([0.6, 1.1, 1.5, 2.4, 3.7, 12.2, 47.9, 108.7])
res[3, :] = np.array([0.5, 1.0, 1.5, 1.9, 2.5, 5.4, 9.7, 14.7])

# calcs = np.array([1, 2, 3, 4, 5, 10, 20, 30])


# for j in range(0,8):
#  now = t.time()
# tmp = f.AmCallLocal(36., 0.2, 1., 40., 0.06, 51, 1000, 49, True)
# for i in range(0,calcs[j]-1):
#  tmp += f.AmCallLocal(36., 0.2, 1., 40., 0.06, 51, 1000, 49, True)
# res[3,j] = t.time() - now


print(res)
x = np.empty((8 * 4, 3), dtype='O')
x[0:8, 0] = res[0, :]
x[0:8, 1] = 'No AD'
x[0:8, 2] = npaths
x[8:16, 2] = npaths
x[16:24, 2] = npaths
x[8:16, 0] = res[2, :]
x[8:16, 1] = 'Google Cloud (50gb RAM)'
x[16:24, 0] = res[1, :]
x[16:24, 1] = 'Laptop (16gb RAM)'
x[24:32, 0] = res[3, :]
x[24:32, 1] = 'Laptop-Batch'
x[24:32, 2] = npaths

x = pd.DataFrame(data=x, columns=("Time in seconds", "Type", "Simulated Paths"))

p = ggplot(aes(y='Time in seconds', x='Simulated Paths', colour='Type'), data=x) + \
    geom_line(size=3, alpha=0.7) + \
    ggtitle('Comparison of computational time')

t = theme_gray()
t._rcParams['font.size'] = 20

p = p + t
p.show()

noad = res[0, :]
gcloud = res[2, :]
laptop = res[1, :]
laptop_batch = res[3, :]

factor = np.empty([4, 8], dtype='O')

factor[0, :] = npaths
factor[1, :] = gcloud / noad
factor[2, :] = laptop / noad
factor[3, :] = laptop_batch / noad

factor = np.around(np.transpose(factor.astype(np.double)), 1).astype(np.object)

print(factor)

table = tolatex(factor, header=("Simulated Paths", "Google Cloud", "Laptop", "Laptop - Batch"))

print(table)
