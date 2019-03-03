import time as t

import numpy as np

import Functions as f

calcs = [1, 2, 3, 4, 5, 10, 20, 36]
res = np.empty(8)

for j in range(0, 8):
    now = t.time()
    tmp = f.AmCallLocal(36., 0.2, 1., 40., 0.06, 51, 1000, 49, True)
    for i in range(0, calcs[j] - 1):
        tmp += f.AmCallLocal(36., 0.2, 1., 40., 0.06, 51, 1000, 49, True)
    res[j] = t.time() - now

print(res)
