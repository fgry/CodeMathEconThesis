import numpy as np

import Functions as f
import array2latex

strike = 40.
r = 0.06
n_paths = 10000

parameters = np.array(([36., .20, 1., strike, r, 51, n_paths, 49],
                       [36., .20, 2., strike, r, 102, n_paths, 100],
                       [36., .40, 1., strike, r, 51, n_paths, 49],
                       [36., .40, 2., strike, r, 102, n_paths, 100],
                       [38., .20, 1., strike, r, 51, n_paths, 49],
                       [38., .20, 2., strike, r, 102, n_paths, 100],
                       [38., .40, 1., strike, r, 51, n_paths, 49],
                       [38., .40, 2., strike, r, 102, n_paths, 100],
                       [40., .20, 1., strike, r, 51, n_paths, 49],
                       [40., .20, 2., strike, r, 102, n_paths, 100],
                       [40., .40, 1., strike, r, 51, n_paths, 49],
                       [40., .40, 2., strike, r, 102, n_paths, 100],
                       [42., .20, 1., strike, r, 51, n_paths, 49],
                       [42., .20, 2., strike, r, 102, n_paths, 100],
                       [42., .40, 1., strike, r, 51, n_paths, 49],
                       [42., .40, 2., strike, r, 102, n_paths, 100],
                       [44., .20, 1., strike, r, 51, n_paths, 49],
                       [44., .20, 2., strike, r, 102, n_paths, 100],
                       [44., .40, 1., strike, r, 51, n_paths, 49],
                       [44., .40, 2., strike, r, 102, n_paths, 100]), dtype='O')

output = np.zeros((20, 9), dtype='O')

for i in range(0, 20):
    output[i, 0:3] = (int(parameters[i, 0]), np.round(parameters[i, 1], 2), int(parameters[i, 2]))
    output[i, 3:9] = np.round(np.asarray((f.AmCall(parameters[i, 0], parameters[i, 1], parameters[i, 2],
                                                   parameters[i, 3], parameters[i, 4], parameters[i, 5],
                                                   parameters[i, 6], parameters[i, 7], True))), 3)

table = array2latex.tolatex(output, header=(
    'Spot', 'Vol', 'Expiry', 'Price', '$\\rho$', '$\Delta$', '$\\mathcal{V}$', '$\\Theta$', '(s.e)'))

print(table)
