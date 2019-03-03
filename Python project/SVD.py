import numpy as np
import torch

import array2latex

singular = torch.tensor([2.2549e+01, 3.9496e+00, 1.4275e+00, 5.3769e-01, 1.9534e-01, 6.5469e-02,
                         2.1100e-02, 6.7860e-03, 1.3990e-03, 7.1395e-04, 1.5285e-05])

singular = singular.numpy().repeat(2).reshape(11, 2).transpose().round(5)
np.set_printoptions(suppress=True)

table = array2latex.tolatex((singular), header=('1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11'))

print(table)
