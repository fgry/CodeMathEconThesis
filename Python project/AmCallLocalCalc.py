import numpy as np

import Functions as f

matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

data = f.AmCallLocal(36.0, 0.2, 1., 40., 0.06, 51, 20000, 49, True, 30, 30)

x = data[6][0]
y = data[6][1][4:15]

print(x)
print(y)

x, y = np.meshgrid(x, y)
z = data[5][4:15, :]
print(z)
fig = plt.figure()
ax = fig.gca(projection='3d')

# Plot the surface.
surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
                       linewidth=1, antialiased=True)
# Customize the z axis.
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
ax.set_title('Vega surface')
ax.set_zlabel('Vega')

plt.ylabel('Spot')
plt.xlabel('Time')
# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()

# plt.show()
