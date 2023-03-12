import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = Axes3D(fig)
x = np.arange(-10, 10, 0.1)
y = np.arange(-10, 10, 0.1)
X, Y = np.meshgrid(x, y)
plt.xlabel('x')
plt.ylabel('y')
Z = X**2 + Y**2
ax.plot_surface(X, Y, Z, rstride = 1, cstride = 1, cmap = 'rainbow')
plt.show()