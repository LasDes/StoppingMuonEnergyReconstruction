import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rcParams
from plotSetup import setupRcParams, COLORS

x = np.linspace(0.0, 6.0, 200)
y1 = np.sin(x) * x
y2 = np.cos(5 * x) - x
y3 = np.cos(7 * x) + x
y4 = np.exp(np.cos(2 * x))

setupRcParams(rcParams, grid=True)
plt.plot(x, y1, color=COLORS["r"], linestyle="-", label="First Plot")
plt.plot(x, y2, color=COLORS["g"], linestyle="-", label="Second Plot")
plt.plot(x, y3, color=COLORS["b"], linestyle="-", label="Third Plot")
plt.plot(x, y4, color=COLORS["y"], linestyle="-", label="Fourth Plot")
plt.xlabel("x-Axis Label")
plt.ylabel("y-Axis Label")
plt.legend(loc="best")
plt.savefig("test_tu.pdf")
