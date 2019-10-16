from matplotlib import pyplot as plt
import numpy as np
from plotSetup import setupRcParams, COLORS as C
from matplotlib import rcParams
from matplotlib.ticker import MultipleLocator
from scipy.interpolate import interp1d

setupRcParams(rcParams)
rcParams["mathtext.default"] = "regular"

d = np.loadtxt("../featureGeneration/ice_coeff.txt", delimiter=" ")
d[:, 1] *= 1e-2
d[:, 3] *= 1e-2
d[:, 2] *= 1e-3
d[:, 4] *= 1e-3

pop1 = d[:, 3] == 0.0
pop2 = d[:, 3] < 0
dust = [1995, 2100]

z = np.linspace(np.min(d[:, 0]), np.max(d[:, 0]), 1000)
b = interp1d(d[:, 0], d[:, 1], kind=3)
a = interp1d(d[:, 0], d[:, 2], kind=3)

fig, ax = plt.subplots(2, figsize=(6, 5), sharex=True)

#ax[0].plot(z, b(z), color=C["g"], linestyle="-")
ax[0].errorbar(d[~(pop1 | pop2), 0], d[~(pop1 | pop2), 1],
	           d[~(pop1 | pop2), 3],
               linestyle="", markeredgewidth=0, marker=".", color=C["g_dark"],
               label="Optical Measurements")
ax[0].errorbar(d[pop1 | pop2, 0], d[pop1 | pop2, 1],
               linestyle="", markeredgewidth=0, marker="^", color=C["g_dark"],
               label="Extra/Interpolated", markersize=4)
ax[0].axvline(dust[0], color="k", linestyle=":")
ax[0].axvline(dust[1], color="k", linestyle=":")
ax[0].xaxis.set_minor_locator(MultipleLocator(100))
ax[0].yaxis.set_minor_locator(MultipleLocator(0.01))

#ax[1].plot(z, a(z), color=C["g"], linestyle="-")
ax[1].errorbar(d[~(pop1 | pop2), 0], d[~(pop1 | pop2), 2],
	           d[~(pop1 | pop2), 4],
               linestyle="", markeredgewidth=0, marker=".", color=C["g_dark"],
               label="Optical Measurements")
ax[1].errorbar(d[pop1 | pop2, 0], d[pop1 | pop2, 2],
               linestyle="", markeredgewidth=0, marker="^", color=C["g_dark"],
               label="Extra/Interpolated", markersize=4)
ax[1].axvline(dust[0], color="k", linestyle=":")
ax[1].axvline(dust[1], color="k", linestyle=":")
ax[1].xaxis.set_minor_locator(MultipleLocator(100))
ax[1].yaxis.set_minor_locator(MultipleLocator(0.001))

ax[1].set_xlabel("Depth / m")
ax[0].set_ylabel("Effective\nScattering Coefficient / $m^{-1}$")
ax[1].set_ylabel("Dust Absorptivity / $m^{-1}$")
ax[1].set_ylim([0.005, 0.034])
ax[0].legend(loc="best", frameon=False)
ax[1].legend(loc="best", frameon=False)
plt.tight_layout()
plt.subplots_adjust(hspace=0)
plt.savefig("dust_plot.pdf")