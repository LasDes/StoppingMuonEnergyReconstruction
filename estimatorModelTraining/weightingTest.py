import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rcParams

from icecube.weighting.fluxes import GaisserH3a, Hoerandel5
from icecube.weighting import weighting

from dataMethods import load_data
from weighting import compoundWeightGenerator

from plotSetup import setupRcParams, COLORS as C

N_8 = 27849
N_7 = 5234

lbl_8, att_8, w, g = load_data("../level4_11058_27849.hdf5")
lbl_7, att_7, w, g = load_data("../level4_11057_5234.hdf5")

gen = compoundWeightGenerator()
gen.add_generator("./gen_11058.pickle", 11058, N_8)
gen.add_generator("./gen_11057.pickle", 11057, N_7)

energy_7 = lbl_7["energy"].as_matrix()[:, 0]
ptype_7 = lbl_7["pdg_encoding"].as_matrix()[:, 0]
energy_8 = lbl_8["energy"].as_matrix()[:, 0]
ptype_8 = lbl_8["pdg_encoding"].as_matrix()[:, 0]

energy = np.concatenate((energy_7, energy_8))
ptype = np.concatenate((ptype_7, ptype_8))
weights = gen.get_weight(energy, ptype)

setupRcParams(rcParams)

log_bins = np.logspace(2.0, 9.0, 150)
energies = [energy[ptype == 2212],
            energy[ptype == 1000020040],
            energy[ptype == 1000070140],
            energy[ptype == 1000130270],
            energy[ptype == 1000260560]]
weights_all = [weights[ptype == 2212],
           weights[ptype == 1000020040],
           weights[ptype == 1000070140],
           weights[ptype == 1000130270],
           weights[ptype == 1000260560]]

plt.hist(energies,
         bins=log_bins,
         weights=weights_all,
         histtype="stepfilled",
         stacked=True,
         color=[C["b"], C["r"], C["y"], C["m"], C["g"]],
         linewidth=0,
         label=["H", "He", "N", "Al", "Fe"])

plt.legend(loc="best", frameon=False)
plt.xscale("log")
plt.ylim([0.0, 0.06])
plt.xlabel("Primary Energy / GeV")
plt.ylabel("Frequency / Hz")
plt.savefig("weighted_fluxes_correct.pdf")
plt.close()

flux = GaisserH3a()
gen_7 = weighting.from_simprod(11057)
gen_8 = weighting.from_simprod(11058)
weights_7 = flux(energy_7, ptype_7) / gen_7(energy_7, ptype_7) / N_7
weights_8 = flux(energy_8, ptype_8) / gen_8(energy_8, ptype_8) / N_8

weights = np.concatenate((weights_7, weights_8))

log_bins = np.logspace(2.0, 9.0, 150)
energies = [energy[ptype == 2212],
            energy[ptype == 1000020040],
            energy[ptype == 1000070140],
            energy[ptype == 1000130270],
            energy[ptype == 1000260560]]
weights_all = [weights[ptype == 2212],
           weights[ptype == 1000020040],
           weights[ptype == 1000070140],
           weights[ptype == 1000130270],
           weights[ptype == 1000260560]]

plt.hist(energies,
         bins=log_bins,
         weights=weights_all,
         histtype="stepfilled",
         stacked=True,
         color=[C["b"], C["r"], C["y"], C["m"], C["g"]],
         linewidth=0,
         label=["H", "He", "N", "Al", "Fe"])

plt.legend(loc="best", frameon=False)
plt.xscale("log")
plt.ylim([0.0, 0.06])
plt.xlabel("Primary Energy / GeV")
plt.ylabel("Frequency / Hz")
plt.savefig("weighted_fluxes_wrong.pdf")
plt.close()