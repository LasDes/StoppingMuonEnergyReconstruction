import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.stats import gaussian_kde
from matplotlib.ticker import AutoMinorLocator, MultipleLocator

from loadData import loadDataHD5

N_SAMPLES = 1000
WHICH_ATT = "zenith_SplineMPE"

def getByName(name, data_names, data):
    try:
        return data[:, data_names == name].flatten()
    except KeyError:
        print("Keyword not found: %s" % name)

(lbl_names, lbl, att_names, att) = loadDataHD5("../extrF.hdf5")

dat = getByName(WHICH_ATT, att_names, att)
lab = getByName("label_det", lbl_names, lbl)
dat1 = dat[lab == 1]
dat2 = dat[lab == 0]

dat1 = dat1[np.isfinite(dat1)]
dat2 = dat2[np.isfinite(dat2)]
kernel1 = gaussian_kde(dat1)
kernel2 = gaussian_kde(dat2)

t = np.linspace(np.min(dat), np.max(dat), N_SAMPLES)
K1 = np.reshape(kernel1(t), N_SAMPLES)
K2 = np.reshape(kernel2(t), N_SAMPLES)

rcParams["font.family"] = "Arial"
fig, ax = plt.subplots(2, sharex=True, figsize=(8,8))

ax[0].fill_between(t, 0.0, 1013.0 / 2154.0 * K1, alpha=0.2, color="b")
ax[0].fill_between(t, 0.0, 1141.0 / 2154.0 * K2, alpha=0.2, color="r")
ax[0].fill_between(t, 0.0, 1013.0 / 2154.0 * K1 + 1141.0 / 2154.0 * K2,
                   alpha=0.1, color="k")
ax[0].plot(t, 1013.0 / 2154.0 * K1, "b")
ax[0].plot(t, 1141.0 / 2154.0 * K2, "r")
ax[0].plot(t, 1013.0 / 2154.0 * K1 + 1141.0 / 2154.0 * K2, "k")
ax[0].legend(["f(Stopping)", "f(Through-Going)", "f(All)"], loc="best",
             fontsize=12)
    
ax[1].set_xlabel("Attribute \"%s\"" % (WHICH_ATT), fontsize=12)
ax[0].set_ylabel("Density Estimation", fontsize=12)
ax[1].plot(t, 1013 * K1 / (1013 * K1 + 1141 * K2), "k")
ax[1].set_ylabel("f(Stopping) / f(All)", fontsize=12)
fig.subplots_adjust(hspace=0)

plt.setp(ax[0].xaxis.get_ticklines(), 'markersize', 4,
         'markeredgewidth', 1)
plt.setp(ax[0].yaxis.get_ticklines(), 'markersize', 4,
         'markeredgewidth', 1)
plt.setp(ax[0].xaxis.get_minorticklines(), 'markersize', 2,
             'markeredgewidth', 1)
plt.setp(ax[0].yaxis.get_minorticklines(), 'markersize', 2,
             'markeredgewidth', 1)
ax[0].tick_params(axis='both', which='major', labelsize=12)

plt.setp(ax[1].xaxis.get_ticklines(), 'markersize', 4,
         'markeredgewidth', 1)
plt.setp(ax[1].yaxis.get_ticklines(), 'markersize', 4,
         'markeredgewidth', 1)
plt.setp(ax[1].xaxis.get_minorticklines(), 'markersize', 2,
             'markeredgewidth', 1)
plt.setp(ax[1].yaxis.get_minorticklines(), 'markersize', 2,
             'markeredgewidth', 1)
ax[1].tick_params(axis='both', which='major', labelsize=12)
plt.savefig("separationPlots/separationPlot_%s.pdf" % WHICH_ATT)
plt.clf()
