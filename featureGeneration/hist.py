import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.stats import gaussian_kde
from matplotlib.ticker import AutoMinorLocator, MultipleLocator

from loadData import loadDataHD5

WHICH_ATT = "zenith_SplineMPE"

c = np.loadtxt("../muonClassification/plots/confi_err.csv", delimiter=",")

print(len(c))

def getByName(name, data_names, data):
    try:
        return data[:, data_names == name].flatten()
    except KeyError:
        print("Keyword not found: %s" % name)

(lbl_names, lbl, att_names, att) = loadDataHD5("../ftable_all.hdf5")

zenith_splinempe = getByName("zenith_SplineMPE", att_names, att)
zenith_true = getByName("zenith_true", lbl_names, lbl)
end_point = getByName("stop_point_z_HVInIcePulses", att_names, att)
stopping = getByName("label_det", lbl_names, lbl)

plt.hist((1950.0 - end_point[stopping == 1]) / np.cos(zenith_true[stopping == 1]), 100)
plt.show()