import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rcParams
from matplotlib.colors import LogNorm, PowerNorm
from dataMethods import load_data
from physicalConstantsMethods import in_ice_range
import pandas as pd

# Load data
dt_lbl, dt_att, weights, g = load_data(["../level4_11058_27849.hdf5",
                                     "../level4_11057_988.hdf5",
                                     "../level4_11499_97852.hdf5"])
scores = np.load("./models/model_29_12_2016.00_42/scores/scores_S.npy")[1]
depth_rfr = np.load("./models/model_29_12_2016.00_42/scores/scores_R.npy")[1]
stop_z_t = np.load("./models/model_29_12_2016.00_42/scores/scores_R.npy")[2]

print(len(dt_lbl))
print(len(scores))

# Prepare data
zenith = dt_att["zenith_SplineMPE"].as_matrix()
stop_z = dt_att["stop_point_z_HVInIcePulses"].as_matrix()
stop_r = dt_att["stop_point_r_HVInIcePulses"].as_matrix()
charge_all = dt_att["charge_v0_HVInIcePulses_all_out"].as_matrix()
charge_shell = dt_att["charge_v1_HVInIcePulses_all_out"].as_matrix()
qratio = charge_shell / charge_all
zenith_t = dt_lbl["zenith_true"].as_matrix()
stopping = dt_lbl["label_in"].as_matrix() == 1
stopping_dc = dt_lbl["label_det"].as_matrix()
cog_z = dt_att["cog_z_HVInIcePulses"].as_matrix()
cog_r = dt_att["cog_HVInIcePulses"].as_matrix()
exit_z = dt_att["exit_z_SplineMPE"].as_matrix()

only_downgoing = (zenith < 87.0 / 180.0 * np.pi)

ranges = in_ice_range(zenith[only_downgoing], 1950.0 - stop_z[only_downgoing])
ranges_rfr = in_ice_range(zenith[only_downgoing & stopping], 1950.0 - depth_rfr[only_downgoing[stopping]])
ranges_t = in_ice_range(zenith_t[only_downgoing & stopping],
                        1950.0 - stop_z_t[only_downgoing[stopping]])

rcParams["font.family"] = "Akkurat"
rcParams["xtick.major.size"] = 6
rcParams["ytick.major.size"] = 6
rcParams["xtick.major.width"] = 1
rcParams["ytick.major.width"] = 1
rcParams["xtick.minor.size"] = 3
rcParams["ytick.minor.size"] = 3
rcParams["xtick.minor.width"] = 1
rcParams["ytick.minor.width"] = 1
rcParams["xtick.direction"] = "out"
rcParams["ytick.direction"] = "out"
rcParams["legend.fontsize"] = "medium"


# True Zenith/SplineMPE Zenith Comparison
plt.figure(figsize=(8, 8))
plt.hist2d(np.cos(zenith_t), np.cos(zenith), 100, cmap="viridis",
           range=[[0.0, 1.0], [0.0, 1.0]], norm=PowerNorm(0.3))
plt.plot([0.0, 1.0], [0.0, 1.0], "w:")
plt.ylabel("cos(Zenith) (SplineMPE)")
plt.xlabel("cos(Zenith) (MC Truth)")
plt.savefig("coszenith_comp.pdf")


print len(stop_z_t[only_downgoing[stopping]])
print len(stop_z[stopping[only_downgoing]])
# True Stopping Depth/HVInIcePulses Stopping Depth Comparison
plt.figure(figsize=(8, 8))
plt.hist2d(1950.0 - stop_z_t[only_downgoing[stopping]], 
           1950.0 - stop_z[stopping[only_downgoing]],
           100, cmap="viridis")
plt.plot([1400.0, 2600.0], [1400.0, 2600.0], "w:")
plt.fill_between([2015.0, 2095.0], [1400.0, 1400.0],
                 [2600.0, 2600.0], color="w", alpha=0.2,
                 linewidth=0)
plt.text(2110.0, 2400.0, "Dust Layer", color="w", rotation=90)
plt.ylabel("Stopping Depth / m (Estimate)")
plt.xlabel("Stopping Depth / m (MC Truth)")
plt.savefig("depth_comp.pdf")


print len(ranges_t)
print len(ranges[only_downgoing & stopping])
# True Ranges/Reconstructed Ranges Comparison
plt.figure(figsize=(8, 8))
plt.hist2d(np.log10(ranges_t), 
           np.log10(ranges[only_downgoing & stopping]),
           100, cmap="viridis",
           range=[[3.0, 5.0], [3.0, 5.0]],
           norm=PowerNorm(0.3))
plt.plot([3.0, 5.0], [3.0, 5.0], "w:")
plt.ylabel("log(Muon Range / m) (Estimate)")
plt.xlabel("log(Muon Range / m) (MC Truth)")
plt.savefig("range_comp.pdf")

# True Ranges/Reconstructed Depths via Regression Comparison
sel = (zenith < 87.0 / 180.0 * np.pi) & (qratio < 0.5) & (stop_r < 350.0) & (stopping == 1)
print(depth_rfr.shape)
print(np.sum(sel))
plt.figure(figsize=(8, 8))
plt.hist2d(1950.0 - stop_z_t[only_downgoing[stopping]], 
           1950.0 - depth_rfr[only_downgoing[stopping]],
           100, cmap="viridis")
plt.plot([1400.0, 2600.0], [1400.0, 2600.0], "w:")
plt.ylabel("Stopping Depth / m (Random Forest Regression)")
plt.xlabel("Stopping Depth / m (MC Truth)")
plt.savefig("rfr_depth_comp.pdf")

plt.figure(figsize=(8, 8))
plt.hist2d(np.log10(ranges_t[stopping[only_downgoing]]), 
           np.log10(ranges_rfr),
           100, cmap="viridis",
           range=[[3.0, 5.0], [3.0, 5.0]],
           norm=PowerNorm(0.3))
plt.plot([3.0, 5.0], [3.0, 5.0], "w:")
plt.ylabel("log(Muon Range / m) (Random Forest Regression)")
plt.xlabel("log(Muon Range / m) (MC Truth)")
plt.savefig("range_rfr_comp.pdf")

# Naive Score/Stop Radius
sel = (stopping == True)

plt.figure(figsize=(8, 8))
plt.hist2d(scores[sel], 
           stop_r[sel],
           101, cmap="viridis", norm=PowerNorm(0.5))
plt.ylabel("Stopping Position R / m")
plt.xlabel("Random Forest Score (Classification on Full Data Set)")
plt.savefig("score_rstop_comp.pdf")

# Naive Score/Stop z
sel = (stopping == True)

plt.figure(figsize=(8, 8))
plt.hist2d(scores[sel], 
           stop_z[sel],
           101, cmap="viridis", norm=PowerNorm(0.5))
plt.ylabel("Stopping Position Z / m")
plt.xlabel("Random Forest Score (Classification on Full Data Set)")
plt.savefig("score_zstop_comp.pdf")

# Naive Score/Qratio
sel = (stopping == True)

plt.figure(figsize=(8, 8))
plt.hist2d(scores[sel], 
           qratio[sel],
           101, cmap="viridis", norm=PowerNorm(0.3))
plt.ylabel("Shell Charge Ratio")
plt.xlabel("Random Forest Score (Classification on Full Data Set)")
plt.savefig("score_qratio_comp.pdf")