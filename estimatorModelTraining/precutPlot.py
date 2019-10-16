#encoding=utf-8
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rcParams
from matplotlib.ticker import MultipleLocator
from plotSetup import setupRcParams
from dataMethods import load_data
from plottingMethods import histogram_cv

# Constants
C_GRAY = "#E1E2E4"
C_GREEN = "#84B919"
C_GREEN_2 = "#428020"

CUT_RSTOP = 400.0
CUT_QRATIO = 0.6
CUT_ZENITH = 1.5184364492

WITH_STOP = True
WITH_DC = True
WITH_CUTS = True

PATH_TO_DATA = ["../level4_11058_27849.hdf5",
				"../level4_11057_988.hdf5",
                "../level4_11499_97852.hdf5"]

# Load data and put them into numpy arrays for easier handling
df_lbl, df_att, freq, groups = load_data(PATH_TO_DATA)

rstop = df_att["stop_point_r_HVInIcePulses"].as_matrix()
zstop = df_att["stop_point_z_HVInIcePulses"].as_matrix()
qratio = (df_att["charge_v1_HVInIcePulses_all_out"] / df_att["charge_v0_HVInIcePulses_all_out"]).as_matrix()
zenith = df_att["zenith_SplineMPE"].as_matrix()

true_rstop = df_lbl["true_stop_r"].as_matrix()
true_zstop = df_lbl["true_stop_z"].as_matrix()
label_stop = df_lbl["label_det"].as_matrix() == 1
label_stop_dc = (df_lbl["label_det"].as_matrix() == 1) & (true_rstop != 0.0) & (true_rstop < 200.0) & (true_zstop + 100.0 < 300.0)

# Statistics of the cut
cut_events = (rstop < CUT_RSTOP) & (qratio < CUT_QRATIO) & (zenith < CUT_ZENITH)
print("Before cut: %f events, %f stopping, %f stopping in dc." % (
	  np.sum(freq), np.sum(freq[label_stop]), np.sum(freq[label_stop_dc])))
print("After cut: %f events, %f stopping, %f stopping in dc." % (np.sum(freq[cut_events]), np.sum(freq[label_stop & cut_events]), np.sum(freq[label_stop_dc & cut_events])))
print("%.2f%% of all events thrown away." % (1.0 - np.sum(freq[cut_events]) / np.sum(freq)))
print("%.2f%% of all stopping events thrown away." % (1.0 - np.sum(freq[label_stop & cut_events]) / np.sum(freq[label_stop])))
print("%.2f%% of all dc stopping events thrown away." % (1.0 - np.sum(freq[label_stop_dc & cut_events]) / np.sum(freq[label_stop_dc])))

# Plot setup
setupRcParams(rcParams)

fig, ax = plt.subplots(3, figsize=(8, 12))
ax[0].set_xlabel("Stopping Point r / m")
ax[1].set_xlabel("Shell Charge Ratio")
ax[2].set_xlabel("Zenith Angle (SplineMPE)")
ax[0].set_ylabel("Frequency / Hz")
ax[1].set_ylabel("Frequency / Hz")
ax[2].set_ylabel("Frequency / Hz")

# Fixing Bins
bins_rstop = np.linspace(0.0, 600.0, 100)
bins_qratio = np.linspace(0.0, 1.0, 100)
bins_zenith = np.linspace(0.0, np.pi, 100)
bins_zstop = np.linspace(-600.0, 600.0, 100)

# Plot for rstop
rst_c, rst_c_std, rst_c_min, rst_c_max = histogram_cv(rstop, 
	                                                  bins_rstop,
	                                                  weights=freq)
rst_s, rst_s_std, rst_s_min, rst_s_max = histogram_cv(rstop[label_stop],
	                                                  bins_rstop,
	                                                  weights=freq)
rst_d, rst_d_std, rst_d_min, rst_d_max = histogram_cv(rstop[label_stop_dc],
	                                                  bins_rstop,
	                                                  weights=freq)
qrt_c, qrt_c_std, qrt_c_min, qrt_c_max = histogram_cv(qratio[label_stop], 
	                                                  bins_qratio,
	                                                  weights=freq)
qrt_s, qrt_s_std, qrt_s_min, qrt_s_max = histogram_cv(qratio[label_stop_dc], 
	                                                  bins_qratio,
	                                                  weights=freq)
qrt_d, qrt_d_std, qrt_d_min, qrt_d_max = histogram_cv(qratio, 
	                                                  bins_qratio,
	                                                  weights=freq)
znt_c, znt_c_std, znt_c_min, znt_c_max = histogram_cv(zenith, 
	                                                  bins_zenith,
	                                                  weights=freq)
znt_s, znt_s_std, znt_s_min, znt_s_max = histogram_cv(zenith[label_stop], 
	                                                  bins_zenith,
	                                                  weights=freq)
znt_d, znt_d_std, znt_d_min, znt_d_max = histogram_cv(zenith[label_stop_dc], 
	                                                  bins_zenith,
	                                                  weights=freq)
zst_c, zst_c_std, zst_c_min, zst_c_max = histogram_cv(zstop, 
	                                                  bins_zstop,
	                                                  weights=freq)
zst_s, zst_s_std, zst_s_min, zst_s_max = histogram_cv(zstop[label_stop], 
	                                                  bins_zstop,
	                                                  weights=freq)
zst_d, zst_d_std, zst_d_min, zst_d_max = histogram_cv(zstop[label_stop_dc], 
	                                                  bins_zstop,
	                                                  weights=freq)

binmid_rstop = (bins_rstop[1:] + bins_rstop[:-1]) / 2.0
ax[0].hist(rstop, bins=bins_rstop, histtype="stepfilled", facecolor="#E1E2E3",       label="All Events", weights=freq)
ax[0].errorbar(binmid_rstop, rst_c, rst_c_std,
              linestyle="", markeredgewidth=0, color="k")

if WITH_STOP is True:
	ax[0].hist(rstop[label_stop], bins=bins_rstop, histtype="stepfilled",
	           facecolor=C_GREEN, label="Stopping Events", edgecolor="k",
	           weights=freq[label_stop])
	ax[0].errorbar(binmid_rstop, rst_s, rst_s_std,
                   linestyle="", markeredgewidth=0, color="k")
if WITH_DC is True:
	ax[0].hist(rstop[label_stop_dc], bins=bins_rstop, histtype="stepfilled",
	           facecolor=C_GREEN_2, edgecolor="k", weights=freq[label_stop_dc],
	           label="Stopping Events DC")
	ax[0].errorbar(binmid_rstop, rst_d, rst_d_std,
                   linestyle="", markeredgewidth=0, color="k")
if WITH_CUTS is True:
	ax[0].axvline(CUT_RSTOP, color="k", linestyle="--", zorder=3)
	ax[0].fill_between([CUT_RSTOP, 600.0], [0, 0], [1e1, 1e1], color="w", alpha=0.75, zorder=2)
	ax[0].text(CUT_RSTOP * 1.03, 1e-2, "%.0f m" % CUT_RSTOP, rotation=90)
ax[0].legend(loc="upper right", frameon=False)
ax[0].set_yscale("log")
ax[0].set_ylim([1e-5, 1.0])
ax[0].xaxis.set_minor_locator(MultipleLocator(25))

# Plot for Qratio
binmid_qratio = (bins_qratio[1:] + bins_qratio[:-1]) / 2.0
ax[1].hist(qratio, bins=bins_qratio, histtype="stepfilled",
	       facecolor="#E1E2E3", label="All Events", weights=freq)
ax[1].errorbar(binmid_qratio, qrt_c, qrt_c_std,
               linestyle="", markeredgewidth=0, color="k")
if WITH_STOP is True:
	ax[1].hist(qratio[label_stop], bins=bins_qratio, histtype="stepfilled",
           	   facecolor=C_GREEN, label="Stopping Events", edgecolor="k",
           	   weights=freq[label_stop])
	ax[1].errorbar(binmid_qratio, qrt_s, qrt_s_std,
                   linestyle="", markeredgewidth=0, color="k")
if WITH_DC is True:
	ax[1].hist(qratio[label_stop_dc], bins=bins_qratio, histtype="stepfilled",
	           facecolor=C_GREEN_2, edgecolor="k", weights=freq[label_stop_dc],
	           label="Stopping Events DC")
	ax[1].errorbar(binmid_qratio, qrt_d, qrt_d_std,
                   linestyle="", markeredgewidth=0, color="k")
if WITH_CUTS is True:
	ax[1].axvline(CUT_QRATIO, color="k", linestyle="--", zorder=3)
	ax[1].fill_between([CUT_QRATIO, 1.0], [0, 0], [1e1, 1e1], color="w", alpha=0.75, zorder=2)
	ax[1].text(CUT_QRATIO * 1.03, 1e-2, "%.2f" % CUT_QRATIO, rotation=90)
ax[1].legend(loc="upper right", frameon=False)
ax[1].set_yscale("log")
ax[1].xaxis.set_minor_locator(MultipleLocator(0.1))

# Plot for Zeniths
binmid_zenith = (bins_zenith[1:] + bins_zenith[:-1]) / 2.0
ax[2].hist(zenith, bins=bins_zenith, histtype="stepfilled", facecolor="#E1E2E3", label="All Events", weights=freq)
ax[2].errorbar(binmid_zenith, znt_c, znt_c_std,
               linestyle="", markeredgewidth=0, color="k")
if WITH_STOP is True:
	ax[2].hist(zenith[label_stop], bins=bins_zenith, histtype="stepfilled",
	           facecolor=C_GREEN, label="Stopping Events", edgecolor="k",
	           weights=freq[label_stop])
	ax[2].errorbar(binmid_zenith, znt_s, znt_s_std,
                   linestyle="", markeredgewidth=0, color="k")
if WITH_DC is True:
	ax[2].hist(zenith[label_stop_dc], bins=bins_zenith, histtype="stepfilled",
	           facecolor=C_GREEN_2, edgecolor="k", weights=freq[label_stop_dc],
	           label="Stopping Events DC")
	ax[2].errorbar(binmid_zenith, znt_d, znt_d_std,
                   linestyle="", markeredgewidth=0, color="k")
if WITH_CUTS is True:
	ax[2].axvline(CUT_ZENITH, color="k", linestyle="--", zorder=3)
	ax[2].fill_between([CUT_ZENITH, np.pi], [0, 0], [1e1, 1e1], color="w", alpha=0.75, zorder=2)
	ax[2].text(CUT_ZENITH * 1.03, 1e-2, u"%.0f°" % (180.0 * CUT_ZENITH / np.pi),
	           rotation=90)
ax[2].set_xlim([0.0, np.pi])
ax[2].set_xticks([0.0, np.pi / 4.0, np.pi / 2.0, 3.0 * np.pi / 4.0, np.pi])
ax[2].set_xticklabels(["0", "pi/4", "pi/2", "3pi/4", "pi"])
ax[2].legend(loc="upper right", frameon=False)
ax[2].set_yscale("log")


plt.savefig("precut_plot1.pdf")