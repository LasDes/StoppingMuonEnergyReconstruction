#encoding=utf-8
import numpy as np
from matplotlib import pyplot as plt
from plotSetup import setupRcParams, COLORS as C
from matplotlib import rcParams
from dataMethods import load_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import pandas as pd
import randomForestReweighter
from randomForestReweighter import calculate_reweighting_factors, calculate_mismatch, chi2_test, proper_sampling

def make_comparison(attribute, mc_att, dt_att, rw_factors_, mc_wgt_,
                    logx=False, logy=False):
    setupRcParams(rcParams)
    fig, ax = plt.subplots(2, sharex=True, figsize=(6, 8))
    plt.subplots_adjust(hspace=0.05)
    chosen_att_dt = dt_att[attribute]
    chosen_att_mc = mc_att[attribute]
    if logx is True:
        min_val = np.min(chosen_att_dt)
        if min_val <= 0.0:
            min_val = 1e-10
        bins = np.logspace(np.log10(min_val), np.log10(np.max(chosen_att_dt)),
        	               100)
    else:
        bins = np.linspace(np.min(chosen_att_dt), np.max(chosen_att_dt), 100)
    h1, b, _ = ax[0].hist(chosen_att_dt, bins=bins, histtype="stepfilled",
                        normed=True, label="Data", facecolor=C["g_light"], linewidth=0)
    h2, b, _ = ax[0].hist(chosen_att_mc, bins=bins, histtype="step",
                        normed=True, label="MC (not reweighted)", weights=mc_wgt_,
                        color=C["g_dark"])
    h1_2, b, _ = ax[1].hist(dt_att[attribute], bins=bins, histtype="stepfilled",
                        normed=True, label="Data", facecolor=C["r_light"], linewidth=0)
    h2_2, b, _ = ax[1].hist(mc_att[attribute], bins=bins, histtype="step",
                        normed=True, label="MC (reweighted)", weights=rw_factors_,
                        color=C["r_dark"])
    if logx is True:
        ax[0].set_xscale("log")
        ax[1].set_xscale("log")
    if logy is True:
        H_limcheck_1 = np.concatenate((h1, h2))
        H_limcheck_2 = np.concatenate((h1_2, h2_2))
        ax[0].set_ylim([np.min(H_limcheck_1[H_limcheck_1 != 0]), np.max(H_limcheck_1)])
        ax[1].set_ylim([np.min(H_limcheck_2[H_limcheck_2 != 0]), np.max(H_limcheck_2)])
        ax[0].set_yscale("log")
        ax[1].set_yscale("log")
    #plt.fill_between((b[1:] + b[:-1]) / 2.0, h1, h2, interpolate=None, step="mid", linewidth=0, alpha=0.2)
    ax[0].legend(loc="upper left")
    ax[1].legend(loc="upper left")

    plt.xlabel(attribute)
    plt.ylabel("Frequency (Normalized)")
    ax[0].set_ylabel("Frequency (Normalized)")
    plt.savefig("./comp_plots/comp_%s.pdf" % attribute)

def auc_score_plot(filename, mc_att, dt_att, mc_wgt, rw_factors, 
	               fraction=0.01):
    choice_mc = np.random.permutation(len(mc_att))[:int(fraction * len(mc_att))]
    choice_dt = np.random.permutation(len(dt_att))[:int(fraction * len(dt_att))]
    auc_wo, fpr_wo, tpr_wo, _ = calculate_mismatch(mc_att[choice_mc],
                                                   dt_att[choice_dt],
                                                   mc_wgt[choice_mc])
    auc_rw, fpr_rw, tpr_rw, _ = calculate_mismatch(mc_att[choice_mc],
                                                   dt_att[choice_dt],
                                                   rw_factors[choice_mc])
    new_fpr = np.linspace(0.0, 1.0, 100)
    tpr_wo_ip = np.zeros((100, 5))
    tpr_rw_ip = np.zeros((100, 5))

    for i, fpr, tpr in zip(range(5), fpr_wo, tpr_wo):
        tpr_wo_ip[:, i] = np.interp(new_fpr, fpr, tpr)
    for i, fpr, tpr in zip(range(5), fpr_rw, tpr_rw):
        tpr_rw_ip[:, i] = np.interp(new_fpr, fpr, tpr)

    tpr_wo_min = np.min(tpr_wo_ip - new_fpr.reshape(-1, 1), axis=1) + new_fpr
    tpr_wo_max = np.max(tpr_wo_ip - new_fpr.reshape(-1, 1), axis=1) + new_fpr
    tpr_rw_min = np.min(tpr_rw_ip - new_fpr.reshape(-1, 1), axis=1) + new_fpr
    tpr_rw_max = np.max(tpr_rw_ip - new_fpr.reshape(-1, 1), axis=1) + new_fpr

    setupRcParams(rcParams)
    plt.figure(figsize=(8, 8))
    plt.plot(new_fpr, np.mean(tpr_wo_ip, axis=1), color=C["r"],
             label=u"No Reweighting, AUC=%.3f ± %.3f" % (np.mean(auc_wo),
             np.std(auc_wo)))
    plt.fill_between(new_fpr, tpr_wo_min, tpr_wo_max, facecolor=C["r_light"], linewidth=0, alpha=0.5)
    plt.plot(new_fpr, np.mean(tpr_rw_ip, axis=1), color=C["g"],
             label=u"With Reweighting, AUC=%.3f ± %.3f" % (np.mean(auc_rw), np.std(auc_rw)))
    plt.fill_between(new_fpr, tpr_rw_min, tpr_rw_max, facecolor=C["g_light"], linewidth=0, alpha=0.5)
    plt.plot([0.0, 1.0], [0.0, 1.0], "k--", label="Random Guess")
    plt.legend(loc="best", frameon=False)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.savefig(filename)

print("Loading MC data...")
mc_lbl, mc_att, mc_wgt, mc_grp = load_data(["../level4_11057_5234.hdf5",
                                            "../level4_11058_27849.hdf5",
                                            "../level4_11499_97852.hdf5"])
print("Loading real data...")
dt_lbl, dt_att, dt_wgt, dt_grp = load_data("../level4_data_01_burn.hdf5",
                                           exp=True)
print("Preparing data...")
mc_att = mc_att.drop(["Run", "Event", "SubEvent", "SubEventStream", "exists"],
                     axis=1)
dt_att = dt_att.drop(["Run", "Event", "SubEvent", "SubEventStream", "exists"],
                     axis=1)
print("Calculating reweighting factors...")

choice_mc = np.random.permutation(len(mc_att))[:10000]
choice_dt = np.random.permutation(len(dt_att))[:10000]
chosen_mc = mc_att.iloc[choice_mc]
chosen_dt = dt_att.iloc[choice_dt]
#rw_factors = np.load("./reweights_40.npy")
rw_factors = mc_wgt[choice_mc]
rw_factors = calculate_reweighting_factors(mc_att.iloc[choice_mc].as_matrix(), 
                                           dt_att.iloc[choice_dt].as_matrix(),
                                           sample=1.0,
                                           n_iterations=50,
                                           reg_power=1.0,
                                           weights_orig=rw_factors)

print("Dumping reweights.")
np.save("reweights.npy", rw_factors)

print("Making comparison plots...")
for att in ["zenith_SplineMPE", 
            "azimuth_SplineMPE",
            "charge_cog_r_HVInIcePulses",
            "time_frame_InIcePulses",
            "proj_frame_HVInIcePulses",
            "freelength_HVInIcePulses",
            "sphericalness_InIcePulses",
            "convhull_vol_InIcePulses",
            "min_pulse_time_InIcePulses"]:
	make_comparison(att, chosen_mc, chosen_dt, rw_factors, mc_wgt[choice_mc])
print("Making ROC plot...")

mask = proper_sampling(rw_factors, sample=100.0)

print(len(mask))

auc_score_plot("auc_comparison.pdf", chosen_mc.as_matrix(),
	           chosen_dt.as_matrix(), mc_wgt[choice_mc],
	           np.ones(len(chosen_dt)), fraction=1.0)