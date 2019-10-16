# # # # # # # # # # # # # # # #
# Methods for Plotting Stuff  #
# # # # # # # # # # # # # # # #
# 
# 2016 T. Hoinka (tobias.hoinka@udo.edu)

import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib import rcParams
from plotSetup import setupRcParams
from physicalConstantsMethods import in_ice_range
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
import os

N_CROSSVAL = 10

C_RED = "#B91959"
C_GREEN = "#84B919"

def validation_plots_classif(scores, label, data, weights, cv, folder, key, config):
    setupRcParams(rcParams, grid=True)

    destination = folder + "/plots_%s" % key
    if not os.path.isdir(destination):
        os.mkdir(destination)

    if scores.ndim > 1:
        classes = np.sort(list(set(label)))
        n_classes = len(classes)
    else:
        n_classes = 1

    for k in range(n_classes):
        if scores.ndim > 1:
            lbl = label == classes[k]
            scr = scores[:, k]
        else:
            lbl = label
            scr = scores
        if "ConfidenceHist" in config:
            confidence_histogram(destination + "/confidence_hist_%d.pdf" % k,
                                 scr, lbl, data, weights, cv)
        if "ROC" in config:
            ROC_curve(destination + "/ROC_curve_%d.pdf" % k, scr, lbl)

        if "PrecisionRecall" in config:
            precision_efficiency(destination + "/precision_recall_%d.pdf" % k,
                                 scr, lbl, data, weights, cv)
        if "DepthCorrelation" in config:
            hist_R(destination + "/depth_corr_%d.pdf" % k,
                   lbl, scr, weights)

def confidence_histogram(filename, scores, label, data, weights, cv_slices,
                         n_estimators=100):
    setupRcParams(rcParams, grid=True)
    bins = np.linspace(0.0, 1.0, n_estimators + 2)
    H_sig = np.zeros((n_estimators + 1, N_CROSSVAL))
    H_bkg = np.zeros((n_estimators + 1, N_CROSSVAL))
    i = 0
    for learn_i, test_i in cv_slices:
        H_sig[:, i] = np.histogram(scores[test_i][label[test_i]],
                                   bins=bins,
                                   weights=weights[test_i][label[test_i]])[0]
        H_bkg[:, i] = np.histogram(scores[test_i][~label[test_i]],
                                   bins=bins,
                                   weights=weights[test_i][~label[test_i]])[0]
        i += 1
    H_mean_sig = np.mean(H_sig, axis=1)
    H_mean_bkg = np.mean(H_bkg, axis=1)
    H_std_sig = np.std(H_sig, axis=1)
    H_std_bkg = np.std(H_bkg, axis=1)
    H_min_sig = np.min(H_sig, axis=1)
    H_min_bkg = np.min(H_bkg, axis=1)
    H_max_sig = np.max(H_sig, axis=1)
    H_max_bkg = np.max(H_bkg, axis=1)

    bin_mids = (bins[:-1] + bins[1:]) / 2.0

    plt.figure()
    plt.step(bin_mids, H_mean_sig, where="mid", color=C_RED, label="Signal")
    plt.errorbar(bin_mids, H_mean_sig, H_std_sig,
                 markeredgewidth=0, color=C_RED, linestyle="")

    plt.step(bin_mids, H_mean_bkg, where="mid", color=C_GREEN,
             label="Background")
    plt.errorbar(bin_mids, H_mean_bkg, H_std_bkg,
                 markeredgewidth=0, color=C_GREEN, linestyle="")
    plt.fill_between(bin_mids, H_min_sig, H_max_sig,
                     alpha=0.3, linewidth=0, color=C_RED,
                     interpolate=False, step="mid")
    plt.fill_between(bin_mids, H_min_bkg, H_max_bkg,
                     alpha=0.3, linewidth=0, color=C_GREEN,
                     interpolate=False, step="mid")
    plt.xlabel("Score")
    plt.ylabel("Counts per Bin")
    plt.yscale("log")
    plt.legend(loc="best", frameon=False)
    plt.savefig(filename)
    plt.close()

def ROC_curve(filename, scores, label):
    setupRcParams(rcParams, grid=True)
    fpr, tpr, thresholds = metrics.roc_curve(label, scores)
    plt.figure(figsize=(7, 7))
    plt.fill_between(fpr, np.zeros(len(fpr)), tpr, facecolor=C_RED,
                    edgecolor="#FFFFFF", alpha=0.2)
    plt.plot(fpr, tpr, color=C_RED, linestyle="-")
    plt.plot([0.0, 1.0], [0.0, 1.0], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(["Area: %f" % (metrics.roc_auc_score(label, scores))],
               loc="best", frameon=False)
    plt.savefig(filename)
    plt.close()

def precision_efficiency(filename, scores, label, data, weights, cv_slices,
                         n_samples=101):
    setupRcParams(rcParams, grid=True)
    cuts = np.linspace(0.0, 1.0, n_samples)
    precision = np.zeros((n_samples, N_CROSSVAL))
    efficiency = np.zeros((n_samples, N_CROSSVAL))
    cv_index = 0
    for learn_i, test_i in cv_slices:
        for i in range(len(cuts)):
            results = scores[test_i] > cuts[i]
            precision[i, cv_index] = metrics.precision_score(label[test_i],
                                                              results,
                                                              sample_weight=weights[test_i])
            efficiency[i, cv_index] = metrics.recall_score(label[test_i],
                                                        results,
                                                        sample_weight=weights[test_i])
        cv_index += 1
    median_precision = np.mean(precision, axis=1)
    std_precision = np.std(precision, axis=1)
    max_precision = np.max(precision, axis=1)
    min_precision = np.min(precision, axis=1)

    median_recall = np.mean(efficiency, axis=1)
    std_recall = np.std(precision, axis=1)
    max_recall = np.max(efficiency, axis=1)
    min_recall = np.min(efficiency, axis=1)

    plt.figure()
    plt.errorbar(cuts, median_precision, 
                 yerr=[median_precision - min_precision, 
                       max_precision - median_precision], 
                 color=C_RED, marker="s", linestyle="", markeredgewidth=0,
                 label="Precision", markersize=3)
    plt.errorbar(cuts, median_recall, 
                 yerr=[median_recall - min_recall, 
                       max_recall - median_recall], 
                 color=C_GREEN, marker="s", linestyle="", markeredgewidth=0,
                 label="Efficiency", markersize=3)
    plt.xlabel("Score")
    plt.legend(loc="best", frameon=False)
    plt.savefig(filename)
    plt.close()

def histogram_cv(data, bins, cv_folds=20, weights=None):
    setupRcParams(rcParams, grid=False)
    if weights is None:
        weights = np.ones_like(data)
    P = np.random.permutation(len(data))
    folds = np.round(np.linspace(0, len(data), cv_folds + 1))
    H = np.zeros((len(bins) - 1, cv_folds))
    for k in range(cv_folds):
        sel = P[int(folds[k]):int(folds[k + 1])]
        H[:, k] = np.histogram(data[sel],
                               bins=bins, weights=weights[sel])[0]
    return (np.sum(H, axis=1), 
            cv_folds * np.std(H, axis=1),
            cv_folds * np.min(H, axis=1),
            cv_folds * np.max(H, axis=1))

def hist_R(filename, true_R, est_R, weights):
    setupRcParams(rcParams, grid=False)
    plt.figure(figsize=(8, 8))
    plt.hist2d(true_R, est_R, 100, cmap="viridis",
               weights=weights)
    plt.xlabel("Stopping Depth / m (MC Truth)")
    plt.ylabel("Stopping Depth / m (Random Forest Regressor)")
    plt.savefig(filename)
    plt.close()

def hist_ranges(filename, ranges_t, ranges, weights):
    setupRcParams(rcParams, grid=False)
    plt.figure(figsize=(8, 8))
    plt.hist2d(np.log10(ranges_t), np.log10(ranges), 100, cmap="viridis",
               weights=weights)
    plt.xlabel("Range / m (MC Truth)")
    plt.ylabel("Range / m (Random Forest Regressor)")
    plt.savefig(filename)
    plt.close()

def validation_R_plots(filename, true_R, est_R, zeniths_t, zeniths, weights):
    sel = zeniths < 87.0 / 180.0 * np.pi
    hist_R(filename + "/depth.pdf", true_R[sel], est_R[sel], weights[sel])
    hist_ranges(filename + "/ranges.pdf",
                in_ice_range(zeniths_t[sel], 1950.0 - true_R[sel]),
                in_ice_range(zeniths[sel], 1950.0 - est_R[sel]),
                weights[sel])
