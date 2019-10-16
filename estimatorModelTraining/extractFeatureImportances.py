"""Extract Feature Importances.
Usage: extractFeatureImportances.py MODEL DATA [--validate --kind=<x> --output=<path>]

-h               Show this.
MODEL            Path to model directory
DATA             Some data for feature names
--validate       Validate Feature Selection (Robustness etc)
--kind=<x>       Classification (either s, q, m or r)
--output=<path>  Output path [default: .]
"""
# # # # # # # # # # # # # # # #
# Extract Feature Importances #
# # # # # # # # # # # # # # # #
# 
# This Script is supposed to extract the feature importances of a
# finished model. Excluded feature importances are set to zero.
#
# 2017 T. Hoinka (tobias.hoinka@udo.edu)

import numpy as np
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier, RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, AdaBoostRegressor
from dataMethods import load_data
from sklearn.feature_selection import SelectKBest
from sklearn.externals import joblib
from docopt import docopt
from plotSetup import setupRcParams, COLORS as C
from matplotlib import rcParams

def jaccard(set1, set2):
    return float(len(set1.intersection(set2))) / float(len(set1.union(set2)))

def robustness(fs, k):
    J = []
    L = np.shape(fs)[1]
    for i in range(L):
        for j in range(L):
            if i != j:
                J += [jaccard(set(fs[:k, i]), set(fs[:k, j]))]
    return np.mean(J), np.std(J), np.max(J), np.min(J)

def robustness_vector(fs, start, end):
    R = np.zeros(end - start + 1)
    R_std = np.zeros(end - start + 1, dtype=float)
    R_max = np.zeros(end - start + 1, dtype=float)
    R_min = np.zeros(end - start + 1, dtype=float)
    K = np.linspace(start, end + 1, end - start + 1)
    for i in range(len(K)):
        R[i], R_std[i], R_max[i], R_min[i] = robustness(fs, K[i])
    return K, R, R_std, R_max, R_min

def robustness_plot(importances, set_fs, filename):
    importances = np.mean(np.array(importances), axis=1)
    importances = importances[np.argsort(importances)][::-1]
    K, r_vec, r_vec_std, r_vec_max, r_vec_min = robustness_vector(set_fs, 1,
                                                                  99)
    setupRcParams(rcParams)
    #plt.bar(K - 0.25, 10.0 * importances[:99], color=C["r"], linewidth=0,
    #        width=0.5)
    plt.errorbar(K, r_vec, yerr=[r_vec - r_vec_min, r_vec_max - r_vec],
                 color=C["g"], marker="s", markersize=4,
                 markeredgewidth=0, linestyle="")
    plt.xlim([0, 100])
    plt.ylim([0.0, 1.0])
    plt.savefig(filename)
    plt.close()

def load_fi(path_to_model, c, cv=None):
    if cv is None:
        pl = joblib.load(path_to_model + "pipeline_%s.pickle" % c)
    else:
        pl = joblib.load(path_to_model + "pipeline_%s_cv_%i.pickle" % (c, cv))
        print("loaded %i" % cv)
    fs = pl.steps[0][1]
    rf = pl.steps[1][1]
    feat_imp = np.zeros_like(fs.get_support(), dtype=float)
    feat_imp[fs.get_support() == 1] = rf.feature_importances_
    return feat_imp

def extract_feature_importances(path_to_model, path_to_data, kind, output,
                                validate):
    data_names = load_data(path_to_data)[1].columns.values
    for i in range(len(data_names)):
        print("%i: %s" % (i, data_names[i]))
    for c in kind:
        cv_fi = np.zeros((297, 10))
        ranking = np.zeros((297, 10), dtype=int)
        for cv in range(10):
            cv_fi[:, cv] = load_fi(path_to_model, c, cv=cv)
            ranking[:, cv] = np.argsort(cv_fi[:, cv])[::-1]
        robustness_plot(cv_fi, ranking, "robustness_%s.pdf" % c)

        np.save(output + "/feature_importance_%s.npy" % c,
                load_fi(path_to_model, c))
        print(data_names[np.argsort(load_fi(path_to_model, c))[::-1]])

if __name__ == "__main__":
    args = docopt(__doc__, version="Extract Feature Importances")
    extract_feature_importances(args["MODEL"], args["DATA"], args["--kind"],
                                args["--output"],
                                args["--validate"])