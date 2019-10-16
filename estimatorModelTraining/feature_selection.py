"""Feature Selection using preselected sample sets.

Usage: feature_selection.py --input_mc <input_mc> --input_data <input_data> -o <output> --n_mismatch <n_mismatch> -j <n_jobs> --n_estimators <n_estimators>

-h --help                     Show this.
--input_mc <input_mc>         Monte Carlo input.
--input_data <input_data>     Data input.
-o <output>                   Output.
--n_mismatch <n_mismatch>     Number of features to throw away due to mismatches.
-j <n_jobs>                   Number of jobs
--n_estimators <n_estimators> Number of estimators to use in each step
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.externals import joblib
from sklearn.metrics import roc_auc_score
from docopt import docopt

def mismatch_selection(mc, data, n_estimators=20, n_jobs=20):
    rf = RandomForestClassifier(n_estimators=n_estimators, n_jobs=n_jobs)
    n_features = mc.shape[1]
    dataset = np.concatenate((mc, data))
    label = np.concatenate((np.zeros(len(mc)), np.ones(len(data))))
    mask = np.ones(n_features, dtype=bool)
    aucs = []
    rm_features = []
    feature_idx = np.arange(n_features)
    for i in range(n_features):
        rf.fit(dataset[::2, feature_idx], label[::2])
        aucs += [roc_auc_score(label[1::2],
            rf.predict_proba(dataset[1::2, feature_idx])[:,1])]
        best = np.argmax(rf.feature_importances_)
        rm_features += [feature_idx[best]]
        feature_idx = np.delete(feature_idx, best)
        print("%i Finished" % i)
    return aucs, rm_features

def feature_selection(mc, label, n_estimators=20, n_jobs=20, reg=False):
    if reg is False:
        rf = RandomForestClassifier(n_estimators=n_estimators, n_jobs=n_jobs)
    else:
        rf = RandomForestRegressor(n_estimators=n_estimators, n_jobs=n_jobs)
    n_features = mc.shape[1]
    mask = np.ones(n_features, dtype=bool)
    aucs = []
    rm_features = []
    feature_idx = np.arange(n_features)
    for i in range(n_features):
        rf.fit(mc[::2, feature_idx], label[::2])
        if reg is True:
            mse = np.mean((label[1::2] - rf.predict(mc[1::2, feature_idx])) ** 2)
            aucs += [mse]
        else:
            aucs += [roc_auc_score(label[1::2],
                rf.predict_proba(mc[1::2, feature_idx])[:,1])]
        worst = np.argmin(rf.feature_importances_)
        rm_features += [feature_idx[worst]]
        feature_idx = np.delete(feature_idx, worst)
        print("%i Finished" % i)
    return aucs, rm_features

def feature_selection_backward(mc, label, n_estimators=20, n_jobs=20,
                               reg=False):
    if reg is False:
        rf = RandomForestClassifier(n_estimators=n_estimators, n_jobs=n_jobs)
    else:
        rf = RandomForestRegressor(n_estimators=n_estimators, n_jobs=n_jobs)
    n_features = mc.shape[1]
    mask = np.ones(n_features, dtype=bool)
    aucs = []
    rm_features = []
    feature_idx = np.arange(n_features)
    for i in range(n_features):
        rf.fit(mc[::2, feature_idx], label[::2])
        if reg is True:
            mse = np.mean((label[1::2] - rf.predict(mc[1::2, feature_idx])) ** 2)
            aucs += [mse]
        else:
            aucs += [roc_auc_score(label[1::2],
                rf.predict_proba(mc[1::2, feature_idx])[:,1])]
        best = np.argmax(rf.feature_importances_)
        rm_features += [feature_idx[best]]
        feature_idx = np.delete(feature_idx, best)
        print("%i Finished" % i)
    return aucs, rm_features

def rm_constant(df):
    S = (np.std(df) != 0.0).values
    return S

def rm_duplicate(df):
    C = np.corrcoef(df.T) - np.eye(df.shape[1])
    mask = np.ones(df.shape[1], dtype=bool)
    for i in range(df.shape[1]):
        if mask[i] == True:
            mask[C[i,:] == 1.0] = False
    return mask

if __name__ == "__main__":
    args = docopt(__doc__)
    mc_label, mc_feat, _, _ = joblib.load(args["--input_mc"])
    dt_feat = joblib.load(args["--input_data"])
    
    #mc_label = mc_label[:1000]
    #mc_feat = mc_feat[:1000]
    #dt_feat = dt_feat[:1000]
    keys = mc_feat.columns.values

    mask = rm_constant(mc_feat)
    mask[mask] = rm_duplicate(mc_feat.values[:,mask])
    keys_masked = keys[mask]

    s_lab = mc_label.Hoinka_Labels_label_in.values

    zent = mc_label.Hoinka_Labels_zenith_true
    azit = mc_label.Hoinka_Labels_azimuth_true
    zens = mc_feat.SplineMPE_zenith
    azis = mc_feat.SplineMPE_azimuth

    ang_err = np.arccos(np.sin(zent) * np.cos(azit) * np.sin(zens) * np.cos(azis)
                        + np.sin(zent) * np.sin(azit) * np.sin(zens) * np.sin(azis)
                        + np.cos(zens) * np.cos(zent))

    q_lab = ang_err[(s_lab == 1) & (mc_label.Hoinka_Labels_n_mu_stop.values == 1)] < 0.1
    m_lab = (mc_label.Hoinka_Labels_n_mu_stop.values == 1)[s_lab == 1]
    r_lab = mc_label.Hoinka_Labels_true_stop_z.values[(s_lab == 1) & (mc_label.Hoinka_Labels_n_mu_stop.values == 1)]
 
    single = (s_lab == 1)
    single[single] = m_lab    
    
    auc_mm, rm_feat_mm = mismatch_selection(mc_feat.values[:,mask],
                                            dt_feat.values[:,mask],
                                            n_estimators=int(args["--n_estimators"]),
                                            n_jobs=int(args["-j"]))
    mask_mm = np.copy(mask)
    sel_mm = rm_feat_mm[:int(args["--n_mismatch"])]
    mask_temp = np.ones(np.sum(mask), dtype=bool)
    mask_temp[sel_mm] = False
    mask_mm[mask_mm] = mask_temp
    print(np.sum(mask_mm))
    print(np.sum(keys[~mask_mm]))
    print(mc_feat.values[s_lab == 1,:][:,mask_mm].shape)
    keys_masked_mm = keys[mask_mm]

    auc_s, rm_feat_s = feature_selection(mc_feat.values[:,mask_mm], s_lab,
                                         n_estimators=int(args["--n_estimators"]),
                                         n_jobs=int(args["-j"]))
    auc_q, rm_feat_q = feature_selection(mc_feat.values[single,:][:,mask_mm],
                                         q_lab,
                                         n_estimators=int(args["--n_estimators"]),
                                         n_jobs=int(args["-j"]))
    auc_m, rm_feat_m = feature_selection(mc_feat.values[s_lab == 1,:][:,mask_mm],
                                         m_lab,
                                         n_estimators=int(args["--n_estimators"]),
                                         n_jobs=int(args["-j"]))
    auc_r, rm_feat_r = feature_selection(mc_feat.values[single,:][:,mask_mm],
                                         r_lab,
                                         reg=True,
                                         n_estimators=int(args["--n_estimators"]),
                                         n_jobs=int(args["-j"]))

    joblib.dump((auc_mm, keys_masked[rm_feat_mm],
                 auc_s, keys_masked_mm[rm_feat_s],
                 auc_q, keys_masked_mm[rm_feat_q],
                 auc_m, keys_masked_mm[rm_feat_m],
                 auc_r, keys_masked_mm[rm_feat_r]), args["-o"])
