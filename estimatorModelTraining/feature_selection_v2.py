"""Feature Selection using preselected sample sets.

Usage: feature_selection_v2.py --input_mc <input_mc> --input_data <input_data> -o <output> --n_mismatch <n_mismatch> -j <n_jobs> --n_estimators <n_estimators> --corr_threshold <corr_threshold>

-h --help                           Show this.
--input_mc <input_mc>               Monte Carlo input.
--input_data <input_data>           Data input.
-o <output>                         Output.
--n_mismatch <n_mismatch>           Number of features to throw away due to mismatches.
-j <n_jobs>                         Number of jobs
--n_estimators <n_estimators>       Number of estimators to use in each step
--corr_threshold <corr_threshold>   Limit for value of spearman-correlation
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.externals import joblib
from sklearn.metrics import roc_auc_score
from docopt import docopt
from scipy.stats import spearmanr

def mismatch_selection(mc, data, n_estimators=20, n_mismatch=100, n_jobs=20,
                        corr_threshold=0.8):
    # initialize rf-classifier object
    rf = RandomForestClassifier(n_estimators=n_estimators, n_jobs=n_jobs)

    n_features = mc.shape[1]

    # merge mc-data and recorderd data together like (mc,mc,.......,data,data)
    dataset = np.concatenate((mc, data))
    # cerate set of labels for merged data like (0,0,0,0,.......,1,1,1,1,1)
    label = np.concatenate((np.zeros(len(mc)), np.ones(len(data))))

    # index list of features still in use
    feature_idx = np.arange(n_features)

    # list of all mismatched features
    rm_features = np.array([], dtype=int)

    # Fit the model only once with full set of features
    rf.fit(dataset[:, feature_idx], label)

    # rf.feature_importances_ cannot be writen on, just create a copy
    relevancy = np.copy(rf.feature_importances_)

    # list of accumulated importances
    rm_imp_acc = np.array([np.sum(relevancy)], dtype=float)

    # list of remaining features after each step
    rm_num = np.array([feature_idx.size], dtype=float)

    # TODO: remove this
    counter = 0

    while True:
        # pick index of the feature that is most relevant for separation
        # of data and mc
        best = feature_idx[np.argmax(relevancy)]

        # iterate over all remaining features and calc spearman rank-order
        # correlation with best
        spearman_corr = np.array([ spearmanr(dataset[:, best], dataset[:, i])[0]
            for i in feature_idx ])

        # replace all the nans with 0, no idea where they are coming from and
        # take absolute values
        spearman_corr = np.abs(np.nan_to_num(spearman_corr))

        # TODO: clean up reports
        counter += 1
        print "Iteration: ", counter
        print "Best Feauture: ", best
        print "Features removable: ", np.sum([ spearman_corr > corr_threshold ])
        print "--------------------------------"

        # execute removal of features strongly correlated to best with p-values
        # over threshold and best itself
        rm_features = np.append(rm_features, [feature_idx[ spearman_corr > corr_threshold ]])
        relevancy = relevancy[ spearman_corr < corr_threshold ]
        feature_idx = feature_idx[ spearman_corr < corr_threshold ]
        rm_imp_acc = np.append(rm_imp_acc, [np.sum(relevancy)])
        rm_num = np.append(rm_num, [feature_idx.size])

        # abort as soon as soon as no more features remain
        if rm_features.size == 0 :
            break

    return rm_features, rm_imp_acc, rm_num

def feature_selection_backward(mc, label, n_estimators=20, n_jobs=20,
                               reg=False, corr_threshold=0.8):
    # create rf-object, classifier or regressor depending on choice
    if reg is False:
        rf = RandomForestClassifier(n_estimators=n_estimators, n_jobs=n_jobs)
    else:
        rf = RandomForestRegressor(n_estimators=n_estimators, n_jobs=n_jobs)

    n_features = mc.shape[1]

    # index list of features still in use
    feature_idx = np.arange(n_features)

    # list of all mismatched features
    rm_features = np.array([], dtype=int)

    # list of importance for mismatched features
    rm_features_imp = np.array([], dtype=float)

    # Fit the model only once with full set of features
    rf.fit(mc[:, feature_idx], label)

    # rf.feature_importances_ cannot be writen on, just create a copy
    relevancy = np.copy(rf.feature_importances_)

    while True:
        # pick index of the feature that is most irrelevant for performance
        worst = feature_idx[np.argmin(relevancy)]

        # iterate over all remaining features and calc spearman rank-order
        # correlation with worst
        spearman_corr = np.array([ spearmanr(mc[:, worst], mc[:, i])[0]
            for i in feature_idx ])

        # replace all the nans with 0, no idea where they are coming from and
        # take absolute values
        spearman_corr = np.abs(np.nan_to_num(spearman_corr))

        # abort as soon as no other features than 'worst' to remove
        if np.sum([ spearman_corr > corr_threshold ]) < 2 :
            break

        # execute removal of features strongly correlated to worst with p-values
        # over threshold and worst itself
        rm_features = np.append(rm_features, [feature_idx[ spearman_corr > corr_threshold ]])
        rm_features_imp = np.append(rm_features_imp, [relevancy[ spearman_corr > corr_threshold ]])
        relevancy = relevancy[ spearman_corr < corr_threshold ]
        feature_idx = feature_idx[ spearman_corr < corr_threshold ]

    return rm_features, rm_features_imp

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

    print "---Begin: mismatch-selection"
    rm_feat_mm, rm_imp_acc_mm, rm_num_mm = mismatch_selection(mc_feat.values[:,mask],
                                            dt_feat.values[:,mask],
                                            n_estimators=int(args["--n_estimators"]),
                                            n_jobs=int(args["-j"]),
                                            corr_threshold=float(args["--corr_threshold"]))
    print "---Completed: mismatch-selection"

    mask_mm = np.copy(mask)
    sel_mm = rm_feat_mm[:int(args["--n_mismatch"])]
    mask_temp = np.ones(np.sum(mask), dtype=bool)
    mask_temp[sel_mm] = False
    mask_mm[mask_mm] = mask_temp
    print("Remaining Features: ", np.sum(mask_mm))
    #print(np.sum(keys[~mask_mm]))
    #print(mc_feat.values[s_lab == 1,:][:,mask_mm].shape)
    keys_masked_mm = keys[mask_mm]

    print "---Begin: feature-selection s"
    #rm_feat_s, rm_feat_imp_s = feature_selection_backward(mc_feat.values[:,mask_mm],
    #                             s_lab,
    #                             n_estimators=int(args["--n_estimators"]),
    #                             n_jobs=int(args["-j"]),
    #                             corr_threshold=float(args["--corr_threshold"]))
    print "---Completed: feature-selection s"

    print "---Begin: feature-selection q"
    #rm_feat_q, rm_feat_imp_q = feature_selection_backward(mc_feat.values[single,:][:,mask_mm],
    #                             q_lab,
    #                             n_estimators=int(args["--n_estimators"]),
    #                             n_jobs=int(args["-j"]),
    #                             corr_threshold=float(args["--corr_threshold"]))
    print "---Completed: feature-selection q"

    print "---Begin: feature-selection m"
    #rm_feat_m, rm_feat_imp_m = feature_selection_backward(mc_feat.values[s_lab == 1,:][:,mask_mm],
    #                             m_lab,
    #                             n_estimators=int(args["--n_estimators"]),
    #                             n_jobs=int(args["-j"]),
    #                             corr_threshold=float(args["--corr_threshold"]))
    print "---Completed: feature-selection m"

    print "---Begin: feature-selection r"
    #rm_feat_r, rm_feat_imp_r = feature_selection_backward(mc_feat.values[single,:][:,mask_mm],
    #                             r_lab,
    #                             reg=True,
    #                             n_estimators=int(args["--n_estimators"]),
    #                             n_jobs=int(args["-j"]),
    #                             corr_threshold=float(args["--corr_threshold"]))
    print "---Completed: feature-selection r"

    #joblib.dump((keys_masked[rm_feat_mm], rm_feat_imp_mm
    #             keys_masked_mm[rm_feat_s], rm_feat_imp_s
    #             keys_masked_mm[rm_feat_q], rm_feat_imp_q
    #             keys_masked_mm[rm_feat_m], rm_feat_imp_m
    #             keys_masked_mm[rm_feat_r], rm_feat_imp_r) args["-o"])
