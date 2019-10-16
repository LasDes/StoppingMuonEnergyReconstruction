# # # # # # # # # # # # # #
# Methods for Pipelining  #
# # # # # # # # # # # # # #
# 
# Methods for creating sklearn pipelines and applying/validating them.
# S_pipeline: Pipeline for Stopping/Through-going classification.
# Q_pipeline: Pipeline for Quality classification.
# M_pipeline: Pipeline for Multiplicitiy estimation.
#
# 2016 T. Hoinka (tobias.hoinka@udo.edu)

import numpy as np
import sys
from crossValidationMethods import group_cv_slices
from featureSelectionMethods import ad_hoc_feature_selection, rm_low_var, rm_weaker_correlated_features
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier, RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE, RFECV, f_regression, SelectFromModel, chi2, mutual_info_regression
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from plottingMethods import validation_plots_classif, validation_R_plots
from configParser import prs_i, prs_f
from sklearn.externals import joblib
from sklearn.utils import check_array
from sklearn.base import BaseEstimator

def pipeline_from_config(cf, data, label, feat_mask):
    if cf["classifier"] == "RandomForestClassifier":
        classifier = RandomForestClassifier()
    elif cf["classifier"] == "GradientBoostingClassifier":
        classifier = GradientBoostingClassifier()
    elif cf["classifier"] == "AdaBoostClassifier":
        classifier = AdaBoostClassifier()
    elif cf["classifier"] == "ExtraTreesClassifier":
        classifier = ExtraTreesClassifier()
    elif cf["classifier"] == "RandomForestRegressor":
        classifier = RandomForestRegressor()
    elif cf["classifier"] == "GradientBoostingRegressor":
        classifier = GradientBoostingRegressor()
    elif cf["classifier"] == "AdaBoostRegressor":
        classifier = AdaBoostRegressor()
    elif cf["classifier"] == "ExtraTreesRegressor":
        classifier = ExtraTreesRegressor()
    else:
        sys.exit("Classifier \"%s\" not supported." % cf["classifier"])
    params = {}
    if cf["n_estimators"] != "None":
        params["n_estimators"] = int(cf["n_estimators"])
    if cf["max_depth"] != "None":
        params["max_depth"] = int(cf["max_depth"])
    if cf["max_features"] != "None":
        params["max_features"] = int(cf["max_features"])
    if cf["min_samples_split"] != "None":
        params["min_samples_split"] = int(cf["min_samples_split"])
    if cf["max_leaf_nodes"] != "None":
        params["max_leaf_nodes"] = int(cf["max_leaf_nodes"])
    if cf["min_impurity_split"] != "None":
        params["min_impurity_split"] = int(cf["min_impurity_split"])
    if cf["n_jobs"] != "None":
        params["n_jobs"] = int(cf["n_jobs"])
    classifier.set_params(**params)
    if int(cf["k"]) == -1:
        k = "all"
    else:
        k = int(cf["k"])
    if cf["feature_selection"][:9] == "from_file":
        f = np.load(cf["feature_selection"][10:])
        feature_selection = f
    elif cf["feature_selection"] == "f_classif":
        f = f_classif(data, label)[0]
    elif cf["feature_selection"] == "mutual_info_classif":
        f = mutual_info_classif(data, label)[0]
    elif cf["feature_selection"] == "f_regression":
        f = f_regression(data, label)[0]
    elif cf["feature_selection"] == "mutual_info_regression":
        f = mutual_info_regression(data, label)[0]
    elif cf["feature_selection"] == "chi2":
        f = chi2(data, label)[0]

    f *= feat_mask
    feature_selection = ad_hoc_feature_selection(f, k)
    return feature_selection, classifier

"""Validation routine for the s pipeline.
Parameters
----------
data : array, shape = [N_samples, N_attributes]
       Attributes for classification.

label : array, shape = [N_samples,]
        Labels, should be boolean or float.

weights : array, shape = [N_samples,]
          Sample weights.

groups : array, shape = [N_samples,]
         Group associations.

config_dict : dict
              Dictionary containing config parameters.

Returns
-------
rf_scores : array, shape = [N_samples,]
            Cross validated scores.
"""
def val_pipeline(data, label, weights, groups, config_dict, key, folder,
                 feat_mask, selection=None):
    if selection is None:
        selection = np.ones(len(data), dtype=bool)
    weights_sel = weights[selection]
    groups_sel = groups[selection]
    n_crossval = int(config_dict[key]["n_crossval"])
    grp_lbl = groups_sel * 100 + label
    cv_slices = group_cv_slices(label, groups_sel, n_slices=n_crossval)
    if key != "pipeline_r":
        n_classes = len(set(label))
        rf_scores = np.zeros((len(data), n_classes))
    else:
        rf_scores = np.zeros(len(data))
    label_top = np.zeros(len(data))
    label_top[selection] = label
    cv_cnt = 0
    rf_scores_sel = rf_scores[selection]
    for learn_i, test_i in cv_slices:
        fs, cf = pipeline_from_config(config_dict[key],
                                      data[selection, :][learn_i],
                                      label[learn_i], feat_mask)
        cf.fit(fs.transform(data[selection, :][learn_i, :]),
                            label[learn_i],
                            sample_weight=weights_sel[learn_i])
        if (config_dict["output"]["save_cv"] == "True") is True:
            joblib.dump(Pipeline([("feature_selection", fs), 
                                  ("classifier", cf)]),
                        folder + "/pipelines/%s_cv_%i.pickle" % (key, cv_cnt),
                        compress=True)
        try:
            rf_scores_sel[test_i] = cf.predict_proba(fs.transform(data[selection, :][test_i, :]))
        except:
            rf_scores_sel[test_i] = cf.predict(fs.transform(data[selection, :][test_i, :]))
        cv_cnt += 1
        print("Finished %s cv split %i." % (key, cv_cnt))
    print("Finished cross validation for %s." % key)
    if (config_dict["output"]["make_final"] == "True") is True:
        fs, cf = pipeline_from_config(config_dict[key], data[selection, :],
                                      label, feat_mask)
        cf.fit(fs.transform(data[selection, :]), label, sample_weight=weights_sel)
        if np.sum(~selection) > 0:
            try:
                rf_scores[~selection] = cf.predict_proba(fs.transform(data[~selection, :]))
                print(rf_scores[~selection])
            except:
                rf_scores[~selection] = cf.predict(fs.transform(data[~selection, :]))
        rf_scores[selection] = rf_scores_sel
        joblib.dump(Pipeline([("feature_selection", fs), 
                                  ("classifier", cf)]),
                    folder + "/pipelines/%s.pickle" % key,
                    compress=True)
    if (config_dict["output"]["save_scores"] == "True") is True:
        np.vstack((rf_scores.T,
                   label_top.T,
                   weights.T)).T.dump(folder + "/scores/%s.npy" % key)
    validation_plots_classif(rf_scores[selection], label,
                             data[selection, :], 
                             weights_sel, cv_slices, folder, key,
                             config_dict[key]["plots"])
    print("Finished %s." % key)

    return rf_scores