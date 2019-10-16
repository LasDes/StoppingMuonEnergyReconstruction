"""Main Routine.

Usage: mainRoutine.py CONFIG

-h     Show this.
CONFIG path to config file
"""
"""
.. module:: main_routine
    :synopsis: Main routine of classification scheme.
.. moduleauthor:: Tobias Hoinka <tobias.hoinka@udo.edu>
"""

import numpy as np
from docopt import docopt

from dataMethods import load_data_beta as load_data
from physicalConstantsMethods import in_ice_range
from pipelineMethods import val_pipeline
from featureSelectionMethods import rm_low_var, rm_weaker_correlated_features
from configParser import config_parser
from randomForestReweighter import RandomForestReweighter

from sklearn.externals import joblib

import time
import os

def prepare_output_directories(directory):
    T = time.localtime()
    data_string = "%02d_%02d_%02d.%02d_%02d" % (T[2], T[1], T[0], T[3], T[4])
    folder = directory + "/models/model_" + data_string
    mkdir_check(folder)
    mkdir_check(folder + "/pipelines")
    mkdir_check(folder + "/scores")
    mkdir_check(folder + "/plots_pipeline_s")
    mkdir_check(folder + "/plots_pipeline_q")
    mkdir_check(folder + "/plots_pipeline_m")
    mkdir_check(folder + "/plots_pipeline_r")
    return folder

def get_labels(label, att):
    label_S = (label["Hoinka_Labels_label_in"].as_matrix() == 1.0)
    label_M = (label["Hoinka_Labels_n_mu_stop"].as_matrix() == 1)[label_S]
    label_R = label["Hoinka_Labels_true_stop_z"].as_matrix()[label_S]
    zenith_splinempe = att["Hoinka_zenith_SplineMPE"].as_matrix()
    zenith_true = label["Hoinka_Labels_zenith_true"].as_matrix()
    azimuth_splinempe = att["Hoinka_azimuth_SplineMPE"].as_matrix()
    azimuth_true = label["Hoinka_Labels_azimuth_true"].as_matrix()
    ang_error = np.arccos(np.cos(azimuth_true-azimuth_splinempe) * np.sin(zenith_true) * np.sin(zenith_splinempe) + np.cos(zenith_true) * np.cos(zenith_splinempe))
    label_Q = (ang_error < 0.1)[label_S]
    return label_S, label_Q, label_M, label_R

def apply_precut_mask(data, precut_conf):
    print(precut_conf)
    mask = np.ones(len(data), dtype=bool)
    for feature, values in precut_conf.iteritems():
        lower = float(values[0])
        upper = float(values[1])
        mask[(data[feature] < upper) & (data[feature] > lower)] = False
    return mask

def load_with_precut_sampling(config_dict):
    data_list = config_dict["data"]["data_list"]
    sample_fraction = config_dict["data"]["sample_fraction"]
    label, att, weights, groups = load_data(data_list, verbosity=True)
    if sample_fraction != 1.0:
        sample_mask = group_select(groups, sample_fraction)
    else:
        sample_mask = np.ones(len(label), dtype=bool)

    # Precut mask is generated if chosen.
    label = label.iloc[sample_mask]
    att = att.iloc[sample_mask]
    weights = weights[sample_mask] / float(sample_fraction)
    groups = groups[sample_mask]
    print("%i samples selected." % np.sum(sample_mask))
    print("Data read succesfully.")
    att_dim = att.shape
    lbl_dim = label.shape
    print("Dataset: %d labels, %d attributes, %d rows." % (lbl_dim[1],
                                                           att_dim[1],
                                                           att_dim[0]))
    return label, att, weights, groups

def mkdir_check(directory):
    if not os.path.isdir(directory):
        os.mkdir(directory)

def group_select(groups, fraction):
    selection = np.zeros(len(groups), dtype=bool)
    for g in set(groups):
        members = np.where(groups == g)[0]
        print(g, int(fraction * float(len(members))))
        selection[members[:int(fraction * float(len(members)))]] = True 
    return selection

#def reweighting_procedure(cf_rw, mc_att, dt_att)

def main_routine(config_dict):
    print("Loading from:")
    print(config_dict["data"]["data_list"])
    label, att, weights, groups = load_with_precut_sampling(config_dict)
    label_S, label_Q, label_M, label_R = get_labels(label, att)
    folder = prepare_output_directories(config_dict["output"]["dir"])

    n_features = np.shape(att)[1]

    var_threshold = float(config_dict["pipeline_s"]["var_threshold"])
    cov_threshold = float(config_dict["pipeline_s"]["cov_threshold"])

    f_mask = np.ones(n_features)
    f_mask *= rm_low_var(att,
                         variance_threshold=var_threshold)
    f_mask *= rm_weaker_correlated_features(att, label_S,
                                            correlation_threshold=cov_threshold)
    if (config_dict["misc"]["skip_s"] == "True") is False:
        val_pipeline(att.as_matrix(), label_S, weights, groups, config_dict,
                    "pipeline_s", folder, f_mask)
    if (config_dict["misc"]["skip_q"] == "True") is False:
        val_pipeline(att.as_matrix(), label_Q, 
                     weights, groups,
                     config_dict, "pipeline_q", folder, f_mask,
                     selection=label_S)
    if (config_dict["misc"]["skip_m"] == "True") is False:
        val_pipeline(att.as_matrix(), label_M,
                     weights, groups,
                     config_dict, "pipeline_m", folder, f_mask,
                     selection=label_S)
    if (config_dict["misc"]["skip_r"] == "True") is False:
        val_pipeline(att.as_matrix(), label_R,
                     weights, groups,
                     config_dict, "pipeline_r", folder, f_mask,
                     selection=label_S)
    print("Models dumped.")
    print("Finished. ")

if __name__ == "__main__":
    args = docopt(__doc__, version="Main Routine")
    config_dict = config_parser(args["CONFIG"])
    main_routine(config_dict)