"""Main Routine.

Usage: mainRoutine.py CONFIG

-h     Show this.
CONFIG path to config file
"""
# # # # # # # # # # # # # # # #
# New Classification Routine  #
# # # # # # # # # # # # # # # #
#
# What this is supposed to achieve:
# a) Load Data from a HDF5 file
# b) Train all classification pipelines
#    1) Stopping/Through-Going ("S")
#    2) Quality ("Q")
#    3) Multiplicity ("M")
# c) Validate those pipelines, also make a few plots etc
# d) Train a pipeline on the complete att_NP set and dump it
#
# 2016 T. Hoinka (tobias.hoinka@udo.edu)

import numpy as np
from docopt import docopt

from dataMethods import load_data
from physicalConstantsMethods import in_ice_range
from pipelineMethods import val_pipeline
from configParser import config_parser, pipeline_from_config

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
    mkdir_check(folder + "/plots_S")
    mkdir_check(folder + "/plots_Q")
    mkdir_check(folder + "/plots_M_1")
    mkdir_check(folder + "/plots_M_2")
    mkdir_check(folder + "/plots_M_3")
    mkdir_check(folder + "/plots_R")
    return folder

def get_labels(label, att):
    label_S = (label["label_in"].as_matrix() == 1.0)
    label_M = (label["n_mu_stop"].as_matrix())[label_S]
    label_M[label_M > 3.0] = 3.0
    label_M[label_M == 0.0] = 1.0

    label_R = label["true_stop_z"].as_matrix()[label_S]
    zenith_splinempe = att["zenith_SplineMPE"].as_matrix()
    zenith_true = label["zenith_true"].as_matrix()
    azimuth_splinempe = att["azimuth_SplineMPE"].as_matrix()
    azimuth_true = label["azimuth_true"].as_matrix()
    ang_error = np.arccos(np.cos(azimuth_true-azimuth_splinempe) * np.sin(zenith_true) * np.sin(zenith_splinempe) + np.cos(zenith_true) * np.cos(zenith_splinempe))
    label_Q = (ang_error < 0.1)[label_S]
    return label_S, label_Q, label_M, label_R

def precut_(zenith, qratio, rstop, zenith_cut, qratio_cut, rstop_cut):
    return (zenith < 87 / 180.0 * np.pi) & (rstop < rstop_cut ) & (qratio < qratio_cut)

def load_with_precut_sampling(config_dict):
    data_list = config_dict["data"]["data_list"]
    precut_config = config_dict["precuts"]
    sample_fraction = config_dict["data"]["sample_fraction"]
    label, att, weights, groups = load_data(data_list)
    if sample_fraction != 1.0:
        sample_mask = group_select(groups, sample_fraction)
    else:
        sample_mask = np.ones(len(label), dtype=bool)

    # Precut mask is generated if chosen.
    rstop_cut = float(precut_config["precut_rstop"])
    qratio_cut = float(precut_config["precut_qratio"])
    zenith_cut = float(precut_config["precut_zenith"])
    zenith = att["zenith_SplineMPE"].as_matrix()
    qratio = (att["charge_v1_HVInIcePulses_all_out"] / att["charge_v0_HVInIcePulses_all_out"]).as_matrix()
    rstop = att["stop_point_r_HVInIcePulses"].as_matrix()
    precut_mask = precut_(zenith, qratio, rstop,
                          zenith_cut, qratio_cut, rstop_cut)
    label = label.iloc[sample_mask & precut_mask]
    att = att.iloc[sample_mask & precut_mask]
    weights = weights[sample_mask & precut_mask] / float(sample_fraction)
    groups = groups[sample_mask & precut_mask]
    print("%i samples selected." % np.sum(sample_mask & precut_mask))
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
    g = list(set(groups))
    selection = np.zeros(len(groups), dtype=bool)
    for i in g:
        members = (groups == i)
        selection[members] = (np.random.rand(np.sum(members)) < fraction)
    return selection

def name_bitmask(selection, names):
    mask = np.zeros(len(names), dtype=bool)
    for i in range(len(names)):
        if names[i] in selection:
            mask[i] = True
    return mask

def main_routine(config_dict):
    print("Loading from:")
    print(config_dict["data"]["data_list"])
    label, att, weights, groups = load_with_precut_sampling(config_dict)
    event_ids = (label["Event"].as_matrix())[:, 0]
    label_S, label_Q, label_M, label_R = get_labels(label, att)
    folder = prepare_output_directories(config_dict["output"]["dir"])
    val_pipeline(att.as_matrix(), label_S, weights, groups, config_dict,
                            "pipeline_s", folder)
    val_pipeline(att.as_matrix()[label_S, :], label_Q, 
                 weights[label_S], groups[label_S],
                 config_dict, "pipeline_q", folder)
    val_pipeline(att.as_matrix()[label_S, :], label_M,
                 weights[label_S], groups[label_S],
                 config_dict, "pipeline_m", folder)
    val_pipeline(att.as_matrix()[label_S, :], label_R,
                 weights[label_S], groups[label_S],
                 config_dict, "pipeline_r", folder)
    print("Models dumped.")
    print("Finished. ")

if __name__ == "__main__":
    args = docopt(__doc__)
    config_dict = config_parser(args["CONFIG"])
    main_routine(config_dict)