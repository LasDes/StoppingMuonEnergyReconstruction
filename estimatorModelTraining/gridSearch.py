import numpy as np
from docopt import docopt
from dataMethods import load_data
from physicalConstantsMethods import in_ice_range
from pipelineMethods import S_pipeline, Q_pipeline, M_pipeline, R_pipeline, val_S_pipeline, val_Q_pipeline, val_M_pipeline, val_R_pipeline
from plottingMethods import validation_plots, validation_R_plots
from sklearn.externals import joblib
import time
import os
from weighting import compoundWeightGenerator

data_sets = ["../11058_all_short.hdf5"]
# Load att_NP
print("Loading from:")
print(data_sets)
label, att, weights, groups = load_data(data_sets)

event_ids = (label["Event"].as_matrix())[:, 0]
label_S = (label["label_in"].as_matrix() == 1.0)
radius = np.linspace(150.0, 250.0, 100)
att_NP = att.as_matrix()
for r in radius:
    label_S_b = label_S & (label["true_stop_r"].as_matrix() < r)

    # S Classification
    scores_S = val_S_pipeline(att_NP, label_S_b, weights, groups, "",
                              n_estimators=100,
                              n_features=80,
                              plots=False)

    cuts = np.linspace(0.0, 1.0, 101)
    perf = np.zeros((101, 2))
    for i in range(101):
        perf[i, 0] = np.sum(weights[scores_S > cuts[i]])
        perf[i, 1] = np.sum(weights[(scores_S > cuts[i]) & label_S]) / perf[i, 0]
        if perf[i, 1] > 0.99:
            print("%.5f, %.5f" % (r, perf[i, 0]))
            break