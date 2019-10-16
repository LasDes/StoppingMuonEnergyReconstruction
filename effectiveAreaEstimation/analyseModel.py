"""Analyse Model v3.
Usage: analyseModel_v3.py MC MODELS [--threshold=<threshold>]

-h                       Show this.
MC                       Path to mc-data.
MODELS                   Path to trained models.
--threshold=<threshold>  Purity threshold. [default: 0.9]
"""
from docopt import docopt
import numpy as np
import pandas as pd
from sklearn.externals import joblib
from sklearn.metrics import roc_auc_score
from dataMethods import load_data_alpha, load_data_beta
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import KFold, StratifiedKFold

def load_and_prepare_data(input_files):
    """Loads and prepares (if needed) the input files.

    Parameters
    ----------
    input_files : list(str)
                  List of paths to input files.

    Returns
    -------
    lab : Pandas Dataframe
          Labels extracted from files.

    att : Pandas Dataframe
          Attributes extracted from files.

    wgt : Array, shape=(len(lab),)
          Weights.

    grp : Array, shape=(len(lab),)
          File association of events.
    """
    if len(input_files) == 1:
        lab, att, wgt, grp = load_data_alpha(input_files, verbosity=False, weights=False)
    else:
        lab, att, wgt, grp = load_data_beta(input_files, verbosity=False, weights=False)
    return (lab, att, wgt, grp)

def gen_labels(label, att):
    """Generates Labels from data.

    Parameters
    ----------
    label : Pandas Dataframe
            Labels

    att : Pandas Dataframe
          Attributes

    Returns
    -------
    labels_S : array, shape=(len(lab),)
               Label for S classification

    labels_Q : array, shape=(len(lab),)
               Label for Q classification

    labels_M : array, shape=(len(lab),)
               Label for M classification

    labels_R : array, shape=(len(lab),)
               Label for R regression
    """
    label_S = (label["Hoinka_Labels_label_in"].values == 1.0)
    label_M = (label["Hoinka_Labels_n_mu_stop"].values == 1) & label_S
    label_R = label["Hoinka_Labels_true_stop_z"].values
    zenith_splinempe = att["Hoinka_zenith_SplineMPE"].values
    zenith_true = label["Hoinka_Labels_zenith_true"].values
    azimuth_splinempe = att["Hoinka_azimuth_SplineMPE"].values
    azimuth_true = label["Hoinka_Labels_azimuth_true"].values
    ang_error = np.arccos(np.cos(azimuth_true-azimuth_splinempe) * np.sin(zenith_true) * np.sin(zenith_splinempe) + np.cos(zenith_true) * np.cos(zenith_splinempe))
    #label_Q = (ang_error < 0.1)
    label_Q = np.log10(ang_error)
    return label_S, label_Q, label_M, label_R

def analyse_model(mc_input, model_input, threshold=0.9):
    lab, att, wgt, grp = load_and_prepare_data(mc_input)

    models = joblib.load(model_input)

    proba_s = models['s'][1].predict_proba(att[models['s'][0]])[:, 1]
    predict_q = models['q'][1].predict(att[models['q'][0]])
    proba_m = models['m'][1].predict_proba(att[models['m'][0]])[:, 1]
    predict_r = models['r'][1].predict(att[models['r'][0]])

    lab_s, lab_q, lab_m, lab_r = gen_labels(lab, att)

    print("Evaluate s-classification")

    print("ROC-AUC-Score = %.3f" % roc_auc_score(lab_s, proba_s, sample_weight=wgt))

    print("Get Cut for s-classification")

    cuts = np.linspace(0.0, 1.0, 101)
    purities = np.zeros(len(cuts))
    print("Number of events: %.6f Hz" % np.sum(wgt))
    print("Number of stopping events: %.6f Hz" % np.sum(wgt[lab_s == 1]))
    for i in range(len(cuts)):
        try:
            purities[i] = np.sum(wgt[(lab_s == 1) & (proba_s > cuts[i])]) / np.sum(wgt[proba_s > cuts[i]])
        except:
            purities[i] = 0.0
        if purities[i] > threshold:
            print("Cut: %.3f, Puritiy: %.5f, Frequency: %.6f" % (
                cuts[i], purities[i], np.sum(wgt[proba_s > cuts[i]])))
            break

    print("Evaluate r-regression")

    print("R^2-Score = %.3f" % models['r'][1].score(att[models['r'][0]], lab_r, sample_weight=wgt))

    #df = pd.DataFrame({'truth': lab_r, 'prediction': predict_r})

    #df.to_csv("/data/user/sninfa/classification_results/v02/regression_results.csv", sep='\t')

if __name__ == "__main__":
    args = docopt(__doc__, version="Analyse Model")
    analyse_model(args["MC"], args['MODELS'], float(args["--threshold"]))
