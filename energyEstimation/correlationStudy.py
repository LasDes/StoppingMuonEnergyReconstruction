"""correlationStudy.
Usage: correlationStudy.py MC MODELS OUTPUT

-h                       Show this.
MC                       Path to mc-data.
MODELS                   Path to trained models.
OUTPUT                   Output directory path.
"""
from docopt import docopt
import numpy as np
import pandas as pd
from sklearn.externals import joblib
from sklearn.metrics import roc_auc_score
from dataMethods_new import load_data_alpha, load_data_beta
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
        lab, att, wgt, grp = load_data_alpha(input_files, verbosity=False)
    else:
        lab, att, wgt, grp = load_data_beta(input_files, verbosity=False)
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
    ang_error = np.arccos(np.cos(azimuth_true-azimuth_splinempe) * np.sin(zenith_true) * np.sin(zenith_splinempe) +
                          np.cos(zenith_true) * np.cos(zenith_splinempe))
    #label_Q = (ang_error < 0.1)
    label_Q = np.log10(ang_error)
    return label_S, label_Q, label_M, label_R

def analyse_model(mc_input, model_input, output):
    lab, att, wgt, grp = load_and_prepare_data(mc_input)

    models = joblib.load(model_input)

    proba_s = models['s'][1].predict_proba(att[models['s'][0]])[:, 1]
    estimate_q = models['q'][1].predict(att[models['q'][0]])
    proba_m = models['m'][1].predict_proba(att[models['m'][0]])[:, 1]
    estimate_r = models['r'][1].predict(att[models['r'][0]])

    lab_s, lab_q, lab_m, lab_r = gen_labels(lab, att)

    df = pd.DataFrame({'true_single_stopping': lab_m,
                       'estimated_single_stopping': proba_m,
                       'n_mu': lab["Hoinka_Labels_n_mu"],
                       'n_mu_stop': lab["Hoinka_Labels_n_mu_stop"],
                       'true_quality': lab_q,
                       'estimated_quality': estimate_q,
                       'zenith_true': lab["Hoinka_Labels_zenith_true"],
                       'zenith_splinempe': att["Hoinka_zenith_SplineMPE"],
                       'energy_mep': lab["Hoinka_Labels_energy_mep"],
                       'energy_stop': lab["Hoinka_Labels_energy_stop"],
                       'true_stop_z': lab["Hoinka_Labels_true_stop_z"],
                       'estimated_stop_z': estimate_r})

    df.to_csv("%s/correlationData.csv" % output, sep='\t')

    joblib.dump(df, "%s/df_correlationData.pickle" % output)

if __name__ == "__main__":
    args = docopt(__doc__, version="Analyse Model")
    analyse_model(args["MC"], args['MODELS'], args["OUTPUT"])