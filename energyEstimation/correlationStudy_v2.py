"""correlationStudy_v2.
Usage: correlationStudy_v2.py MC MODELS OUTPUT BATCHES

-h                       Show this.
MC                       Path to mc-data.
MODELS                   Path to trained models.
OUTPUT                   Output directory path.
BATCHES                  Number of batches to split up input.
"""
from docopt import docopt
import numpy as np
import pandas as pd
from sklearn.externals import joblib
from sklearn.metrics import roc_auc_score
from dataMethods_v3 import load_data
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import KFold, StratifiedKFold
import h5py


def load_and_prepare_data(input_files, batch):
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
    lab, att, wgt, grp = load_data(input_files, batch, verbosity=False)

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


def analyse_model(mc_input, model_input, output, n_batches):
    df_list = []

    for f in mc_input.split(","):
        print("Loading data from %s ..." % f)

        file = h5py.File(f)
        n_input_lines = file['Hoinka_Labels'].size  # 2127602
        # f.close()

        steps = np.linspace(0, n_input_lines, num=n_batches).astype(int)

        intervals = [(steps[i], steps[i + 1]) for i in range(len(steps) - 1)]

        for n, batch in enumerate(intervals):
            print("...Processing batch %i" %n)
            lab, att, wgt, grp = load_and_prepare_data(file, batch)

            models = joblib.load(model_input)

            proba_s = models['s'][1].predict_proba(att[models['s'][0]])[:, 1]
            estimate_q = models['q'][1].predict(att[models['q'][0]])
            proba_m = models['m'][1].predict_proba(att[models['m'][0]])[:, 1]
            estimate_r = models['r'][1].predict(att[models['r'][0]])

            lab_s, lab_q, lab_m, lab_r = gen_labels(lab, att)

            df = pd.DataFrame({'true_stopping': lab_s,
                               'estimated_stopping': proba_s,
                               'true_single_stopping': lab_m,
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
                               'estimated_stop_z': estimate_r,
                               'weight': wgt,
                               'in_ice_pulses': att["Hoinka_proj_std_TWSRTHVInIcePulses"],
                               'n_dir_doms' : att["BestTrackDirectHitsA_n_dir_doms"],
                               'smoothness' : att["BestTrackDirectHitsICA_dir_track_hit_distribution_smoothness"],
                               'best_track_z' : att["BestTrack_z"],
                               'MPEFit_TWHV_azimuth' : att["MPEFit_TWHV_azimuth"],
                               'muon_speed' : att["SplineMPETruncatedEnergy_SPICEMie_ORIG_Muon_speed"]})
            df_list += [df]

    results = pd.concat(df_list).reset_index()

    joblib.dump(results, "%s/df_correlationData_fullSet.pickle" % output)


if __name__ == "__main__":
    args = docopt(__doc__, version="Analyse Model")
    analyse_model(args["MC"], args['MODELS'], args["OUTPUT"], int(args["BATCHES"]))