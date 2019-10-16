"""Train a model for final classification..

Usage: train_model.py -i <input> -f <fs_output> -j <n_jobs> -o <output> -s <subsample_fraction> [--n_estimators <n_estimators> --min_samples <min_samples>]

-h --help                     Show this.
-i <input>                    Path to input files in the format path,path,...
-f <fs_output>                Path to output of feature selection routine.
-j <n_jobs>                   Number of jobs.
-o <output>                   Output directory.
-s <subsample_fraction>       Subsample fraction [default: 1.0]
--n_estimators <n_estimators> Number of estimators to be used [default: 200].
--min_samples <min_samples>   Minimum number of samples in leaves [default: 1].
"""
import numpy as np
from docopt import docopt

from dataMethods_new import load_data_alpha, load_data_beta
from physicalConstantsMethods import in_ice_range

from sklearn.externals import joblib

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import KFold, StratifiedKFold

def load_and_prepare_data(input_files, fraction):
    """Loads and prepares (if needed) the input files.

    Parameters
    ----------
    input_files : list(str)
                  List of paths to input files.

    fraction : float
               Fraction of samples to use.

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
        lab, att, wgt, grp = load_data_alpha(input_files, verbosity=True)
    else:
        lab, att, wgt, grp = load_data_beta(input_files, verbosity=True)
    if fraction < 1.0:
        subsample = np.random.choice(len(att), fraction * len(att))
    else:
        subsample = np.ones(len(att), dtype=bool)
    return (lab.iloc[subsample],
            att.iloc[subsample],
            wgt[subsample],
            grp[subsample])

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

def gaussian_smoothing(x, s):
    """Smoothes the vector x with a sigma of s using a Gaussian kernel.

    Parameters
    ----------
    x : array, shape=(length,)
        Vector to be smoothed.

    s : float
        Sigma Parameter of Gaussian kernel.

    Returns
    -------
    output : array, shape=(length,)
             Smoothed vector.

    """
    output = np.zeros(len(x))
    for i in range(len(output)):
        gaussian = np.exp(-(np.arange(len(x)) - i) ** 2 / (2.0 * s ** 2))
        gaussian /= np.sum(gaussian)
        output[i] = np.sum(x * gaussian)
    return output

def get_best_n_features(aucs, sigma=10.0):
    """Finds best feature number according to the result of the scoring
    function.

    Parameters
    ----------
    aucs : array, shape=(n_features,)
           The scoring function.

    sigma : float
            Sigma of the Gaussian kernel used for smoothing.

    Returns
    -------
    n_features : int
                 Optimal number of features.
    """
    aucs_smoothed = gaussian_smoothing(aucs[::-1], sigma)
    #n_features = np.argwhere(np.diff(aucs_smoothed) < 0.0)[0][0]
    n_features = np.argmax(aucs_smoothed)
    return n_features

def apply_fs(fs_path):
    """Evaluates the output of the feature selection routine.

    Parameters
    ----------
    fs_path : string
              Path to the output of the feature selection routine.

    Returns
    -------
    fs_s : list(str)
           List of the names of the selected names for the S classification.

    fs_q : list(str)
           List of the names of the selected names for the S classification.

    fs_m : list(str)
           List of the names of the selected names for the S classification.

    fs_r : list(str)
           List of the names of the selected names for the S classification.
    """
    fs_output = joblib.load(fs_path)
    n_features_s = get_best_n_features(np.array(fs_output[2]))
    n_features_q = get_best_n_features(np.array(fs_output[4]))
    n_features_m = get_best_n_features(np.array(fs_output[6]))
    n_features_r = get_best_n_features(-np.array(fs_output[8]))

    return (fs_output[3][::-1][:n_features_s],
            fs_output[5][::-1][:n_features_q],
            fs_output[7][::-1][:n_features_m],
            fs_output[9][::-1][:n_features_r])

def classifier(att, fs, lab, n_estimators=100, n_jobs=-1, subsample=None,
               min_samples=1):
    """Builds and validates a classification model using cross validation.

    Parameters
    ----------
    att : Pandas Dataframe
          Attributes

    fs : list(str)
         List of names of features selected.

    lab : array, shape=(len(att),)
          Label

    n_estimators : int
                   Number of estimators to be used.

    n_jobs : int
             Number of jobs.

    subsample : array, shape=(len(att),), dtype=bool
                Boolean mask to decide which samples to use for training.

    Returns
    -------
    predictions : array, shape=(len(att),2)
                  Array with the labels in the first and the cross validated
                  predictions in the second column.

    model : sklearn.ensemble.RandomForestClassifier object
            The final model.
    """
    if subsample is None:
        sel = np.ones(len(att), dtype=bool)
    else:
        sel = subsample
    att_sel = att.iloc[sel]
    lab_sel = lab[sel]
    cv = StratifiedKFold(n_splits=10, shuffle=True).split(att_sel, lab_sel)
    pred_all = np.zeros(len(att))
    pred_sel = np.zeros(np.sum(sel))

    for (train_idx, test_idx), i in zip(cv, range(10)):
        print("CV split %i / 10" % (i + 1))
        rf = RandomForestClassifier(n_estimators=n_estimators,
                                    min_samples_leaf=min_samples,
                                    n_jobs=n_jobs)
        rf.fit(att_sel[fs].values[train_idx,:], lab_sel[train_idx])
        pred_sel[test_idx] = rf.predict_proba(att_sel[fs].values[test_idx,:])[:,1]
    rf_final = RandomForestClassifier(n_estimators=n_estimators,
                                      min_samples_leaf=min_samples,
                                      n_jobs=n_jobs)
    rf_final.fit(att_sel[fs].values, lab_sel)
    pred_all[sel] = pred_sel
    if subsample is not None:
        pred_all[~sel] = rf_final.predict_proba(att[fs].values[~sel, :])[:,1]
    return np.vstack((lab, pred_all)).T, rf_final

def regressor(att, fs, lab, n_estimators=100, n_jobs=-1, subsample=None,
              min_samples=1):
    """Builds and validates a regression model using cross validation.

    Parameters
    ----------
    att : Pandas Dataframe
          Attributes

    fs : list(str)
         List of names of features selected.

    lab : array, shape=(len(att),)
          Label

    n_estimators : int
                   Number of estimators to be used.

    n_jobs : int
             Number of jobs.

    Returns
    -------
    predictions : array, shape=(len(att),2)
                  Array with the labels in the first and the cross validated
                  predictions in the second column.

    model : sklearn.ensemble.RandomForestRegressor object
            The final model.
    """
    if subsample is None:
        sel = np.ones(len(att), dtype=bool)
    else:
        sel = subsample
    att_sel = att.iloc[sel]
    lab_sel = lab[sel]
    cv = KFold(n_splits=10, shuffle=True).split(att_sel)
    pred_all = np.zeros(len(att))
    pred_sel = np.zeros(np.sum(sel))

    for (train_idx, test_idx), i in zip(cv, range(10)):
        print("CV split %i / 10" % (i + 1))
        rf = RandomForestRegressor(n_estimators=n_estimators,
                                   min_samples_leaf=min_samples,
                                   n_jobs=n_jobs)
        rf.fit(att_sel[fs].values[train_idx,:], lab_sel[train_idx])
        pred_sel[test_idx] = rf.predict(att_sel[fs].values[test_idx,:])
    rf_final = RandomForestRegressor(n_estimators=n_estimators,
                                     min_samples_leaf=min_samples,
                                     n_jobs=n_jobs)
    rf_final.fit(att_sel[fs].values, lab_sel)
    pred_all[sel] = pred_sel
    if subsample is not None:
        pred_all[~sel] = rf_final.predict(att[fs].values[~sel, :])
    return np.vstack((lab, pred_all)).T, rf_final

if __name__ == "__main__":
    args = docopt(__doc__)
    input_files = args["-i"].split(",")

    print("Loading data from")
    for n in input_files:
        print(n)

    lab, att, wgt, grp = load_and_prepare_data(input_files, float(args["-s"]))

    #lab, att, wgt, grp = joblib.load("/home/thoinka/data/mc_short.pickle")

    print("Data: %i samples, %i features" % (len(att), att.shape[1]))

    print("Generating labels ...")

    lab_s, lab_q, lab_m, lab_r = gen_labels(lab, att)

    print("S: %i, Q: %i, M: %i, R: %i" % (len(lab_s), len(lab_q), len(lab_m), len(lab_r)))

    print("Loading feature selection results ...")

    fs_s, fs_q, fs_m, fs_r = apply_fs(args["-f"])

    print("""S: %i features
Q: %i features
M: %i features
R: %i features""" % (len(fs_s), len(fs_q), len(fs_m), len(fs_r)))

    print("Evaluating S classification ...")

    n_estimators = int(args["--n_estimators"])
    min_samples = int(args["--min_samples"])
    n_jobs = int(args["-j"])

    sel_single = (lab_m == 1) & (lab_s == 1)
    sel_downgoing = att["Hoinka_zenith_SplineMPE"].values < 85.0 / 180.0 * np.pi

    cv_pred_s, pipeline_s = classifier(att, fs_s, lab_s,
                                       n_estimators=n_estimators,
                                       min_samples=min_samples,
                                       n_jobs=n_jobs)
    print("Evaluating Q classification ...")

    cv_pred_q, pipeline_q = regressor(att, fs_q,
                                      lab_q,
                                      subsample=sel_single & sel_downgoing,
                                      n_estimators=n_estimators,
                                      min_samples=min_samples,
                                      n_jobs=n_jobs)

    print("Evaluating M classification ...")

    cv_pred_m, pipeline_m = classifier(att, fs_m,
                                       lab_m,
    #                                  subsample=sel_stop,
                                       n_estimators=n_estimators,
                                       min_samples=min_samples,
                                       n_jobs=n_jobs)

    print("Evaluating R regression ...")

    cv_pred_r, pipeline_r = regressor(att, fs_r,
                                      lab_r,
                                      subsample=sel_single,
                                      n_estimators=n_estimators,
                                      min_samples=min_samples,
                                      n_jobs=n_jobs)

    print("Dumping models ...")

    joblib.dump({"s": (fs_s, pipeline_s),
                 "q": (fs_q, pipeline_q),
                 "m": (fs_m, pipeline_m),
                 "r": (fs_r, pipeline_r)}, "%s/models.pickle" % args["-o"])

    print("Dumping cross validation results ...")

    joblib.dump({"s": cv_pred_s,
                 "q": cv_pred_q,
                 "m": cv_pred_m,
                 "r": cv_pred_r,
                 "zenith": np.vstack((lab.Hoinka_Labels_zenith_true.values,
                                      att.SplineMPE_zenith.values)),
                 "weights": wgt}, "%s/cv_predictions.pickle" % args["-o"])
