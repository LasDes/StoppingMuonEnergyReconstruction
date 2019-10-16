"""Feature Selection using preselected sample sets.

Usage: feature_selection.py --input_mc <input_mc> --input_data <input_data> -o <output>

-h --help                           Show this.
--input_mc <input_mc>               Monte Carlo input.
--input_data <input_data>           Data input.
-o <output>                         Output.
"""
import numpy as np
import pandas as pd
from rpy2.robjects.packages import importr
base = importr('base')
mr = importr('mRMRe')
import rpy2.robjects as ro
import rpy2.rinterface as ri
from rpy2.robjects import pandas2ri
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.externals import joblib
from sklearn.metrics import roc_auc_score
from docopt import docopt
from dataMethods import load_data_beta as load_data

def mismatch_selection(mc, data):
    dataset = np.concatenate((mc, data))
    dataframe = pd.DataFrame(dataset)
    label = np.concatenate((np.zeros(len(mc)), np.ones(len(data))))
    dataframe['label'] = label
    print "---Activating R Library"
    pandas2ri.activate()
    rdataframe = pandas2ri.py2ri(dataframe)
    mrmrData = mr.mRMR_data(data = (rdataframe))
    solutionCount = 1                               # TODO: maybe increase
    # use all available features for mRMR
    feature_count = np.shape(dataset)[1]

    print "---Starting mRMR-Algorithm"
    selectionEnsemble = mr.mRMR_ensemble(data = mrmrData,
        target_indices = dataframe.columns.get_loc('label')+1,
        feature_count = feature_count, solution_count = solutionCount )
    selectedFeatures = list([i-1 for i in list(mr.solutions(selectionEnsemble)[0])])
    selectedScores = list(mr.scores(selectionEnsemble)[0])

    # TODO: additional code required if solutionCount is set to >1

    # remove infs and nans
    dataframe = pd.DataFrame({'score': selectedScores,'feat': selectedFeatures})
    dataframe = dataframe.replace([np.inf, -np.inf], np.nan).dropna()

    # use the cummulated mRMR-score as a performance-score
    dataframe['score'] =  dataframe['score'].cumsum()
    print "---Done"

    return dataframe['score'], dataframe['feat']

def feature_selection(mc, label):
    dataframe = pd.DataFrame(mc)
    dataframe['label'] = label
    print "---Activating R Library"
    pandas2ri.activate()
    rdataframe = pandas2ri.py2ri(dataframe)
    mrmrData = mr.mRMR_data(data = (rdataframe))
    solutionCount = 1                               # TODO: maybe increase
    # use all available features for mRMR
    feature_count = np.shape(mc)[1]

    print "---Starting mRMR-Algorithm"
    selectionEnsemble = mr.mRMR_ensemble(data = mrmrData,
        target_indices = dataframe.columns.get_loc('label')+1,
        feature_count = feature_count, solution_count = solutionCount )
    selectedFeatures = [i-1 for i in list(mr.solutions(selectionEnsemble)[0])]
    selectedScores = list(mr.scores(selectionEnsemble)[0])

    # TODO: additional code required if solutionCount is set to >1

    # remove infs and nans
    dataframe = pd.DataFrame({'score': selectedScores,'feat': selectedFeatures})
    dataframe = dataframe.replace([np.inf, -np.inf], np.nan).dropna()

    # use the cummulated mRMR-score as a performance-score
    dataframe['score'] =  dataframe['score'].cumsum()
    print "---Done"

    return dataframe['score'], dataframe['feat']

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
    print("*****Loading mc-data and exp-data from hdf5")
    args = docopt(__doc__)
    mc_label, mc_feat, _, _ = joblib.load(args["--input_mc"])
    dt_feat = joblib.load(args["--input_data"])

    print("*****Removing constant and duplicated features")
    keys = mc_feat.columns.values

    mask = rm_constant(mc_feat)
    mask[mask] = rm_duplicate(mc_feat.values[:,mask])
    keys_masked = keys[mask]

    print("*****Generating labels")
    e_lab = mc_label.Hoinka_Labels_energy_mep.values

    zent = mc_label.Hoinka_Labels_zenith_true
    azit = mc_label.Hoinka_Labels_azimuth_true
    zens = mc_feat.SplineMPE_zenith
    azis = mc_feat.SplineMPE_azimuth

    # TODO: Use estimated labels instead of mc-labels
    m_lab = (mc_label.Hoinka_Labels_n_mu_stop.values == 1) & (mc_label.Hoinka_Labels_label_in.values == 1.0)

    ang_err = np.arccos(np.sin(zent) * np.cos(azit) * np.sin(zens) * np.cos(azis)
                        + np.sin(zent) * np.sin(azit) * np.sin(zens) * np.sin(azis)
                        + np.cos(zens) * np.cos(zent))

    q_lab = np.log10(ang_err)

    print("*****Executing mismatch selection")
    mm_score, mm_features = mismatch_selection(mc_feat.values[:,mask], dt_feat.values[:,mask])

    # take the number of features with the highest cummulated score as optimum
    mm_cut = np.argmax(mm_score) + 1
    print "%d features removed due to mismatch" %mm_cut

    mask_mm = np.copy(mask)
    sel_mm = mm_features[:mm_cut]
    mask_temp = np.ones(np.sum(mask), dtype=bool)
    mask_temp[sel_mm] = False
    mask_mm[mask_mm] = mask_temp
    keys_masked_mm = keys[mask_mm]

    # Reduce mc-data to single stopping HQ events

    mc_label = mc_label[(m_lab == 1.0) & (q_lab < -0.6)]
    mc_feat = mc_feat[(m_lab == 1.0) & (q_lab < -0.6)]

    e_lab = mc_label.Hoinka_Labels_energy_mep.values

    print("*****Executing feature-selection for unfolding of mep energy")
    score, features = feature_selection(mc_feat.values[:,mask_mm], e_lab)

    joblib.dump({'mm_score': mm_score,
                 'mm_features': keys_masked[mm_features],
                 'score': score,
                 'features': keys_masked_mm[features]
                 }, args["-o"])
