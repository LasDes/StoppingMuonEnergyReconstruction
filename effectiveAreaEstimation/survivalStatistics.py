"""survivalStatistics.
Usage: survivalStatistics.py LEVEL0 LEVEL3 LEVEL4 MODELS OUTPUT

-h --help  Show this screen.
LEVEL0     Input path to HDF5 file containing (pre-sim) mgs-data.
LEVEL3     Input path to HDF5 file containing (level 3) mgs-data.
LEVEL4     Input path to HDF5 file containing (level 4) mgs-data.
MODELS     Input path to pickle file containing models.
OUTPUT     Output directory path.
"""

import numpy as np
import pandas as pd
import h5py
from docopt import docopt
import glob
from sklearn.externals import joblib
from dataMethods import load_data_beta as load_data

def main(level_0, level_3, level_4, model_path, output):
    print("*****Step1: reading data")
    l0 = pd.read_hdf(level_0, key='Hoinka_Labels')
    l0_stopping = l0[l0.label_in > 0]
    l0_stopping.reset_index(inplace=True, drop=True)
    l0.reset_index(inplace=True, drop=True)

    l3 = pd.read_hdf(level_3, key='Hoinka_Labels')
    l3_stopping = l3[l3.label_in > 0]
    l3_stopping.reset_index(inplace=True, drop=True)
    l3.reset_index(inplace=True, drop=True)

    l4 = pd.read_hdf(level_4, key='Hoinka_Labels')
    l4_stopping = l4[l4.label_in > 0]
    l4_stopping.reset_index(inplace=True, drop=True)
    l4.reset_index(inplace=True, drop=True)

    print('total events %.f' % l0.shape[0])
    print('total events stopping %.f' % l0_stopping.shape[0])

    print("*****Step2: applying model")

    models = joblib.load(model_path)
    _, att, _, _ = load_data(level_4, weights=False, verbosity=False)
    proba_s = models['s'][1].predict_proba(att[models['s'][0]])[:, 1]
    l5 = l4[proba_s > 0.740]
    l5.reset_index(inplace=True, drop=True)
    l5_stopping = l5[l5.label_in > 0]
    l5_stopping.reset_index(inplace=True, drop=True)

    print("*****Step3: binning data by energy")
    e_bins = np.logspace(2, 5, 30)

    l0['e_bins'] = pd.cut(l0['energy_total'], e_bins)
    l0_stopping['e_bins'] = pd.cut(l0_stopping['energy_total'], e_bins)
    l3['e_bins'] = pd.cut(l3['energy_total'], e_bins)
    l3_stopping['e_bins'] = pd.cut(l3_stopping['energy_total'], e_bins)
    l4['e_bins'] = pd.cut(l4['energy_total'], e_bins)
    l4_stopping['e_bins'] = pd.cut(l4_stopping['energy_total'], e_bins)
    l5['e_bins'] = pd.cut(l5['energy_total'], e_bins)
    l5_stopping['e_bins'] = pd.cut(l5_stopping['energy_total'], e_bins)

    l0_binned = l0.groupby('e_bins').size()
    l0_stopping_binned = l0_stopping.groupby('e_bins').size()
    l3_binned = l3.groupby('e_bins').size()
    l3_stopping_binned = l3_stopping.groupby('e_bins').size()
    l4_binned = l4.groupby('e_bins').size()
    l4_stopping_binned = l4_stopping.groupby('e_bins').size()
    l5_binned = l5.groupby('e_bins').size()
    l5_stopping_binned = l5_stopping.groupby('e_bins').size()

    result_e = pd.concat(
        {'l0': l0_binned, 'l0_stopping': l0_stopping_binned, 'l3': l3_binned, 'l3_stopping': l3_stopping_binned,
         'l4': l4_binned, 'l4_stopping': l4_stopping_binned, 'l5': l5_binned, 'l5_stopping': l5_stopping_binned},
        axis=1)

    print("*****Step4: binning data by multiplicity")
    n_bins = np.linspace(1, 121, 31)

    l0['n_bins'] = pd.cut(l0['n_mu'], n_bins)
    l0_stopping['n_bins'] = pd.cut(l0_stopping['n_mu'], n_bins)
    l3['n_bins'] = pd.cut(l3['n_mu'], n_bins)
    l3_stopping['n_bins'] = pd.cut(l3_stopping['n_mu'], n_bins)
    l4['n_bins'] = pd.cut(l4['n_mu'], n_bins)
    l4_stopping['n_bins'] = pd.cut(l4_stopping['n_mu'], n_bins)
    l5['n_bins'] = pd.cut(l5['n_mu'], n_bins)
    l5_stopping['n_bins'] = pd.cut(l5_stopping['n_mu'], n_bins)

    l0_binned = l0.groupby('n_bins').size()
    l0_stopping_binned = l0_stopping.groupby('n_bins').size()
    l3_binned = l3.groupby('n_bins').size()
    l3_stopping_binned = l3_stopping.groupby('n_bins').size()
    l4_binned = l4.groupby('n_bins').size()
    l4_stopping_binned = l4_stopping.groupby('n_bins').size()
    l5_binned = l5.groupby('n_bins').size()
    l5_stopping_binned = l5_stopping.groupby('n_bins').size()

    result_n = pd.concat(
        {'l0': l0_binned, 'l0_stopping': l0_stopping_binned, 'l3': l3_binned, 'l3_stopping': l3_stopping_binned,
         'l4': l4_binned, 'l4_stopping': l4_stopping_binned, 'l5': l5_binned, 'l5_stopping': l5_stopping_binned},
        axis=1)

    print("*****Step5: writing to output")
    result_e.to_csv("%s/survival_byEnergy_results.csv" % output, sep='\t')
    joblib.dump(result_e, "%s/df_survival_byEnergy_results.pickle" % output)

    result_n.to_csv("%s/survival_byMulti_results.csv" % output, sep='\t')
    joblib.dump(result_n, "%s/df_survival_byMulti_results.pickle" % output)

    joblib.dump(pd.DataFrame([proba_s > 0.740]), "%s/df_level5.pickle" % output)


if __name__ == "__main__":
    args = docopt(__doc__)
    main(args["LEVEL0"], args["LEVEL3"], args["LEVEL4"], args["MODELS"], args["OUTPUT"])