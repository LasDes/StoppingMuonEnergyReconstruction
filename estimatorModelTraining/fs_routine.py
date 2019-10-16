"""Feature Selection Routine. This is a routine that selects appropriate
features based on two criterions: The way they represent the label and whether
or not the features contain high amounts of mismatches with the data.

Usage: fs_routine.py --input_mc <input_mc,...> --input_data <input_data,...> -o <output>

-h --help                     Show this.
--input_mc <input_mc,...>     MC input.
--input_data <input_data,...> Data input.
-o <output>                   Output path.
"""
import numpy as np
import pandas as pd
from dataMethods import load_data_beta as load_data
from docopt import docopt
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib

def sample_to_weights(wgt, size):
	"""Samples according to given weights.

	Parameters
	----------
	wgt : array, shape=[n_samples,]
	      Weights, should be positive.

	size : int
	       Size of sample.

	Returns
	-------
	choice : array, shape=[size,]
	         List of picked samples.
	"""
	p_norm = wgt / np.sum(wgt)
	choice = np.random.choice(len(wgt), replace=False, p=p_norm, size=size)
	return choice

if __name__ == "__main__":
	args = docopt(__doc__)
	mc_files = args["--input_mc"].split(",")
	data_files = args["--input_data"].split(",")

	mc_label, mc_data, mc_wgt, mc_grp = load_data(mc_files, verbosity=True)
	dt_label, dt_data, dt_wgt, dt_grp = load_data(data_files, weights=False,
		                                          verbosity=True)

	w_mask = sample_to_weights(mc_wgt, len(dt_data))
	mc_data = mc_data.values[w_mask,:]
	mc_label = mc_label.Hoinka_Labels_label_in.values[w_mask]
	rfecv_a = RFECV(estimator=RandomForestClassifier(n_estimators=20,
	                                               n_jobs=24),
	              verbose=True, step=0.05)
	rfecv_a.fit(mc_data, mc_label)

	label = np.concatenate((np.zeros(len(mc_data)), np.ones(len(dt_data))))
	data = np.concatenate((mc_data, dt_data.values))

	rfecv_b = RFECV(estimator=RandomForestClassifier(n_estimators=20,
	                                               n_jobs=24),
	              verbose=True, step=0.05)
	rfecv_b.fit(data, label)
	joblib.dump((rfecv_a, rfecv_b, dt_data.columns.values), args["-o"])