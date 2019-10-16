"""Data/MC Mismatches Study.

Usage: data_mc_mismatches.py --data_input <data_input> --mc_input <mc_input> -o <output>

-h --help                  Show this.
--data_input <data_input>  Data input. Multiple files separated by comma.
--mc_input <mc_input>      MC input. Multiple files separated by comma.
-o <output>                Output file.
"""
import numpy as np
import pandas as pd
from dataMethods import load_data_beta as load_data
from sklearn.ensemble import RandomForestClassifier
from docopt import docopt
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
	data_files = args["--data_input"].split(",")
	mc_files = args["--mc_input"].split(",")

	print(args["-o"])

	mc_label, mc_data, mc_wgt, mc_grp = load_data(mc_files, verbosity=True)
	dt_label, dt_data, dt_wgt, dt_grp = load_data(data_files, weights=False,
		                                          verbosity=True)

	print(len(mc_data), len(dt_data))

	mc_data = mc_data.values[sample_to_weights(mc_wgt, len(dt_data)),:]

	print(mc_data.shape, dt_data.values.shape)

	label = np.concatenate((np.zeros(len(mc_data)), np.ones(len(dt_data))))
	data = np.concatenate((mc_data, dt_data.values))

	rf = RandomForestClassifier(n_estimators=100, n_jobs=32)
	
	rf.fit(data[::2], label[::2])
	P = rf.predict_proba(data[1::2])
	joblib.dump((rf, dt_data.columns.values, P, label), args["-o"])



