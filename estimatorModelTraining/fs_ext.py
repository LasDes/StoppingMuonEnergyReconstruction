"""FS

Usage: fs_ext.py --input_mc <pickle> --input_dt <pickle> --output <output>

-h --help                   Show this.
--input_mc <pickle>         Input pickle for mc data.
--input_dt <pickle>         Input pickle for real data.
--output <output>           Output path.
"""
import numpy as np
import pandas as pd
from featureSelectionMethods import rm_low_var, rm_weaker_correlated_features
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from docopt import docopt

if __name__ == "__main__":
	args = docopt(__doc__)
	mc_lab, mc_dat, _, _ = joblib.load(args["--input_mc"])
	dt_dat = joblib.load(args["--input_dt"])

	D = np.concatenate((mc_dat.values, dt_dat.values))
	L = np.concatenate((np.zeros(len(mc_dat)), np.ones(len(dt_dat))))

	f_mask = rm_low_var(dat, 1e-3) & rm_weaker_correlated_features(dat, lab.Hoinka_Labels_label_in, 0.999)

	D = D[:, f_mask]
	keys = mc_dat.columns.values[f_mask]

	aucs = []
	dropped_features = []
	dropped_mask = np.ones(np.sum(f_mask), dtype=bool)
	while(np.sum(dropped_mask) > 0):
		rf = RandomForestClassifier(n_estimators=50, n_jobs=16)
		rf.fit(D[::2,dropped_mask], L[::2])
		P = rf.predict_proba(D[1::2,dropped_mask])
		fi = rf.feature_importances_
		df = np.argmax(fi)
		dropped_mask[dropped_mask] *= ~(fi == np.max(fi))
		aucs += [roc_auc_score(L, P[:,1])]
		dropped_features += [K[df]]

	joblib.dump((dropped_features, aucs), args["--output"])