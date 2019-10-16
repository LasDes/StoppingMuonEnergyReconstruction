"""Apply trained model to datasets.

Usage: apply_model.py -i <input> -m <model> -o <output>

-h --help               Show this.
-i <input>              Input files (hdf5).
-m <model>              Model (pickle).
-o <output>             Output (csv).
"""

import numpy as np
import pandas as pd
from docopt import docopt

from dataMethods import load_data_beta as load_data
from physicalConstantsMethods import in_ice_range

from sklearn.externals import joblib

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import KFold, StratifiedKFold

def apply_model_classif(model, data, letter):
	P = model[letter][1].predict_proba(data[model[letter][0]])
	return P[:,1]

def apply_model_regress(model, data, letter):
	P = model[letter][1].predict(data[model[letter][0]])
	return P

if __name__ == "__main__":
	args = docopt(__doc__)
	model = joblib.load(args["-m"])

	files = args["-i"].split(",")

	extracted_data = []

	for f in files:
		print("Loading data from %s ..." % f)

		_, data, _, _ = load_data(f, weights=False, verbosity=True)
		temp_data = pd.read_hdf(f, "BestTrack")
		runs = temp_data["Run"]
		events = temp_data["Event"]
		subevents = temp_data["SubEvent"]

		print("Applying S classification ...")
		score_s = apply_model_classif(model, data, "s")

		print("Applying M classification ...")
		score_m = apply_model_classif(model, data, "m")

		print("Applying Q classification ...")
		score_q = apply_model_regress(model, data, "q")

		print("Applying R regression ...")
		score_r = apply_model_regress(model, data, "r")

		zenith = data["Hoinka_zenith_SplineMPE"].values

		df = pd.DataFrame({"run":      runs,
		                   "event":    events,
		                   "subevent": subevents,
						   "score_s":  score_s,
		                   "score_m":  score_m,
		                   "score_q":  score_q,
		                   "score_r":  score_r,
		                   "zenith":   zenith})

		extracted_data += [df]

	print("Dumping results ...")
	pd.concat(extracted_data).to_csv(args["-o"])

	print("Done.")
