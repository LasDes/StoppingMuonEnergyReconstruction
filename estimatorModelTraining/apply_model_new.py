"""Apply trained model to datasets.

Usage: apply_model.py -i <input> -m <model> -o <output> -b <batches>

-h --help               Show this.
-i <input>              Input files (hdf5).
-m <model>              Model (pickle).
-o <output>             Output (csv).
-b <batches>            Number of batches to split input into. [default: 1]
"""

import numpy as np
import pandas as pd
from docopt import docopt

# from dataMethods import load_data_beta as load_data
from dataMethods_v3 import load_data
from physicalConstantsMethods import in_ice_range
import h5py

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

	n_batches = int(args["-b"])

	extracted_data = []

	for f in files:
		print("Loading data from %s ..." % f)

		print("Loading temp data and labels")

		temp_data = pd.read_hdf(f, "BestTrack")
		label = pd.read_hdf(f, "Hoinka_Labels")
		runs = temp_data["Run"]
		events = temp_data["Event"]
		subevents = temp_data["SubEvent"]
		stopping = label["label_in"].values

		file = h5py.File(f)

		n_input_lines = file['Hoinka_Labels'].size

		steps = np.linspace(0, n_input_lines, num=n_batches).astype(int)

		intervalls = [(steps[i], steps[i + 1]) for i in range(len(steps) - 1)]

		chunk_collection = []

		print("Beginning to load data in chunks and apply models")

		for n, batch in enumerate(intervalls):
			print("...Processing batch %i" % n)
			_, data, _, _ = load_data(file, batch, weights=False, verbosity=False)

			print("Applying S classification ...")
			score_s = apply_model_classif(model, data, "s")

			print("Applying M classification ...")
			score_m = apply_model_classif(model, data, "m")

			print("Applying Q classification ...")
			score_q = apply_model_regress(model, data, "q")

			print("Applying R regression ...")
			score_r = apply_model_regress(model, data, "r")

			zenith = data["Hoinka_zenith_SplineMPE"].values

			chunk = pd.DataFrame({"score_s": score_s,
								  "score_m": score_m,
								  "score_q": score_q,
								  "score_r": score_r,
								  "zenith": zenith})

			chunk_collection += [chunk]

		df = pd.concat(chunk_collection).reset_index()

		df["run"] = runs
		df["event"] = events
		df["subevent"] = subevents
		df["stopping"] = stopping

		extracted_data += [df]

	print("Dumping results ...")
	pd.concat(extracted_data).to_csv(args["-o"])

	print("Done.")
