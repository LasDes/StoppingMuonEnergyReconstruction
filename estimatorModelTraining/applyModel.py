"""Model Application.

Usage: applyModel.py MODEL DATA OUTPUT

-h         Show this
MODEL      Path to Model.
DATA       Path to Data.
OUTPUT     Path to Output.
"""

# # # # # # # # # # # # #
# Reapply trained model #
# # # # # # # # # # # # # 
#
# This is meant to reapply a dumped model trained
# and validated in the main routine. The result is
# saved as a csv.
#
# 2016 T. Hoinka (tobias.hoinka@udo.edu)

from sklearn.externals import joblib
from dataMethods import load_data_beta as load_data
import pandas as pd
from docopt import docopt

def apply_model(path_to_model, filename_list, output):
	# Load models
	pipeline_S = joblib.load(path_to_model + "pipeline_s.pickle")
	pipeline_Q = joblib.load(path_to_model + "pipeline_q.pickle")
	pipeline_M = joblib.load(path_to_model + "pipeline_m.pickle")
	pipeline_R = joblib.load(path_to_model + "pipeline_r.pickle")

	# Load data
	lbl, att, wgt, grp = load_data(filename_list, weights=False,
		                           verbosity=True)
	data_table = att.as_matrix()
	print("Data loaded successfully.")

	# Apply model to data
	score_S = pipeline_S.predict_proba(data_table)
	print("Applied S-Classification.")

	score_Q = pipeline_Q.predict_proba(data_table)
	print("Applied Q-Classification.")

	score_M = pipeline_M.predict_proba(data_table)
	print("Applied M-Classification.")

	score_R = pipeline_R.predict(data_table)
	print("Applied R-Classification.")

	# Write results to file
	results = pd.DataFrame()
	results["score_S"] = score_S[:, 1]
	results["score_Q"] = score_Q[:, 1]
	results["score_M"] = score_M[:, 0]
	results["zenith"] = att["Hoinka_zenith_SplineMPE"].values
	results["stopping_depth"] = score_R
	results.to_csv(output)
	print("Results written to %s." % output)
	print("Finished.")

if __name__ == "__main__":
	args = docopt(__doc__, version="Model Application")
	apply_model(args["MODEL"], args["DATA"], args["OUTPUT"])