"""Feature Selection.
Usage: featureSelection.py PATH_MODEL PATH_DATA [--n_features=<n_features>]

-h                          Show this.
PATH_MODEL                  Path to trained models.
PATH_DATA                   Path to data with same names.
--n_features=<n_features>   Number of features to dump. [default: 40]
"""
# # # # # # # # # # # # # # # # # # # # # # #
# Feature Selection + Validation from Model #
# # # # # # # # # # # # # # # # # # # # # # #
# 
# Feature Selection from a trained (and validated) Model.
#
# 2016 T. Hoinka (tobias.hoinka@udo.edu)

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rcParams
from plotSetup import setupRcParams, COLORS as C
from docopt import docopt
from dataMethods import load_data
from sklearn.externals import joblib

def jaccard(set1, set2):
	return float(len(set1.intersection(set2))) / float(len(set1.union(set2)))

def robustness(fs, k):
	J = []
	for i in range(len(fs)):
		for j in range(len(fs)):
			if i != j:
				J += [jaccard(set(fs[i][-k:]), set(fs[j][-k:]))]
	return np.mean(J), np.std(J), np.max(J), np.min(J)

def robustness_vector(fs, start, end):
	R = np.zeros(end - start + 1)
	R_std = np.zeros(end - start + 1)
	R_max = np.zeros(end - start + 1)
	R_min = np.zeros(end - start + 1)
	K = np.linspace(start, end + 1, end - start + 1)
	for i in range(len(K)):
		R[i], R_std[i], R_max[i], R_min[i] = robustness(fs, K[i])
	return K, R, R_std, R_max, R_min

def robustness_plot(importances, set_fs, filename):
	importances = np.mean(np.array(importances), axis=0)
	importances = importances[np.argsort(importances)][::-1]
	K, r_vec, r_vec_std, r_vec_max, r_vec_min = robustness_vector(set_fs, 1,
	                                                              99)
	setupRcParams(rcParams)
	plt.bar(K - 0.25, 10.0 * importances[:99], color=C["r"], linewidth=0,
		    width=0.5)
	plt.errorbar(K, r_vec, yerr=[r_vec - r_vec_min, r_vec_max - r_vec],
		         color=C["g"], marker="s", markersize=4,
		         markeredgewidth=0, linestyle="")
	plt.xlim([0, 100])
	plt.ylim([0.0, 1.0])
	plt.savefig(filename)
	plt.close()

def ranking_plot(best_names, best_vals, filename):
	setupRcParams(rcParams)
	plt.figure(figsize=(8,18))
	plt.barh(np.linspace(39.75, 0.75, 40), best_vals[::-1],
	         color=C["g"], linewidth=0, height=0.5)
	plt.yticks(np.linspace(40, 1, 40), best_names[::-1])
	plt.tight_layout()
	plt.savefig(filename)
	plt.close()

def fs_from_model(path_model, path_data, n_features):
	L, A, W, G = load_data(path_data)
	names = A.columns.values

	# Load models and get all feature importances
	names_fs = []
	set_fs = []
	importances = []
	for i in range(10):
		M = joblib.load(path_model + "S_cv%i_pipeline.pickle" % i)
		rf = M.steps[1][1]
		fs = M.steps[0][1]
		sel_names = names[fs.get_support()]
		best = sel_names[np.argsort(rf.feature_importances_)]
		names_fs += [best]
		set_fs += [np.argsort(rf.feature_importances_)]
		importances += [rf.feature_importances_]
	robustness_plot(importances, set_fs, "feature_selection_S.pdf")
	ranking_plot(names_fs[0][set_fs[0][-40:]],
		         importances[0][set_fs[0][-40:]], "ranking_S.pdf")

	names_fs = []
	set_fs = []
	importances = []
	for i in range(10):
		M = joblib.load(path_model + "Q_cv%i_pipeline.pickle" % i)
		rf = M.steps[1][1]
		fs = M.steps[0][1]
		sel_names = names[fs.get_support()]
		best = sel_names[np.argsort(rf.feature_importances_)]
		names_fs += [best]
		set_fs += [np.argsort(rf.feature_importances_)]
		importances += [rf.feature_importances_]
	robustness_plot(importances, set_fs, "feature_selection_Q.pdf")
	ranking_plot(names_fs[0][set_fs[0][-40:]],
		         importances[0][set_fs[0][-40:]], "ranking_Q.pdf")

	names_fs = []
	set_fs = []
	importances = []
	for i in range(10):
		M = joblib.load(path_model + "M_cv%i_pipeline.pickle" % i)
		rf = M.steps[1][1]
		fs = M.steps[0][1]
		sel_names = names[fs.get_support()]
		best = sel_names[np.argsort(rf.feature_importances_)]
		names_fs += [best]
		set_fs += [np.argsort(rf.feature_importances_)]
		importances += [rf.feature_importances_]
	robustness_plot(importances, set_fs, "feature_selection_M.pdf")
	ranking_plot(names_fs[0][set_fs[0][-40:]],
		         importances[0][set_fs[0][-40:]], "ranking_M.pdf")

	names_fs = []
	set_fs = []
	importances = []
	for i in range(10):
		M = joblib.load(path_model + "R_cv%i_pipeline.pickle" % i)
		rf = M.steps[1][1]
		fs = M.steps[0][1]
		sel_names = names[fs.get_support()]
		best = sel_names[np.argsort(rf.feature_importances_)]
		names_fs += [best]
		set_fs += [np.argsort(rf.feature_importances_)]
		importances += [rf.feature_importances_]
	robustness_plot(importances, set_fs, "feature_selection_R.pdf")
	ranking_plot(names_fs[0][set_fs[0][-40:]],
		         importances[0][set_fs[0][-40:]], "ranking_R.pdf")

	print("Validation complete.")

	M = joblib.load(path_model + "pipeline_S.pickle")
	rf = M.steps[1][1]
	fs = M.steps[0][1]
	sel_names = names[fs.get_support()]
	best = sel_names[np.argsort(rf.feature_importances_)]
	np.save("feature_section_S_40.npy", best[-40:])

	M = joblib.load(path_model + "pipeline_Q.pickle")
	rf = M.steps[1][1]
	fs = M.steps[0][1]
	sel_names = names[fs.get_support()]
	best = sel_names[np.argsort(rf.feature_importances_)]
	np.save("feature_section_Q_40.npy", best[-40:])

	M = joblib.load(path_model + "pipeline_M.pickle")
	rf = M.steps[1][1]
	fs = M.steps[0][1]
	sel_names = names[fs.get_support()]
	best = sel_names[np.argsort(rf.feature_importances_)]
	np.save("feature_section_M_40.npy", best[-40:])

	M = joblib.load(path_model + "pipeline_R.pickle")
	rf = M.steps[1][1]
	fs = M.steps[0][1]
	sel_names = names[fs.get_support()]
	best = sel_names[np.argsort(rf.feature_importances_)]
	np.save("feature_section_R_20.npy", best[-20:])
	
if __name__ == "__main__":
    args = docopt(__doc__, version="Main Routine")
    fs_from_model(args["PATH_MODEL"], args["PATH_DATA"], args["--n_features"])