"""Analyse Model.
Usage: analyseModel.py PATH [--threshold=<threshold>]

-h                       Show this.
PATH                     Path to model.
--threshold=<threshold>  Purity threshold. [default: 0.9]
"""
from docopt import docopt
import numpy as np

def analyse_model(path_model, threshold):
	score_s = np.load(path_model)
	print(np.shape(score_s))
	label = score_s[:, 2]
	scores = score_s[:, 1]
	weight = score_s[:, 3]
	cuts = np.linspace(0.0, 1.0, 101)
	purities = np.zeros(len(cuts))
	print("Number of events: %.6f Hz" % np.sum(weight))
	print("Number of stopping events: %.6f Hz" % np.sum(weight[label == 1]))
	for i in range(len(cuts)):
		try:
			purities[i] = np.sum(weight[(label == 1) & (scores > cuts[i])]) / np.sum(weight[scores > cuts[i]])
		except:
			purities[i] = 0.0
		if purities[i] > threshold:
			print("Cut: %.3f, Puritiy: %.5f, Frequency: %.6f" % (cuts[i], purities[i], np.sum(weight[scores > cuts[i]])))
			break

if __name__ == "__main__":
	args = docopt(__doc__, version="Analyse Model")
	analyse_model(args["PATH"], float(args["--threshold"]))
