"""Analyse Model New.
Usage: analyseModel_new.py PATH [--threshold=<threshold>]

-h                       Show this.
PATH                     Path to cv_predictions.
--threshold=<threshold>  Purity threshold. [default: 0.9]
"""
from docopt import docopt
import numpy as np
from sklearn.externals import joblib


def analyse_model(path_cv_predictions, threshold):
    cvPred = joblib.load(path_cv_predictions)
    print(np.shape(cvPred))
    for classification in ['s', 'm']:
        print("Evaluating %s-classification") % classification
        results = cvPred[classification]
        label = results[:,0]
        scores = results[:,1]
        weight = cvPred['weights']
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
                print("Cut: %.3f, Puritiy: %.5f, Frequency: %.6f" % (
                cuts[i], purities[i], np.sum(weight[scores > cuts[i]])))
                break


if __name__ == "__main__":
    args = docopt(__doc__, version="Analyse Model")
    analyse_model(args["PATH"], float(args["--threshold"]))
