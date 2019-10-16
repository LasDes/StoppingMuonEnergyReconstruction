"""Plots the results of a feature-selection Skript.

Usage: feature_selection_plots.py --results <results> --storage <storage> --version <version>

-h --help                       Show this.
--results <results>             File wwith the results from feature_selection.py
--storage <storage>             Directory to store plots
--version <version>             Version of the fs-Algoritm used
"""
import numpy as np, pandas as pd
from sklearn.externals import joblib
from docopt import docopt
from scipy.interpolate import splrep, splev
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

def gaussian_smoothing(x, s):
    """Smoothes the vector x with a sigma of s using a Gaussian kernel.

    Parameters
    ----------
    x : array, shape=(length,)
        Vector to be smoothed.

    s : float
        Sigma Parameter of Gaussian kernel.

    Returns
    -------
    output : array, shape=(length,)
             Smoothed vector.

    """
    output = np.zeros(len(x))
    for i in range(len(output)):
        gaussian = np.exp(-(np.arange(len(x)) - i) ** 2 / (2.0 * s ** 2))
        gaussian /= np.sum(gaussian)
        output[i] = np.sum(x * gaussian)
    return output

def plot_roc_auc_score(scores, dest):
    """
    Recieves list 'scores' of all auc_score arrays for
    mm-, s-, q,-, m-, r-feature selection.
    Plots data together with a smoothed line to files and stores them in a
    directory specified by 'dest'.
    """
    s = 10

    plt.plot(scores[0], 'ro',markersize=2)
    plt.xlabel("Number of removed features")
    plt.ylabel("ROC AUC")
    #xnew = np.linspace(0,np.size(scores[0])-1,100)
    #power_smooth = spline(np.arange(np.size(scores[0])), scores[0], xnew)
    #tck = splrep(np.arange(np.size(scores[0])), scores[0], s=.01)
    #smooth = splev(np.linspace(0,np.size(scores[0])-1,100), tck)
    smooth = gaussian_smoothing(scores[0], s)
    plt.plot(np.arange(1,np.size(scores[0])+1), smooth)
    plt.savefig("%smimatch_plot.png" %dest)
    plt.clf()

    fig, ax = plt.subplots(2, 2, figsize=(25,15))
    # s plot
    ax[0,0].plot(scores[1][::-1], 'ro',markersize=2)
    ax[0,0].set_xlabel("Number of features")
    ax[0,0].set_ylabel("ROC AUC")
    ax[0,0].set_ylim(0.954, 0.960)
    #smoothness = 0.001
    #tck = splrep(np.arange(np.size(scores[1])), scores[1], s=smoothness)
    #smooth = splev(np.linspace(0,np.size(scores[1])-1,100), tck)
    smooth = gaussian_smoothing(scores[1][::-1], s)
    max_loc = np.argmax(smooth)
    ax[0,0].vlines(max_loc, 0.0, 1.0, color='black', linestyles='dashed')
    ax[0,0].annotate("%d Features" %max_loc, xy=(max_loc,0.957), xytext=(max_loc+1,0.957))
    ax[0,0].plot(np.arange(1,np.size(scores[1])+1), smooth)

    # q plot
    ax[0,1].plot(scores[2][::-1], 'ro',markersize=2)
    ax[0,1].set_xlabel("Number of features")
    ax[0,1].set_ylabel("Mean Squared Error")
    ax[0,1].set_ylim(0.926, 0.942)
    #smoothness = 0.02
    #tck = splrep(np.arange(np.size(scores[2])), scores[2], s=smoothness)
    #smooth = splev(np.linspace(0,np.size(scores[2])-1,100), tck)
    smooth = gaussian_smoothing(scores[2][::-1], s)
    max_loc = np.argmax(smooth)
    ax[0,1].vlines(max_loc, 0.0, 1.0, color='black', linestyles='dashed')
    ax[0,1].annotate("%d Features" %max_loc, xy=(max_loc,0.934), xytext=(max_loc+1,0.934))
    ax[0,1].plot(np.arange(1,np.size(scores[2])+1), smooth)

    # m plot
    ax[1,0].plot(scores[3][::-1], 'ro',markersize=2)
    ax[1,0].set_xlabel("Number of features")
    ax[1,0].set_ylabel("ROC AUC")
    ax[1,0].set_ylim(0.865, 0.885)
    #smoothness = 0.240
    #tck = splrep(np.arange(np.size(scores[3])), scores[3], s=smoothness)
    #smooth = splev(np.linspace(0,np.size(scores[3])-1,100), tck)
    smooth = gaussian_smoothing(scores[3][::-1], s)
    max_loc = np.argmax(smooth)
    ax[1,0].vlines(max_loc, 0.0, 1.0, color='black', linestyles='dashed')
    ax[1,0].annotate("%d Features" %max_loc, xy=(max_loc,0.875), xytext=(max_loc+1,0.875))
    ax[1,0].plot(np.arange(1,np.size(scores[3])+1), smooth)

    # r plot
    ax[1,1].plot(np.dot(scores[4][::-1], -1), 'ro',markersize=2)
    ax[1,1].set_xlabel("Number of features")
    ax[1,1].set_ylabel("-Mean Squared Error")
    ax[1,1].set_ylim(-9000, -8650)
    #smoothness = 22500000
    #tck = splrep(np.arange(np.size(scores[4])), scores[4], s=smoothness)
    #smooth = splev(np.linspace(0,np.size(scores[4])-1,100), tck)
    smooth = gaussian_smoothing(scores[4][::-1], s)
    max_loc = np.argmax(-smooth)
    ax[1,1].vlines(max_loc, -10000, -8000, color='black', linestyles='dashed')
    ax[1,1].annotate("%d Features" %max_loc, xy=(max_loc,-8900), xytext=(max_loc+1,-8900))
    ax[1,1].plot(np.arange(1,np.size(scores[4])+1), -smooth)

    fig.savefig("%sfeature_plots.png" %dest)

def plot_mRMR_score(mm_score, s_score, q_score, m_score, r_score, dest):
    """
    Recieves list 'scores' of all mRMR_score arrays for
    mm-, s-, q,-, m-, r-feature selection.
    Plots data to files and stores them in a
    directory specified by 'dest'.
    """

    # mismatch plot
    plt.plot(np.arange(1, np.size(mm_score)+1), mm_score)
    plt.xlabel("Number of features")
    plt.ylabel("cumulative mRMR-Score")
    plt.grid(True)
    plt.xlim(0,60)
    plt.ylim(0,0.02)
    plt.title('mRMR-score for mismatched features')
    max_loc = np.argmax(mm_score)
    plt.vlines(max_loc, 0.0, 0.05, color='black', linestyles='dashed')
    plt.annotate("%d Features" %max_loc, xy=(max_loc,0.01), xytext=(max_loc+1,0.01))
    plt.savefig("%smismatch_plot_v3.png" %dest)
    plt.clf()

    # feature plots
    fig, ax = plt.subplots(2, 2, figsize=(25,15))
    # s selection
    ax[0,0].plot(np.arange(1, np.size(s_score)+1), s_score)
    ax[0,0].set_xlabel("Number of features")
    ax[0,0].set_ylabel("cumulative mRMR-Score")
    ax[0,0].grid(True)
    ax[0,0].set_xlim(0,160)
    ax[0,0].set_ylim(0,0.25)
    ax[0,0].set_title('mRMR-score for s-features')
    max_loc = np.argmax(s_score)
    ax[0,0].vlines(max_loc, 0.0, 0.3, color='black', linestyles='dashed')
    ax[0,0].annotate("%d Features" %max_loc, xy=(max_loc,0.125), xytext=(max_loc+1,0.125))

    # q selection
    ax[0,1].plot(np.arange(1, np.size(q_score)+1), q_score)
    ax[0,1].set_xlabel("Number of features")
    ax[0,1].set_ylabel("cumulative mRMR-Score")
    ax[0,1].grid(True)
    ax[0,1].set_xlim(0,300)
    ax[0,1].set_ylim(0,0.6)
    ax[0,1].set_title('mRMR-score for q-features')
    max_loc = np.argmax(q_score)
    ax[0,1].vlines(max_loc, 0.0, 0.6, color='black', linestyles='dashed')
    ax[0,1].annotate("%d Features" %max_loc, xy=(max_loc,0.3), xytext=(max_loc+1,0.3))

    # m selection
    ax[1,0].plot(np.arange(1, np.size(m_score)+1), m_score)
    ax[1,0].set_xlabel("Number of features")
    ax[1,0].set_ylabel("cumulative mRMR-Score")
    ax[1,0].grid(True)
    ax[1,0].set_xlim(0,180)
    ax[1,0].set_ylim(0,0.4)
    ax[1,0].set_title('mRMR-score for m-features')
    max_loc = np.argmax(m_score)
    ax[1,0].vlines(max_loc, 0.0, 0.4, color='black', linestyles='dashed')
    ax[1,0].annotate("%d Features" %max_loc, xy=(max_loc,0.2), xytext=(max_loc+1,0.2))

    # r selection
    ax[1,1].plot(np.arange(1, np.size(r_score)+1), r_score)
    ax[1,1].set_xlabel("Number of features")
    ax[1,1].set_ylabel("cumulative mRMR-Score")
    ax[1,1].grid(True)
    ax[1,1].set_xlim(0,400)
    ax[1,1].set_ylim(0,2)
    ax[1,1].set_title('mRMR-score for r-features')
    max_loc = np.argmax(r_score)
    ax[1,1].vlines(max_loc, 0.0, 2.0, color='black', linestyles='dashed')
    ax[1,1].annotate("%d Features" %max_loc, xy=(max_loc,1.0), xytext=(max_loc+1,1.0))

    fig.savefig("%sfeature_plot_v3.png" %dest)
    fig.clf()

if __name__ == "__main__":
    args = docopt(__doc__)
    # call plotting methode
    if(args["--version"] == 'v1'):
        # read in reaults from pickle
        auc_mm, features_mm, auc_s, features_s, auc_q, features_q, auc_m, features_m, auc_r, features_r = joblib.load(args["--results"])
        plot_roc_auc_score([auc_mm, auc_s, auc_q, auc_m, auc_r], args["--storage"])
    elif(args["--version"] == 'v2'):
        pass                                  # TODO: not used right now
    elif(args["--version"] == 'v3'):
        # read in reaults from pickle
        mm_score, mm_features, s_score, s_features, q_score, q_features, m_score, m_features, r_score, r_features = joblib.load(args["--results"])
        plot_mRMR_score(mm_score, s_score[::-1], q_score[::-1], m_score[::-1], r_score[::-1], args["--storage"])
    else:
        print "Invalid Version ID! / Choose v1, v2 or v3 / v2 currently does nothing!"
