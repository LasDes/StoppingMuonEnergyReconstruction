# A two dimensional histogram fucntion that's actually usable.

import numpy as np
from matplotlib import pyplot as plt

def hist_2d(X, Y,              # Input
            nbins=[10, 10],    # Number of bins [n_x, n_y]
            weights=None,      # Weights for binning
            ranges=None,       # Range [[xmin, xmax], [ymin, ymax]]
            logx=False,        # Logarithmic x scale
            logy=False,        # Logarithmic y scale
            logz=False,        # Logairthmic z scale (counts)
            xlab=None,         # Label of xaxis
            ylab=None,         # Label of yaxis
            colorscale=False,  # Show colorscale or not
            diagonal=False):   # Show diagonal for correlation plots
    print(nbins)
    if ranges == None:
    	ranges = [[np.min(X), np.max(X)],
    	          [np.min(Y), np.max(Y)]]
    if logx is True:
    	x_bins = np.logspace(np.log10(ranges[0][0]),
    	                     np.log10(ranges[0][1]),
    	                     nbins[0])
    else:
    	x_bins = np.linspace(ranges[0][0], ranges[0][1], nbins[0])
    if logx is True:
    	y_bins = np.logspace(np.log10(ranges[1][0]),
    	                     np.log10(ranges[1][1]),
    	                     nbins[1])
    else:
    	y_bins = np.linspace(ranges[1][0], ranges[1][1], nbins[1])
    H, x_edges, y_edges = np.histogram2d(X, Y, bins=[x_bins, y_bins])
    plt.imshow(np.rot90(H), extent=[ranges[0][0], ranges[0][1],
                                    ranges[1][0], ranges[1][1]])
    plt.xlabel(xlab)
    plt.ylabel(ylab)

    if logx is True:
    	plt.xscale("log")

    if logy is True:
    	plt.yscale("log")