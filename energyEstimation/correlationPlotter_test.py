"""correlationPlotter.
Usage: correlationPlotter.py INPUT OUTPUT

-h --help  Show this screen.
INPUT      Input pickle containing dataframe with results from correlationStudy.
OUTPUT     Output directory path.
"""

import numpy as np
import pandas as pd
from sklearn.externals import joblib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.ticker import LogFormatter
from matplotlib.ticker import LogFormatterExponent
from docopt import docopt
import glob

# R_E = 12713550
R_E = 6371000
depth_icecube = 1950
cut = 87.0 / 180 * np.pi


def calc_range(depth, zenith):
    return np.sqrt((R_E**2 - (R_E - (depth_icecube + depth))**2)*np.sin(zenith)**2 + R_E**2 * np.cos(zenith)**2) -\
           (R_E - (depth_icecube + depth)) * np.cos(zenith)


def calc_in_ice_track(z, zenith):
    b = R_E - depth_icecube + z
    return -b * np.cos(zenith) + np.sqrt((R_E**2 - b**2) * np.sin(zenith)**2 + R_E**2 * np.cos(zenith)**2) +\
           np.finfo(float).max * (R_E**2 - b**2 < 0.0)


def main(input, output):
    plt.rc('text', usetex=True)

    df = joblib.load(input)
    df['energy_stop_log'] = np.log10(df['energy_stop'])
    df['energy_mep_log'] = np.log10(df['energy_mep'])
    df['true_range'] = calc_in_ice_track(df.true_stop_z, df.zenith_true)
    df['estimated_range'] = calc_in_ice_track(df.estimated_stop_z, df.zenith_splinempe)
    df['true_range_log'] = np.log10(df['true_range'])
    df['estimated_range_log'] = np.log10(df['estimated_range'])

    cmap = plt.cm.jet
    #cmap.norm = colors.LogNorm(vmin=10e-15, vmax=10e-5)
    cmap.set_under('w')

    # reduce to stopping events only!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    df = df[(df.true_single_stopping) & (df.zenith_true < cut)]
    # df = df[(df.true_stopping) & (df.n_mu == 1)]

    print("*****Plot energy hist for stopping")
    plt.hist(df.energy_mep, color='yellowgreen', weights=df.weight, log=True, rwidth=0.9,
             bins=np.logspace(np.log10(df.energy_mep.min()), np.log10(df.energy_mep.max()), 30))
    plt.xscale('log')
    plt.xlabel(r'$E_{p} / GeV$')
    plt.ylabel(r'event rate $1/Hz$')
    # plt.title('')
    plt.savefig("%s/energyHist_test.png" % output)
    plt.close()

    # reduce to high quality events only!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    df = df[df.true_quality < -0.6]


if __name__ == "__main__":
    args = docopt(__doc__)
    main(args["INPUT"], args["OUTPUT"])
