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

R_E = 6371000
depth_icecube = 1950
cut = 77.0 / 180 * np.pi


def calc_range(depth, zenith):
    b = R_E - depth_icecube + depth
    return -b * np.cos(zenith) + np.sqrt((R_E ** 2 - b ** 2) * np.sin(zenith) ** 2 + R_E ** 2 * np.cos(zenith) ** 2) + \
           np.finfo(float).max * (R_E ** 2 - b ** 2 < 0.0)


def main(input, output):
    plt.rc('text', usetex=True)
    plt.rc('axes', labelsize=15)
    plt.rc('figure', figsize=(6.4, 5.2))

    df = joblib.load(input)
    df['energy_stop_log'] = np.log10(df['energy_stop'])
    df['energy_mep_log'] = np.log10(df['energy_mep'])
    df['true_range'] = calc_range(df.true_stop_z, df.zenith_true)
    df['estimated_range'] = calc_range(df.estimated_stop_z, df.zenith_splinempe)
    df['true_range_log'] = np.log10(df['true_range'])
    df['estimated_range_log'] = np.log10(df['estimated_range'])

    cmap = plt.cm.magma
    cmap.set_under('w')

    # reduce to stopping events only!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    df = df[(df.estimated_single_stopping > 0.79) & (df.zenith_splinempe < cut)]

    # reduce to high quality events only with physically plausible ranges!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    df = df[df.estimated_quality < -0.6]

    print("*****Plot correlation between estimated range and true range (high quality)")
    spearman = df['estimated_range'].corr(df['true_range'], method='spearman')
    plt.hist2d(df.estimated_range_log, df.true_range_log, bins=(100, 100), cmap=cmap,
               norm=colors.LogNorm(vmin=10e-8), weights=df.weight)
    plt.xlabel(r'$\log_{10}(r_{est} / m)$')
    plt.ylabel(r'$\log_{10}(r_{MC} / m)$')
    # plt.title('')
    cbar = plt.colorbar()
    cbar.ax.set_ylabel(r'event rate $1/s$', rotation=90, labelpad=20)
    plt.text(3.6, 3.25, 'spearman correlation: %.3f' % spearman)
    plt.savefig("%s/trueRangeVSestimatedRange_HQ_estStop.png" % output)
    plt.close()

    print("*****Plot correlation between estimated range and parent particle energy (high quality)")
    spearman = df['energy_mep'].corr(df['estimated_range'], method='spearman')
    plt.hist2d(df.estimated_range_log, df.energy_mep_log, bins=(100, 100), cmap=cmap,
               norm=colors.LogNorm(vmin=10e-7), weights=df.weight)
    plt.ylabel(r'$\log_{10}(E_{p} / GeV)$')
    plt.xlabel(r'$\log_{10}(r_{est} / m)$')
    # plt.title('')
    cbar = plt.colorbar()
    cbar.ax.set_ylabel(r'event rate $1/s$', rotation=90, labelpad=20)
    plt.text(3.4, 8.0, 'spearman correlation: %.3f' % spearman)
    plt.savefig("%s/estimatedRangeVSmepEnergy_HQ_estStop.png" % output)
    plt.close()

    df = df[df.energy_stop > 0]

    print("*****Plot correlation between true range and muon energy (high quality)")
    spearman = df['energy_stop'].corr(df['estimated_range'], method='spearman')
    plt.hist2d(df.estimated_range_log, df.energy_stop_log, bins=(100, 100), cmap=cmap,
               norm=colors.LogNorm(vmin=10e-7), weights=df.weight)
    plt.ylabel(r'$\log_{10}(E_{\mu} / GeV)$')
    plt.xlabel(r'$\log_{10}(r_{est} / m)$')
    # plt.title('')
    cbar = plt.colorbar()
    cbar.ax.set_ylabel(r'event rate $1/s$', rotation=90, labelpad=20)
    plt.text(3.4, 5.5, 'spearman correlation: %.3f' % spearman)
    plt.savefig("%s/trueRangeVSmuonEnergy_HQ_estStop.png" % output)
    plt.close()


if __name__ == "__main__":
    args = docopt(__doc__)
    main(args["INPUT"], args["OUTPUT"])
