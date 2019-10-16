"""correlationPlotter_nu.
Usage: correlationPlotter_nu.py INPUT OUTPUT

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
cut = 87.0 / 180 * np.pi


def calc_range(depth, zenith):
    b = R_E - depth_icecube + depth
    return -b * np.cos(zenith) + np.sqrt((R_E ** 2 - b ** 2) * np.sin(zenith) ** 2 + R_E ** 2 * np.cos(zenith) ** 2) + \
           np.finfo(float).max * (R_E ** 2 - b ** 2 < 0.0)


def main(input, output):
    plt.rc('text', usetex=True)
    plt.rc('axes', labelsize=15)
    plt.rc('figure', figsize=(6.4, 5.2))

    print("*****Loading pickled dataframe")

    df = joblib.load(input)
    df['energy_stop_log'] = np.log10(df['energy_stop'])
    df['energy_mep_log'] = np.log10(df['energy_mep'])
    df['true_range'] = calc_range(df.true_stop_z, df.zenith_true)
    df['estimated_range'] = calc_range(df.estimated_stop_z, df.zenith_splinempe)
    df['true_range_log'] = np.log10(df['true_range'])
    df['estimated_range_log'] = np.log10(df['estimated_range'])
    df['zenith_true_mod'] = df.zenith_true.apply(lambda x: x if x <= np.pi else 2*np.pi - x)
    df['zenith_true_deg'] = df.zenith_true_mod.apply(lambda x: x / np.pi * 180)
    df['cosz'] = np.cos(df.zenith_true_mod)


    cmap = plt.cm.magma
    cmap.set_under('w')

    print("*****From %i events in total, %i are stopping events (%f %%)"
          %(df.shape[0], df[(df.true_single_stopping)].shape[0],
            df[(df.true_single_stopping)].shape[0]/float(df.shape[0])*100))

    print("*****Plot zenith hist")
    df.zenith_true_deg.hist(histtype='step', weights=df.weight, fill=False, log=True, bins=np.linspace(0,180,30)
                            , label='total')
    df[df.true_single_stopping].zenith_true_deg.hist(histtype='step', weights=df[df.true_single_stopping].weight,
                                                     fill=False, log=True, bins=np.linspace(0, 180, 30),
                                                     label='true stopping')
    df[df.estimated_stopping > 0.9].zenith_true_deg.hist(histtype='step', weights=df[df.estimated_stopping > 0.9].weight,
                                                         fill=False, log=True, bins=np.linspace(0, 180, 30),
                                                         label='estimated stopping')
    plt.xlabel(r'$\theta / ^\circ$')
    plt.xlim(0,180)
    plt.ylabel(r'event rate $1/s$')
    plt.legend(loc='lower right')
    plt.savefig("%s/zenith_hist.png" % output)
    plt.close()

    # reduce to stopping events only!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    df = df[(df.true_single_stopping)]



if __name__ == "__main__":
    args = docopt(__doc__)
    main(args["INPUT"], args["OUTPUT"])
