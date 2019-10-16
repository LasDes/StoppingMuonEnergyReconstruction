"""survivalStatisticsPlotter.
Usage: survivalStatisticsPlotter.py INPUT_E INPUT_N OUTPUT

-h --help  Show this screen.
INPUT_E      Input pickle containing dataframe with results from survivalStatistics (aggregation by Energy).
INPUT_N      Input pickle containing dataframe with results from survivalStatistics (aggregation by Multiplicity).
OUTPUT     Output directory path.
"""

import numpy as np
import pandas as pd
from sklearn.externals import joblib
import matplotlib.pyplot as plt
from docopt import docopt
import glob

def main(input_e, input_n, output):
    plt.rc('text', usetex=True)

    df = joblib.load(input_e)
    df['lim'] = df.index.categories.left
    df = df.append(df[-1:])
    df.lim[-1:] = df.index.categories[-1:].right
    df.set_index(df.lim, inplace=True)
    df['l0'].plot(loglog=True, drawstyle="steps-post", label='level0 events')
    df['l0_stopping'].plot(loglog=True, drawstyle="steps-post", label='level0 stopping events', grid=True)
    plt.xlabel(r'mean myon energie $\overline{E}_{\mu} / \text{GeV}$')
    plt.ylabel(r'$\#$')
    plt.legend(loc='best')
    plt.savefig("%s/level0_e_survival.png" % output)
    plt.close()

    df = joblib.load(input_e)
    df['lim'] = df.index.categories.left
    df = df.append(df[-1:])
    df.lim[-1:] = df.index.categories[-1:].right
    df.set_index(df.lim, inplace=True)
    df['l4'].plot(loglog=True, drawstyle="steps-post", label='level3/4 events', grid=True)
    df['l4_stopping'].plot(loglog=True, drawstyle="steps-post", label='level3/4 stopping events', grid=True)
    df['l5'].plot(loglog=True, drawstyle="steps-post", label='level5 events', grid=True)
    df['l5_stopping'].plot(loglog=True, drawstyle="steps-post", label='level5 stopping events', grid=True)
    plt.xlabel(r'mean myon energie $\overline{E}_{\mu} / GeV$')
    plt.ylabel(r'$\#$')
    plt.legend(loc='best')
    plt.savefig("%s/level4_e_survival.png" % output)
    plt.close()

    df = joblib.load(input_n)
    df['lim'] = df.index.categories.left
    df = df.append(df[-1:])
    df.lim[-1:] = df.index.categories[-1:].right
    df.set_index(df.lim, inplace=True)
    df['l0'].plot(logy=True, drawstyle="steps-post", label='level0 events')
    df['l0_stopping'].plot(logy=True, drawstyle="steps-post", label='level0 stopping events', grid=True)
    plt.xlabel(r'muon bundle multiplicity $n$')
    plt.ylabel(r'$\#$')
    plt.legend(loc='best')
    plt.savefig("%s/level0_n_survival.png" % output)
    plt.close()

    df = joblib.load(input_n)
    df['lim'] = df.index.categories.left
    df = df.append(df[-1:])
    df.lim[-1:] = df.index.categories[-1:].right
    df.set_index(df.lim, inplace=True)
    df['l4'].plot(logy=True, drawstyle="steps-post", label='level3/4 events', grid=True)
    df['l4_stopping'].plot(logy=True, drawstyle="steps-post", label='level3/4 stopping events', grid=True)
    df['l5'].plot(logy=True, drawstyle="steps-post", label='level5 events', grid=True)
    df['l5_stopping'].plot(logy=True, drawstyle="steps-post", label='level5 stopping events', grid=True)
    plt.xlabel(r'muon bundle multiplicity $n$')
    plt.ylabel(r'$\#$')
    plt.legend(loc='best')
    plt.savefig("%s/level4_n_survival.png" % output)
    plt.close()

if __name__ == "__main__":
    args = docopt(__doc__)
    main(args["INPUT_E"], args["INPUT_N"], args["OUTPUT"])