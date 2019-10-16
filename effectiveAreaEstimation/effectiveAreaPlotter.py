"""effectiveAreaPlotter.
Usage: effectiveAreaPlotter.py INPUT OUTPUT

-h --help  Show this screen.
INPUT      Input pickle containing dataframe with results from effectiveAreaCalculator.
OUTPUT     Output directory path.
"""

import numpy as np
import pandas as pd
from sklearn.externals import joblib
import matplotlib.pyplot as plt
from docopt import docopt
import glob

def main(input, output):
    #plt.rc('text', usetex=True)
    plt.rc('axes', labelsize=15)
    plt.rc('figure', figsize=(6.4, 5.2))

    df = joblib.load(input)['e']
    df['center'] = np.sqrt(df.index.categories.left * df.index.categories.right)
    df['lim'] = df.index.categories.left
    df = df.append(df[-1:])
    df.lim[-1:] = df.index.categories[-1:].right
    df.set_index(df.lim, inplace=True)
    df.plot(loglog=True, drawstyle="steps-post", y='effective_area', legend=False, color='yellowgreen')
    # plt.errorbar(x=df.center, y=df.effective_area, yerr=df.error, c='yellowgreen', capsize=4, fmt='none')
    plt.xlabel(r'$E_{\mu} / GeV$')
    plt.ylabel(r'$A_{eff} / m^2$')
    plt.xlim(100, 3000)
    #fig.title('Effective area by muon energy')
    plt.savefig("%s/effectiveArea_byE.png" % output)
    plt.close()

    df = joblib.load(input)['zen']
    df['center'] = np.sqrt(df.index.categories.left.values * df.index.categories.right.values)
    df['lim'] = df.index.categories.left.values
    df = df.append(df[-1:])
    df.iloc[-1:, df.columns.get_loc('lim')] = df.index.categories[-1:].right.values
    # df['lim'] = df.lim.apply(lambda x: x / np.pi * 180)
    # df['center'] = df.center.apply(lambda x: x / np.pi * 180)
    df.set_index(df.lim, inplace=True)
    df.plot(logy=True, drawstyle="steps-post", y='effective_area', legend=False, color='yellowgreen')
    # plt.errorbar(x=df.center, y=df.effective_area, yerr=df.error, c='yellowgreen', capsize=4, fmt='none')
    plt.xlabel(r'$\cos(\theta)$')
    plt.ylabel(r'$A_{eff} / m^2$')
    # plt.xlim(0, 90)
    # fig.title('Effective area by muon energy')
    plt.savefig("%s/effectiveArea_byZen.png" % output)
    plt.close()

if __name__ == "__main__":
    args = docopt(__doc__)
    main(args["INPUT"], args["OUTPUT"])
