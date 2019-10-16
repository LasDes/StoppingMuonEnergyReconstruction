"""correlationPlotter_v2_mcStop.
Usage: correlationPlotter_v2_mcStop.py INPUT OUTPUT

-h --help  Show this screen.
INPUT      Input pickle containing dataframe with results from correlationStudy.
OUTPUT     Output directory path.
"""

import numpy as np
import pandas as pd
from sklearn.externals import joblib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
# from matplotlib.ticker import LogFormattercd
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
    plt.rc('figure', figsize=(6.4, 4.8))

    df = joblib.load(input)
    df['energy_stop_log'] = np.log10(df['energy_stop'])
    df['energy_mep_log'] = np.log10(df['energy_mep'])
    df['true_range'] = calc_range(df.true_stop_z, df.zenith_true)
    df['estimated_range'] = calc_range(df.estimated_stop_z, df.zenith_splinempe)
    df['true_range_log'] = np.log10(df['true_range'])
    df['estimated_range_log'] = np.log10(df['estimated_range'])

    cmap = plt.cm.magma
    #cmap.norm = colors.LogNorm(vmin=10e-15)
    cmap.set_under('w')

    print("*****Plot correlation between parent particle energy and multiplicity")
    spearman = df['energy_mep'].corr(df['n_mu'], method='spearman')
    plt.hist2d(df.energy_mep_log, np.log10(df['n_mu']), bins=(100, 100), cmap=cmap,
               norm=colors.LogNorm(), vmin=10e-8, weights=df.weight)
    plt.xlabel(r'$\log_{10}(E_{p} / GeV)$')
    plt.ylabel(r'$\log_{10}(n_{\mu})$')
    # plt.title('')
    #formatter = LogFormatterExponent(base=10)
    cbar = plt.colorbar()
    cbar.ax.set_ylabel(r'event rate $1/s$', rotation=90, labelpad=20)
    plt.text(3, 5, 'spearman correlation: %.3f' % spearman)
    plt.savefig("%s/mepEnergyVSMulti.png" % output)
    plt.close()

    print("*****Plot multiplicity hist for stopping")
    total = df.weight.sum()
    multi1 = df[df.n_mu == 1].weight.sum()
    multi2 = df[df.n_mu == 2].weight.sum()
    multi3 = df[df.n_mu == 3].weight.sum()
    multi4 = df[df.n_mu == 4].weight.sum()
    multi5 = df[df.n_mu == 5].weight.sum()
    multiG = df[df.n_mu > 5].weight.sum()
    bar = plt.bar(np.arange(6), [multi1, multi2, multi3, multi4, multi5, multiG],
                  color='yellowgreen')
    plt.xticks(np.arange(6), [r'$n_{\mu} = 1$', r'$n_{\mu} = 2$', r'$n_{\mu} = 3$', r'$n_{\mu} = 4$',
                              r'$n_{\mu} = 5$', r'$n_{\mu} > 5$'])
    percentages = [multi1 / total * 100,
                   multi2 / total * 100,
                   multi3 / total * 100,
                   multi4 / total * 100,
                   multi5 / total * 100,
                   multiG / total * 100]
    for i, rect in enumerate(bar):
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2.0, height, '{:.2f} \%'.format(percentages[i]), ha='center',
                 va='bottom')
    plt.xlabel(r'muon bundle multiplicity $n_{\mu}$')
    plt.ylabel(r'event rate $1/s$')
    # plt.title('')
    plt.savefig("%s/multiHist_mcStop.png" % output)
    plt.close()

    # reduce to stopping events only!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    df = df[(df.true_single_stopping) & (df.zenith_true < cut)]

    print("*****Plot correlation between parent particle energy and multiplicity for stopping")
    spearman = df['energy_mep'].corr(df['n_mu'], method='spearman')
    plt.hist2d(df.energy_mep_log, np.log10(df['n_mu']), bins=(100, 100), cmap=cmap,
               norm=colors.LogNorm(vmin=10e-8), weights=df.weight)
    plt.xlabel(r'$\log_{10}(E_{p} / GeV)$')
    plt.ylabel(r'$\log_{10}(n_{\mu})$')
    # plt.title('')
    cbar = plt.colorbar()
    cbar.ax.set_ylabel(r'event rate $1/s$', rotation=90, labelpad=20)
    plt.text(3, 5, 'spearman correlation: %.3f' % spearman)
    plt.savefig("%s/mepEnergyVSMulti_mcStop.png" % output)
    plt.close()

    print("*****Plot energy hist for stopping")
    plt.hist(df.energy_mep, color='yellowgreen', weights=df.weight, log=True, rwidth=0.9,
             bins=np.logspace(np.log10(df.energy_mep.min()), np.log10(df.energy_mep.max()), 30))
    plt.xscale('log')
    plt.xlabel(r'$E_{p} / GeV$')
    plt.ylabel(r'$1/s$')
    # plt.title('')
    plt.savefig("%s/energyHist_mcStop.png" % output)
    plt.close()

    print("*****Plot correlation between estimated stopping depth and true stopping depth")
    spearman = df['estimated_stop_z'].corr(df['true_stop_z'], method='spearman')
    plt.hist2d(df.true_stop_z, df.estimated_stop_z, bins=(100, 100), cmap=cmap,
               norm=colors.LogNorm(vmin=10e-8), weights=df.weight)
    plt.ylabel(r'$Z_{est} / m$')
    plt.xlabel(r'$Z_{MC} / m$')
    #plt.title('')
    cbar = plt.colorbar()
    cbar.ax.set_ylabel(r'event rate $1/s$', rotation=90, labelpad=20)
    plt.text(0, -400, 'spearman correlation: %.3f' % spearman, bbox=dict(facecolor='white', edgecolor='none', alpha=0.9))
    plt.savefig("%s/trueStoppingVSestimatedStopping_mcStop.png" % output)
    plt.close()

    print("*****Plot correlation between estimated quality and true quality")
    spearman = df['estimated_quality'].corr(df['true_quality'], method='spearman')
    plt.hist2d(df.true_quality, df.estimated_quality, bins=(100, 100), cmap=cmap,
               norm=colors.LogNorm(vmin=10e-8), weights=df.weight)
    plt.ylabel(r'Estimated Quality')
    plt.xlabel(r'True Quality')
    # plt.title('')
    cbar = plt.colorbar()
    cbar.ax.set_ylabel(r'event rate $1/s$', rotation=90, labelpad=20)
    plt.text(-3.7, 0, 'spearman correlation: %.3f' % spearman,
             bbox=dict(facecolor='white', edgecolor='none', alpha=0.9))
    plt.savefig("%s/trueQualityVSestimatedQuality_mcStop.png" % output)
    plt.close()

    print("*****Plot correlation between true stopping depth and muon energy")

    spearman = df['true_stop_z'].corr(df['energy_stop'], method='spearman')
    plt.hist2d(df.true_stop_z, df.energy_stop_log, bins=(100, 100), cmap=cmap,
               norm=colors.LogNorm(vmin=10e-8), weights=df.weight)
    plt.ylabel(r'$\log_{10}(E_{\mu,surf} / GeV)$')
    plt.xlabel(r'$Z_{MC} / m$')
    # plt.title('')
    cbar = plt.colorbar()
    cbar.ax.set_ylabel(r'event rate $1/s$', rotation=90, labelpad=20)
    plt.text(0, 5, 'spearman correlation: %.3f' % spearman)
    plt.savefig("%s/trueStoppingVSmuonEnergy_mcStop.png" % output)
    plt.close()

    print("*****Plot correlation between parent particle energy and muon energy")

    spearman = df['energy_mep'].corr(df['energy_stop'], method='spearman')
    plt.hist2d(df.energy_stop_log, df.energy_mep_log, bins=(100, 100), cmap=cmap,
               norm=colors.LogNorm(vmin=10e-8), weights=df.weight)
    plt.xlabel(r'$\log_{10}(E_{\mu,surf} / GeV)$')
    plt.ylabel(r'$\log_{10}(E_{p} / GeV)$')
    # plt.title('')
    cbar = plt.colorbar()
    cbar.ax.set_ylabel(r'event rate $1/s$', rotation=90, labelpad=20)
    plt.text(3, 8, 'spearman correlation: %.3f' % spearman)
    plt.savefig("%s/mepEnergyVSmuonEnergy_mcStop.png" % output)
    plt.close()

    print("*****Plot correlation between parent particle energy and true stopping depth")

    spearman = df['energy_mep'].corr(df['true_stop_z'], method='spearman')
    plt.hist2d(df.true_stop_z, df.energy_mep_log, bins=(100, 100), cmap=cmap,
               norm=colors.LogNorm(vmin=10e-8), weights=df.weight)
    plt.ylabel(r'$\log_{10}(E_{p} / GeV)$')
    plt.xlabel(r'$Z_{MC} / m$')
    # plt.title('')
    cbar = plt.colorbar()
    cbar.ax.set_ylabel(r'event rate $1/s$', rotation=90, labelpad=20)
    plt.text(-400, 8, 'spearman correlation: %.3f' % spearman)
    plt.savefig("%s/mepEnergyVStrueStopping_mcStop.png" % output)
    plt.close()

    print("*****Plot correlation between estimated range and true range")
    spearman = df['estimated_range'].corr(df['true_range'], method='spearman')
    plt.hist2d(df.estimated_range_log, df.true_range_log, bins=(100, 100), cmap=cmap,
               norm=colors.LogNorm(vmin=10e-8), weights=df.weight)
    plt.xlabel(r'$\log_{10}(r_{est} / m)$')
    plt.ylabel(r'$\log_{10}(r_{MC} / m)$')
    # plt.title('')
    cbar = plt.colorbar()
    cbar.ax.set_ylabel(r'event rate $1/s$', rotation=90, labelpad=20)
    plt.text(5.3, 4.3, 'spearman correlation: %.3f' % spearman, bbox=dict(facecolor='white', edgecolor='none', alpha=0.9))
    plt.savefig("%s/trueRangeVSestimatedRange_mcStop.png" % output)
    plt.close()

    print("*****Plot correlation between true range and muon energy")
    spearman = df['energy_stop'].corr(df['true_range'], method='spearman')
    plt.hist2d(df.true_range_log, df.energy_stop_log, bins=(100, 100), cmap=cmap,
               norm=colors.LogNorm(vmin=10e-8), weights=df.weight)
    plt.ylabel(r'$\log_{10}(E_{\mu,surf} / GeV)$')
    plt.xlabel(r'$\log_{10}(r_{MC} / m)$')
    # plt.title('')
    cbar = plt.colorbar()
    cbar.ax.set_ylabel(r'event rate $1/s$', rotation=90, labelpad=20)
    plt.text(3.2, 5.0, 'spearman correlation: %.3f' % spearman)
    plt.savefig("%s/trueRangeVSmuonEnergy_mcStop.png" % output)
    plt.close()

    print("*****Plot correlation between true range and parent particle energy")
    spearman = df['energy_mep'].corr(df['true_range'], method='spearman')
    plt.hist2d(df.true_range_log, df.energy_mep_log, bins=(100, 100), cmap=cmap,
               norm=colors.LogNorm(vmin=10e-8), weights=df.weight)
    plt.ylabel(r'$\log_{10}(E_{p} / GeV)$')
    plt.xlabel(r'$\log_{10}(r_{MC} / m)$')
    # plt.title('')
    cbar = plt.colorbar()
    cbar.ax.set_ylabel(r'event rate $1/s$', rotation=90, labelpad=20)
    plt.text(3.2, 8.0, 'spearman correlation: %.3f' % spearman)
    plt.savefig("%s/trueRangeVSmepEnergy_mcStop.png" % output)
    plt.close()

    print("*****Plot correlation between estimated range and parent particle energy")
    spearman = df['energy_mep'].corr(df['estimated_range'], method='spearman')
    plt.hist2d(df.estimated_range_log, df.energy_mep_log, bins=(100, 100), cmap=cmap,
               norm=colors.LogNorm(vmin=10e-8), weights=df.weight)
    plt.ylabel(r'$\log_{10}(E_{p} / GeV)$')
    plt.xlabel(r'$\log_{10}(r_{est} / m)$')
    # plt.title('')
    cbar = plt.colorbar()
    cbar.ax.set_ylabel(r'event rate $1/s$', rotation=90, labelpad=20)
    plt.text(3.2, 8.0, 'spearman correlation: %.3f' % spearman)
    plt.savefig("%s/estimatedRangeVSmepEnergy_mcStop.png" % output)
    plt.close()

    # reduce to high quality events only!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    df = df[(df.true_quality < -0.6) & (df.zenith_true < cut)]

    print("*****Plot correlation between true range and muon energy (high quality)")
    spearman = df['energy_stop'].corr(df['true_range'], method='spearman')
    plt.hist2d(df.true_range_log, df.energy_stop_log, bins=(100, 100), cmap=cmap,
               norm=colors.LogNorm(vmin=10e-8), weights=df.weight)
    plt.ylabel(r'$\log_{10}(E_{\mu,surf} / GeV)$')
    plt.xlabel(r'$\log_{10}(r_{MC} / m)$')
    # plt.title('')
    cbar = plt.colorbar()
    cbar.ax.set_ylabel(r'event rate $1/s$', rotation=90, labelpad=20)
    plt.text(3.2, 4.5, 'spearman correlation: %.3f' % spearman, bbox=dict(facecolor='white', edgecolor='none', alpha=0.9))
    plt.savefig("%s/trueRangeVSmuonEnergy_HQ_mcStop.png" % output)
    plt.close()

    print("*****Plot correlation between estimated range and true range (high quality)")
    spearman = df['estimated_range'].corr(df['true_range'], method='spearman')
    plt.hist2d(df.estimated_range_log, df.true_range_log, bins=(100, 100), cmap=cmap,
               norm=colors.LogNorm(vmin=10e-8), weights=df.weight)
    plt.xlabel(r'$\log_{10}(r_{est} / m)$')
    plt.ylabel(r'$\log_{10}(r_{MC} / m)$')
    # plt.title('')
    cbar = plt.colorbar()
    cbar.ax.set_ylabel(r'event rate $1/s$', rotation=90, labelpad=20)
    plt.text(3.6, 3.2, 'spearman correlation: %.3f' % spearman)
    plt.savefig("%s/trueRangeVSestimatedRange_HQ_mcStop.png" % output)
    plt.close()

    print("*****Plot correlation between true range and parent particle energy (high quality)")
    spearman = df['energy_mep'].corr(df['true_range'], method='spearman')
    plt.hist2d(df.true_range_log, df.energy_mep_log, bins=(100, 100), cmap=cmap,
               norm=colors.LogNorm(vmin=10e-8), weights=df.weight)
    plt.ylabel(r'$\log_{10}(E_{p} / GeV)$')
    plt.xlabel(r'$\log_{10}(r_{MC} / m)$')
    # plt.title('')
    cbar = plt.colorbar()
    cbar.ax.set_ylabel(r'event rate $1/s$', rotation=90, labelpad=20)
    plt.text(3.2, 6.5, 'spearman correlation: %.3f' % spearman, bbox=dict(facecolor='white', edgecolor='none', alpha=0.9))
    plt.savefig("%s/trueRangeVSmepEnergy_HQ_mcStop.png" % output)
    plt.close()

    print("*****Plot correlation between estimated range and parent particle energy (high quality)")
    spearman = df['energy_mep'].corr(df['estimated_range'], method='spearman')
    plt.hist2d(df.estimated_range_log, df.energy_mep_log, bins=(100, 100), cmap=cmap,
               norm=colors.LogNorm(vmin=10e-8), weights=df.weight)
    plt.ylabel(r'$\log_{10}(E_{p} / GeV)$')
    plt.xlabel(r'$\log_{10}(r_{est} / m)$')
    # plt.title('')
    cbar = plt.colorbar()
    cbar.ax.set_ylabel(r'event rate $1/s$', rotation=90, labelpad=20)
    plt.text(3.2, 8.0, 'spearman correlation: %.3f' % spearman)
    plt.savefig("%s/estimatedRangeVSmepEnergy_HQ_mcStop.png" % output)
    plt.close()

    print("*****Plot correlation between zenith angle and parent particle energy (high quality)")
    spearman = df['energy_mep'].corr(df.zenith_true, method='spearman')
    plt.hist2d(df.zenith_true * 180 / np.pi, df.energy_mep_log, bins=(100, 100), cmap=cmap,
               norm=colors.LogNorm(vmin=10e-8), weights=df.weight)
    plt.ylabel(r'$\log_{10}(E_{p} / GeV)$')
    plt.xlabel(r'$\theta / ^\circ$')
    # plt.title('')
    cbar = plt.colorbar()
    cbar.ax.set_ylabel(r'event rate $1/s$', rotation=90, labelpad=20)
    plt.text(3.2, 8.0, 'spearman correlation: %.3f' % spearman)
    plt.savefig("%s/zenithVSmepEnergy_HQ_mcStop.png" % output)
    plt.close()


if __name__ == "__main__":
    args = docopt(__doc__)
    main(args["INPUT"], args["OUTPUT"])
