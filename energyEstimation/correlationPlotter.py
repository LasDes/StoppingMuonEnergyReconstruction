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
from docopt import docopt
import glob

R_E = 12713550
depth_icecube = 1950
cut = 87.0 / 180 * np.pi

def calc_range(depth, zenith):
    return np.sqrt((R_E**2 - (R_E - (depth_icecube + depth))**2)*np.sin(zenith)**2 + R_E**2 * np.cos(zenith)**2) -\
           (R_E - (depth_icecube + depth)) * np.cos(zenith)


def main(input, output):
    plt.rc('text', usetex=True)

    df = joblib.load(input)
    df['energy_stop_log'] = np.log10(df['energy_stop'])
    df['energy_mep_log'] = np.log10(df['energy_mep'])
    df['true_range'] = calc_range(df.true_stop_z, df.zenith_true)
    df['estimated_range'] = calc_range(df.estimated_stop_z, df.zenith_splinempe)
    df['true_range_log'] = np.log10(df['true_range'])
    df['estimated_range_log'] = np.log10(df['estimated_range'])

    df_stopping = df[df.single_stopping]

    df_cut_stopping = df_stopping[df_stopping.zenith_splinempe < cut]

    df_quality_cut_stopping = df_stopping[(df_stopping.true_quality < -0.6) & (df_stopping.zenith_splinempe < cut)]

    cmap = plt.cm.jet
    cmap.set_under(color='white')

    print("*****Plot correlation between parent particle energy and multiplicity")
    pearson = df['energy_mep_log'].corr(np.log10(df['n_mu']))
    spearman = df['energy_mep_log'].corr(np.log10(df['n_mu']), method='spearman')
    kendall = df['energy_mep_log'].corr(np.log10(df['n_mu']), method='kendall')
    plt.hist2d(df.energy_mep_log, np.log10(df['n_mu']), bins=(100, 100), cmap=cmap, vmin=0.1)
    plt.xlabel(r'log10 parent particle energie $\log(E_{p} / GeV)$')
    plt.ylabel(r'log10 muon bundle multiplicity $\log(n)$')
    # plt.title('')
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('$\#$', rotation=270, labelpad=20)
    plt.text(3, 5, 'pearson correlation: %.3f\nspearman correlation: %.3f\nkendall correlation: %.3f' % (pearson, spearman, kendall))
    plt.savefig("%s/mepEnergyVSMulti.png" % output)
    plt.close()

    print("*****Plot correlation between estimated stopping depth and true stopping depth")
    pearson = df_stopping['estimated_stop_z'].corr(df_stopping['true_stop_z'])
    spearman = df_stopping['estimated_stop_z'].corr(df_stopping['true_stop_z'], method='spearman')
    kendall = df_stopping['estimated_stop_z'].corr(df_stopping['true_stop_z'], method='kendall')
    plt.hist2d(df_stopping.true_stop_z, df_stopping.estimated_stop_z, bins=(100, 100), cmap=cmap, vmin=0.1)
    plt.ylabel(r'estimated stopping depth $Z_{est} / m$')
    plt.xlabel(r'true stopping depth $Z_{MC} / m$')
    #plt.title('')
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('$\#$', rotation=270, labelpad=20)
    plt.text(0,-400, 'pearson correlation: %.3f\nspearman correlation: %.3f\nkendall correlation: %.3f' % (pearson, spearman, kendall))
    plt.savefig("%s/trueStoppingVSestimatedStopping.png" % output)
    plt.close()

    print("*****Plot correlation between true stopping depth and muon energy")

    pearson = df_stopping['true_stop_z'].corr(df_stopping['energy_stop_log'])
    spearman = df_stopping['true_stop_z'].corr(df_stopping['energy_stop_log'], method='spearman')
    kendall = df_stopping['true_stop_z'].corr(df_stopping['energy_stop_log'], method='kendall')
    plt.hist2d(df_stopping.true_stop_z, df_stopping.energy_stop_log, bins=(100, 100), cmap=cmap, vmin=0.1)
    plt.ylabel(r'log10 mean myon energie $\log(\overline{E}_{\mu} / GeV)$')
    plt.xlabel(r'true stopping depth $Z_{MC} / m$')
    # plt.title('')
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('$\#$', rotation=270, labelpad=20)
    plt.text(0, 5, 'pearson correlation: %.3f\nspearman correlation: %.3f\nkendall correlation: %.3f' % (pearson, spearman, kendall))
    plt.savefig("%s/trueStoppingVSmuonEnergy.png" % output)
    plt.close()

    print("*****Plot correlation between parent particle energy and muon energy")

    pearson = df_stopping['energy_mep_log'].corr(df_stopping['energy_stop_log'])
    spearman = df_stopping['energy_mep_log'].corr(df_stopping['energy_stop_log'], method='spearman')
    kendall = df_stopping['energy_mep_log'].corr(df_stopping['energy_stop_log'], method='kendall')
    plt.hist2d(df_stopping.energy_stop_log, df_stopping.energy_mep_log, bins=(100, 100), cmap=cmap, vmin=0.1)
    plt.xlabel(r'log10 mean myon energie $\log(\overline{E}_{\mu} / GeV)$')
    plt.ylabel(r'log10 parent particle energie $\log(E_{p} / GeV)$')
    # plt.title('')
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('$\#$', rotation=270, labelpad=20)
    plt.text(3, 8, 'pearson correlation: %.3f\nspearman correlation: %.3f\nkendall correlation: %.3f' % (pearson, spearman, kendall))
    plt.savefig("%s/mepEnergyVSmuonEnergy.png" % output)
    plt.close()

    print("*****Plot correlation between parent particle energy and true stopping depth")

    pearson = df_stopping['energy_mep_log'].corr(df_stopping['true_stop_z'])
    spearman = df_stopping['energy_mep_log'].corr(df_stopping['true_stop_z'], method='spearman')
    kendall = df_stopping['energy_mep_log'].corr(df_stopping['true_stop_z'], method='kendall')
    plt.hist2d(df_stopping.true_stop_z, df_stopping.energy_mep_log, bins=(100, 100), cmap=cmap, vmin=0.1)
    plt.ylabel(r'log10 parent particle energie $\log(E_{p} / GeV)$')
    plt.xlabel(r'true stopping depth $Z_{MC} / m$')
    # plt.title('')
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('$\#$', rotation=270, labelpad=20)
    plt.text(-400, 8, 'pearson correlation: %.3f\nspearman correlation: %.3f\nkendall correlation: %.3f' % (pearson, spearman, kendall))
    plt.savefig("%s/mepEnergyVStrueStopping.png" % output)
    plt.close()

    print("*****Plot correlation between estimated range and true range")
    pearson = df_cut_stopping['estimated_range_log'].corr(df_cut_stopping['true_range_log'])
    spearman = df_cut_stopping['estimated_range_log'].corr(df_cut_stopping['true_range_log'],
                                                                   method='spearman')
    kendall = df_cut_stopping['estimated_range_log'].corr(df_cut_stopping['true_range_log'],
                                                          method='kendall')
    plt.hist2d(df_cut_stopping.estimated_range_log, df_cut_stopping.true_range_log, bins=(100, 100),
               cmap=cmap, vmin=0.1)
    plt.xlabel(r'log10 estimated range $\log(Z_{est} / m)$')
    plt.ylabel(r'log10 true stopping range $\log(Z_{MC} / m)$')
    # plt.title('')
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('$\#$', rotation=270, labelpad=20)
    plt.text(3.3, 4.3, 'pearson correlation: %.3f\nspearman correlation: %.3f\nkendall correlation: %.3f' % (pearson, spearman, kendall))
    plt.savefig("%s/trueRangeVSestimatedRange.png" % output)
    plt.close()

    print("*****Plot correlation between estimated range and true range (high quality)")
    pearson = df_quality_cut_stopping['estimated_range_log'].corr(df_quality_cut_stopping['true_range_log'])
    spearman = df_quality_cut_stopping['estimated_range_log'].corr(df_quality_cut_stopping['true_range_log'],
                                                                   method='spearman')
    kendall = df_quality_cut_stopping['estimated_range_log'].corr(df_quality_cut_stopping['true_range_log'],
                                                          method='kendall')
    plt.hist2d(df_quality_cut_stopping.estimated_range_log, df_quality_cut_stopping.true_range_log, bins=(100, 100),
               cmap=cmap, vmin=0.1)
    plt.xlabel(r'log10 estimated range $\log(Z_{est} / m)$')
    plt.ylabel(r'log10 true stopping range $\log(Z_{MC} / m)$')
    # plt.title('')
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('$\#$', rotation=270, labelpad=20)
    plt.text(3.3, 4.3, 'pearson correlation: %.3f\nspearman correlation: %.3f\nkendall correlation: %.3f' % (pearson, spearman, kendall))
    plt.savefig("%s/trueRangeVSestimatedRangeHQ.png" % output)
    plt.close()

    print("*****Plot correlation between true range and muon energy")
    pearson = df_cut_stopping['energy_stop_log'].corr(df_cut_stopping['true_range_log'])
    spearman = df_cut_stopping['energy_stop_log'].corr(df_cut_stopping['true_range_log'],
                                                           method='spearman')
    kendall = df_cut_stopping['energy_stop_log'].corr(df_cut_stopping['true_range_log'],
                                                                  method='kendall')
    plt.hist2d(df_cut_stopping.true_range_log, df_cut_stopping.energy_stop_log, bins=(100, 100),
               cmap=cmap, vmin=0.1)
    plt.ylabel(r'log10 mean myon energie $\log(\overline{E}_{\mu} / GeV)$')
    plt.xlabel(r'log10 true stopping range $\log(Z_{MC} / m)$')
    # plt.title('')
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('$\#$', rotation=270, labelpad=20)
    plt.text(3.2, 5.0, 'pearson correlation: %.3f\nspearman correlation: %.3f\nkendall correlation: %.3f' % (pearson, spearman, kendall))
    plt.savefig("%s/trueRangeVSmuonEnergy.png" % output)
    plt.close()

    print("*****Plot correlation between true range and parent particle energy")
    pearson = df_cut_stopping['energy_mep'].corr(df_cut_stopping['true_range_log'])
    spearman = df_cut_stopping['energy_mep'].corr(df_cut_stopping['true_range_log'],
                                                       method='spearman')
    kendall = df_cut_stopping['energy_mep'].corr(df_cut_stopping['true_range_log'],
                                                                  method='kendall')
    plt.hist2d(df_cut_stopping.true_range_log, df_cut_stopping.energy_mep_log, bins=(100, 100),
               cmap=cmap, vmin=0.1)
    plt.ylabel(r'log10 parent particle energie $\log(E_{p} / GeV)$')
    plt.xlabel(r'log10 true stopping range $\log(Z_{MC} / m)$')
    # plt.title('')
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('$\#$', rotation=270, labelpad=20)
    plt.text(3.2, 8.0, 'pearson correlation: %.3f\nspearman correlation: %.3f\nkendall correlation: %.3f' % (pearson, spearman, kendall))
    plt.savefig("%s/trueRangeVSmepEnergy.png" % output)
    plt.close()

    print("*****Plot correlation between estimated range and parent particle energy")
    pearson = df_cut_stopping['energy_mep'].corr(df_cut_stopping['estimated_range_log'])
    spearman = df_cut_stopping['energy_mep'].corr(df_cut_stopping['estimated_range_log'],
                                                  method='spearman')
    kendall = df_cut_stopping['energy_mep'].corr(df_cut_stopping['estimated_range_log'],
                                                 method='kendall')
    plt.hist2d(df_cut_stopping.estimated_range_log, df_cut_stopping.energy_mep_log, bins=(100, 100),
               cmap=cmap, vmin=0.1)
    plt.ylabel(r'log10 parent particle energie $\log(E_{p} / GeV)$')
    plt.xlabel(r'log10 true stopping range $\log(Z_{MC} / m)$')
    # plt.title('')
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('$\#$', rotation=270, labelpad=20)
    plt.text(3.2, 8.0, 'pearson correlation: %.3f\nspearman correlation: %.3f\nkendall correlation: %.3f' % (
    pearson, spearman, kendall))
    plt.savefig("%s/estimatedRangeVSmepEnergy.png" % output)
    plt.close()

    print("*****Plot correlation between true range and parent particle energy (high quality)")
    pearson = df_quality_cut_stopping['energy_mep'].corr(df_quality_cut_stopping['true_range_log'])
    spearman = df_quality_cut_stopping['energy_mep'].corr(df_quality_cut_stopping['true_range_log'],
                                                  method='spearman')
    kendall = df_quality_cut_stopping['energy_mep'].corr(df_quality_cut_stopping['true_range_log'],
                                                                  method='kendall')
    plt.hist2d(df_quality_cut_stopping.true_range_log, df_quality_cut_stopping.energy_mep_log, bins=(100, 100),
               cmap=cmap, vmin=0.1)
    plt.ylabel(r'log10 parent particle energie $\log(E_{p} / GeV)$')
    plt.xlabel(r'log10 true stopping range $\log(Z_{MC} / m)$')
    # plt.title('')
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('$\#$', rotation=270, labelpad=20)
    plt.text(3.2, 8.0, 'pearson correlation: %.3f\nspearman correlation: %.3f\nkendall correlation: %.3f' % (pearson, spearman, kendall))
    plt.savefig("%s/trueRangeVSmepEnergyHQ.png" % output)
    plt.close()

    print("*****Plot correlation between estimated range and parent particle energy (high quality)")
    pearson = df_quality_cut_stopping['energy_mep'].corr(df_quality_cut_stopping['estimated_range_log'])
    spearman = df_quality_cut_stopping['energy_mep'].corr(df_quality_cut_stopping['estimated_range_log'],
                                                          method='spearman')
    kendall = df_quality_cut_stopping['energy_mep'].corr(df_quality_cut_stopping['estimated_range_log'],
                                                         method='kendall')
    plt.hist2d(df_quality_cut_stopping.estimated_range_log, df_quality_cut_stopping.energy_mep_log, bins=(100, 100),
               cmap=cmap, vmin=0.1)
    plt.ylabel(r'log10 parent particle energie $\log(E_{p} / GeV)$')
    plt.xlabel(r'log10 estimated stopping range $\log(Z_{MC} / m)$')
    # plt.title('')
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('$\#$', rotation=270, labelpad=20)
    plt.text(3.2, 8.0, 'pearson correlation: %.3f\nspearman correlation: %.3f\nkendall correlation: %.3f' % (
    pearson, spearman, kendall))
    plt.savefig("%s/estimatedRangeVSmepEnergyHQ.png" % output)
    plt.close()


if __name__ == "__main__":
    args = docopt(__doc__)
    main(args["INPUT"], args["OUTPUT"])