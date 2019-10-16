"""effectiveAreaCalculator_v5.
Usage: effectiveAreaCalculator.py PRE POST MC MODELS OUTPUT EMIN EMAX ZENMAX EBINS ZENBINS BATCHES [--read_pickle]

-h --help  Show this screen.
PRE        Input path to HDF5 file containing (pre-sim) mgs-data.
POST       Input path to HDF5 file containing (level 4) mgs-data.
MC         Input path to HDF5 file containing (level 4) corsica data.
MODELS     Input path to pickle file containing models.
OUTPUT     Output path.
EMIN       Lower Limit of energy interval (logscale: 10^EMIN).
EMAX       Upper limit of energy interval (logscale: 10^EMAX).
ZENMAX     Upper Limit of zenith interval
EBINS      Number of energy intervals.
ZENBINS    Number of zenith intervals.
BATCHES                  Number of batches to split up input.
--read_pickle      Flag determining weather corsica data have to be read from hdf5 or from pickle
"""

import numpy as np
import pandas as pd
import h5py
from docopt import docopt
from sklearn.externals import joblib
from dataMethods_mgs import load_data as load_data_mgs
from dataMethods_corsica import load_data as load_data_corsica

r_sim = 800
h_sim = 1600
A_sim = 2 * np.pi * r_sim * h_sim + 2 * np.pi * r_sim**2


def gen_labels(label, att):
    """Generates Labels from data.

    Parameters
    ----------
    label : Pandas Dataframe
            Labels

    att : Pandas Dataframe
          Attributes

    Returns
    -------
    labels_S : array, shape=(len(lab),)
               Label for S classification

    labels_Q : array, shape=(len(lab),)
               Label for Q classification

    labels_M : array, shape=(len(lab),)
               Label for M classification

    labels_R : array, shape=(len(lab),)
               Label for R regression
    """
    label_S = (label["Hoinka_Labels_label_in"].values == 1.0)
    label_M = (label["Hoinka_Labels_n_mu_stop"].values == 1) & label_S
    label_R = label["Hoinka_Labels_true_stop_z"].values
    zenith_splinempe = att["Hoinka_zenith_SplineMPE"].values
    zenith_true = label["Hoinka_Labels_zenith_true"].values
    azimuth_splinempe = att["Hoinka_azimuth_SplineMPE"].values
    azimuth_true = label["Hoinka_Labels_azimuth_true"].values
    ang_error = np.arccos(np.cos(azimuth_true-azimuth_splinempe) * np.sin(zenith_true) * np.sin(zenith_splinempe) +
                          np.cos(zenith_true) * np.cos(zenith_splinempe))
    # label_Q = (ang_error < 0.1)
    label_Q = np.log10(ang_error)
    return label_S, label_Q, label_M, label_R


def calc_generated_area(radius, height, zenith):
    return np.pi * 2 * radius * np.cos(zenith) + 2 * radius * height * np.sin(zenith)


def main(input_pre, input_post, mc_input, model_path, output, eMin, eMax, zenMax, ebins, zenbins, n_batches,
         read_pickle=False):
    cut = zenMax / 180 * np.pi

    print("*****Step1: calculate effective area by zenith angle from muon gun data")

    result = None

    input_pre_list = input_pre.split(",")
    input_post_list = input_post.split(",")

    total_count = 0
    l4_count = 0
    ssm_true_count = 0
    ssm_est_count = 0
    ssm_est_hq_count = 0

    for i in range(len(input_pre_list)):
        f_pre = input_pre_list[i]
        f_post = input_post_list[i]
        print("Loading data from %s and %s ..." % (f_pre, f_post))

        file_pre = h5py.File(f_pre)
        file_post = h5py.File(f_post)

        steps_pre = np.linspace(0, file_pre['Hoinka_Labels'].size, num=n_batches+1).astype(int)
        steps_post = np.linspace(0, file_post['Hoinka_Labels'].size, num=n_batches+1).astype(int)

        intervals_pre = [(steps_pre[i], steps_pre[i + 1]) for i in range(len(steps_pre) - 1)]
        intervals_post = [(steps_post[i], steps_post[i + 1]) for i in range(len(steps_post) - 1)]

        for n, batches in enumerate(zip(intervals_pre, intervals_post)):
            print("...Processing batch %i" % n)

            # read labeled (pre-sim) mgs-data
            # pre_data = pd.read_hdf(input_pre, key='Hoinka_Labels')
            pre_data, _, pre_data_weight, _ = load_data_mgs(file_pre, batches[0], verbosity=False)
            pre_data['MuonWeight'] = pre_data_weight
            store_total = len(pre_data.index)
            pre_data= pre_data[pre_data.Hoinka_Labels_zenith_true < cut]
            true_stopping = pre_data[pre_data.Hoinka_Labels_label_in > 0][
                ['Hoinka_Labels_azimuth_true', 'Hoinka_Labels_zenith_true', 'Hoinka_Labels_energy_stop',
                 'Hoinka_Labels_true_stop_z', 'Hoinka_Labels_n_mu_stop', 'MuonWeight']]

            pre_data.reset_index(inplace=True, drop=True)
            pre_data['Hoinka_Labels_zenith_true_cos'] = np.cos(pre_data.Hoinka_Labels_zenith_true)

            true_stopping.reset_index(inplace=True, drop=True)
            true_stopping['Hoinka_Labels_zenith_true_cos'] = np.cos(true_stopping.Hoinka_Labels_zenith_true)

            # import models
            models = joblib.load(model_path)

            # read level 3 mgs-data
            post_data, att, post_data_weight, _ = load_data_mgs(file_post, batches[1], verbosity=False)

            store_l4 = len(post_data.index)

            # apply s-classificator to level3 data
            proba_s = models['s'][1].predict_proba(att[models['s'][0]])[:, 1]
            proba_m = models['m'][1].predict_proba(att[models['m'][0]])[:, 1]
            predict_q = models['q'][1].predict(att[models['q'][0]])
            zenith_splinempe = att["Hoinka_zenith_SplineMPE"]
            del att

            # apply cut to labeled level 3 data at 0.74 for 95% purity
            # post_data = pd.read_hdf(input_post, key='Hoinka_Labels')
            post_data['MuonWeight'] = post_data_weight
            predicted_stopping = post_data[(proba_m > 0.79) & (predict_q < -0.6) & (zenith_splinempe < cut)][
                ['Hoinka_Labels_azimuth_true', 'Hoinka_Labels_zenith_true', 'Hoinka_Labels_energy_stop',
                 'Hoinka_Labels_true_stop_z', 'Hoinka_Labels_n_mu_stop', 'MuonWeight']]

            predicted_stopping.reset_index(inplace=True, drop=True)
            predicted_stopping['Hoinka_Labels_zenith_true_cos'] = np.cos(predicted_stopping.Hoinka_Labels_zenith_true)

            # perform aggregation and store results
            zen_bins = np.linspace(np.cos(cut), 1, num=zenbins+1)

            pre_data['zen_bin'] = pd.cut(pre_data['Hoinka_Labels_zenith_true_cos'], zen_bins)
            pre_data_agg = pre_data[['MuonWeight', 'zen_bin']].groupby('zen_bin').sum()

            true_stopping['zen_bin'] = pd.cut(true_stopping['Hoinka_Labels_zenith_true_cos'], zen_bins)
            true_stopping_agg = true_stopping[['MuonWeight', 'zen_bin']].groupby('zen_bin').sum()

            predicted_stopping['zen_bin'] = pd.cut(predicted_stopping['Hoinka_Labels_zenith_true_cos'], zen_bins)
            predicted_stopping_agg = predicted_stopping[['MuonWeight', 'zen_bin']].groupby('zen_bin').sum()

            if result is None:
                result = pd.concat({'true_count': true_stopping_agg, 'predicted_count': predicted_stopping_agg,
                                    'total_count': pre_data_agg}, axis=1)
                result.fillna(0, inplace=True)
            else:
                result.true_count += true_stopping_agg.fillna(0)
                result.predicted_count += predicted_stopping_agg.fillna(0)
                result.total_count += pre_data_agg.fillna(0)

            total_count += store_total
            l4_count += store_l4
            ssm_true_count += len(post_data[post_data.Hoinka_Labels_label_in > 0].index)
            ssm_est_count += len(post_data[(proba_m > 0.79)].index)
            ssm_est_hq_count += len(post_data[(proba_m > 0.79) & (predict_q < -0.6) & (zenith_splinempe < cut)].index)

    # prevent divisions by zero
    result.total_count = result['total_count'].replace(0.0, 1.0)
    result.true_count = result['true_count'].replace(0.0, 1.0)

    # calc effective areas
    result['effective_area'] = A_sim * result['predicted_count'] / result['true_count']
    result['effective_area_total'] = A_sim * result['predicted_count'] / result['total_count']

    result.to_csv("%s/effArea_mgs.csv" % output, sep='\t')

    joblib.dump(result, "%s/effArea_mgs.pickle" % output)

    print('total_count : %i' % total_count)
    print('l4_count : %i' % l4_count)
    print('ssm_true_count : %i' % ssm_true_count)
    print('ssm_est_count : %i' % ssm_est_count)
    print('ssm_est_hq_count : %i' % ssm_est_hq_count)

    print("*****Step2: calculate effective area by muon energy from corsica data")

    if read_pickle == False:
        # read corsica mc data and write to df
        df_list_true = []
        df_list_est = []

        for f in mc_input.split(","):
            print("Loading data from %s ..." % f)

            file = h5py.File(f)
            n_input_lines = file['Hoinka_Labels'].size

            steps = np.linspace(0, n_input_lines, num=n_batches+1).astype(int)

            intervals = [(steps[i], steps[i + 1]) for i in range(len(steps) - 1)]

            for n, batch in enumerate(intervals):
                print("...Processing batch %i" % n)
                lab, att, wgt, grp = load_data_corsica(file, batch, verbosity=False)

                models = joblib.load(model_path)

                proba_s = models['s'][1].predict_proba(att[models['s'][0]])[:, 1]
                estimate_q = models['q'][1].predict(att[models['q'][0]])
                proba_m = models['m'][1].predict_proba(att[models['m'][0]])[:, 1]
                estimate_r = models['r'][1].predict(att[models['r'][0]])

                lab_s, lab_q, lab_m, lab_r = gen_labels(lab, att)

                df = pd.DataFrame({'single_stopping': lab_m,
                                   'quality': lab_q,
                                   'zenith': lab["Hoinka_Labels_zenith_true"],
                                   'stop_z': lab["Hoinka_Labels_true_stop_z"],
                                   'energy_stop': lab["Hoinka_Labels_energy_stop"],
                                   'weight': wgt['G3'],
                                   'weight_G4': wgt['G4'],
                                   'weight_H': wgt['H']})

                df2 = pd.DataFrame({'single_stopping': proba_m,
                                   'quality': estimate_q,
                                   'zenith': att["Hoinka_zenith_SplineMPE"],
                                   'stop_z': estimate_r,
                                   'energy_stop': lab["Hoinka_Labels_energy_stop"],
                                   'weight': wgt['G3'],
                                   'weight_G4': wgt['G4'],
                                   'weight_H': wgt['H']})

                df_list_true += [df]
                df_list_est += [df2]

        result_mc = pd.concat(df_list_true).reset_index()
        result_mc_est = pd.concat(df_list_est).reset_index()

        result_mc['zenith_cos'] = np.cos(result_mc.zenith)
        result_mc_est['zenith_cos'] = np.cos(result_mc_est.zenith)

        # store readout
        joblib.dump(result_mc, "%s/df_corsica.pickle" % output)
        joblib.dump(result_mc_est, "%s/df_corsica_est.pickle" % output)
    else:
        print("Loading data from %s/df_corsica.pickle ..." % output)
        result_mc = joblib.load("%s/df_corsica.pickle" % output)
        print("Loading data from %s/df_corsica_est.pickle ..." % output)
        result_mc_est = joblib.load("%s/df_corsica_est.pickle" % output)

    # reduce to single stopping events with zenith below max
    result_mc = result_mc[(result_mc.single_stopping) & (result_mc.zenith < cut)]

    # retrieve effective area by zenith from previous result
    zen_bins = np.linspace(np.cos(cut), 1, num=zenbins+1)
    result_mc['zen_bin'] = pd.cut(result_mc.zenith_cos, zen_bins)
    result_mc['effective_area'] = result.effective_area.loc[result_mc.zen_bin].values
    result_mc['effective_area_total'] = result.effective_area_total.loc[result_mc.zen_bin].values

    # aggregate by muon energy
    e_bins = np.logspace(np.log10(eMin), np.log10(eMax), num=ebins+1)
    result_mc['e_bin'] = pd.cut(result_mc.energy_stop, e_bins)

    result_mc.dropna(inplace=True)

    result_mc_agg = result_mc[['effective_area', 'e_bin']].groupby('e_bin').\
        agg(lambda x: np.average(x,weights=result_mc.loc[x.index, "weight"]))

    result_mc_agg.fillna(0.0, inplace=True)

    result_mc_agg_total = result_mc[['effective_area_total', 'e_bin']].groupby('e_bin'). \
        agg(lambda x: np.average(x, weights=result_mc.loc[x.index, "weight"]))

    result_mc_agg.fillna(0.0, inplace=True)

    # store aggregation result
    result_mc_agg.to_csv("%s/effArea_mgs_corsica.csv" % output, sep='\t')
    result_mc_agg_total.to_csv("%s/effArea_mgs_corsica_total.csv" % output, sep='\t')

    joblib.dump(result_mc_agg, "%s/effArea_mgs_corsica.pickle" % output)
    joblib.dump(result_mc_agg_total, "%s/effArea_mgs_corsica_total.pickle" % output)

    print("*****Finished Succesfull!")


if __name__ == "__main__":
    args = docopt(__doc__)
    main(args["PRE"], args["POST"], args["MC"], args["MODELS"], args["OUTPUT"], float(args["EMIN"]),
         float(args["EMAX"]), float(args["ZENMAX"]), int(args["EBINS"]), int(args["ZENBINS"]), int(args["BATCHES"]),
         args["--read_pickle"])
