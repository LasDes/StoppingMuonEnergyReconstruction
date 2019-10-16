"""unfolding_v1.
Usage: effectiveAreaCalculator.py

-h --help  Show this screen.
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import LogNorm
from scipy.optimize import minimize
from sklearn.externals import joblib
from docopt import docopt

from tqdm import tqdm
import multiprocessing as mp
from joblib import Parallel, delayed

import warnings
warnings.filterwarnings('ignore')

R_E = 6371000
depth_icecube = 1950
cut = 87.0 / 180 * np.pi


def calc_range(depth, zenith):
    b = R_E - depth_icecube + depth
    return -b * np.cos(zenith) + np.sqrt((R_E ** 2 - b ** 2) * np.sin(zenith) ** 2 + R_E ** 2 * np.cos(zenith) ** 2) + np.finfo(float).max * (R_E ** 2 - b ** 2 < 0.0)

r_gen = 800
h_gen = 1600
A_gen = 2 * np.pi * r_gen * h_gen + 2 * np.pi * r_gen**2

eMin = 400
eMax = 30000

nbins_E = 20
nbins_r = 40

binning_E = np.logspace(np.log10(eMin), np.log10(eMax), nbins_E + 1)
binning_r = np.logspace(np.log10(2000), np.log10(40000), nbins_r + 1)

binning_idx_r = np.arange(nbins_r + 3) - 0.5
binning_idx_E = np.arange(nbins_E + 3) - 0.5

def llh_poisson(A, f_est, g):
    return np.sum(A.dot(f_est) - g * np.log(np.abs(A.dot(f_est)) + 1e-8))
    # return np.sum(A.dot(f_est) - g * np.log(np.abs(A.dot(f_est / acceptance)) + 1e-8))

def only_positive(f_est):
    return np.finfo('float').max * (f_est < 0.0).any()

def C_matrix(n):
    I = np.eye(n)
    C = 2.0 * I - np.roll(I, 1) - np.roll(I, -1)
    return C

def tikhonov_reg(f_est, tau, acceptance):
    C = C_matrix(len(f_est) - 2)
    return tau * np.sum(C.dot(np.log(np.abs(f_est[1:-1] / acceptance[1:-1]) + 10e+8)) ** 2)

def mcmc(x0, fun, step_size=1.5, n_steps=10000, print_acceptance=False, print_results=False):
    x = [x0]
    f = [fun(x0)]
    acc = 0
    for _ in range(n_steps):
        x_new = x[-1] + step_size * np.random.randn(len(x0))
        f_new = fun(x_new)
        prop_eval = -np.log(np.random.rand()) > f_new - f[-1]
        if prop_eval:
            x.append(x_new)
            f.append(f_new)
            acc += 1
        else:
            if print_results:
                print 'x_new: {}'.format(x_new)
                print 'f_new: {}'.format(f_new)
            x.append(x[-1])
            f.append(f[-1])
    if print_acceptance:
        print('{}% of proposed steps accepted.'.format(100 * acc / n_steps))
    return np.array(x), np.array(f)


def get_pull(pull_id, tau, input_df, acceptance, output=None, max_iter=100000, model='G3'):
    np.random.seed()
    permute = np.random.permutation(len(input_df))

    # mc = input_df.loc[permute[input_df.index % 2 != 0]]
    # data = input_df.loc[permute[input_df.index % 2 == 0]]

    if model is 'G3':
        sel = np.random.choice(input_df.index, len(input_df.index) / 2, p=input_df.loc[input_df.index, "weight_norm"],
                               replace=False)
    if model is 'G4':
        sel = np.random.choice(input_df.index, len(input_df.index) / 2,
                               p=input_df.loc[input_df.index, "weight_G4_norm"], replace=False)
    if model is 'H':
        sel = np.random.choice(input_df.index, len(input_df.index) / 2, p=input_df.loc[input_df.index, "weight_H_norm"],
                               replace=False)
    data = input_df[input_df.index.isin(sel)]
    mc = input_df[~input_df.index.isin(sel)]

    mc['energy_idx'] = np.digitize(mc.energy_stop, binning_E)
    mc['range_idx'] = np.digitize(mc.range, binning_r)

    data['energy_idx'] = np.digitize(data.energy_stop, binning_E)
    data['range_idx'] = np.digitize(data.range, binning_r)

    H, _, _ = np.histogram2d(mc['range_idx'], mc['energy_idx'], (binning_idx_r, binning_idx_E))
    A = H / np.sum(H, axis=0)
    A = np.nan_to_num(A)

    f_data, _ = np.histogram(data.energy_idx, binning_idx_E)
    g_data, _ = np.histogram(data.range_idx, binning_idx_r)

    function = lambda f_est: llh_poisson(A, f_est, g_data) \
                             + only_positive(f_est) \
                             + tikhonov_reg(f_est, tau, acceptance)

    result = minimize(function, x0=f_data, method='Nelder-Mead', options={'maxiter': 10000})

    x0 = 50.0 * np.ones(len(f_data))

    x_sample, f_sample = mcmc(x0, function, step_size=1.2, n_steps=10000)
    x_sample, f_sample = mcmc(result.x, function, step_size=1.2, n_steps=100000, print_acceptance=False,
                              print_results=False)

    if output is None:
        return [f_data, np.median(x_sample, axis=0), np.std(x_sample, axis=0)]
    else:
        output.put([f_data, np.median(x_sample, axis=0), np.std(x_sample, axis=0)])

reg_coeff_list = [1,10,50,100,250,500,1000,10000]

def main():
    data_dir = '/home/sninfa/jupyter/data'

    area = joblib.load('%s/400-30k_20Bins/effArea_mgs_corsica_total.pickle' % data_dir)
    input_df = joblib.load('%s/df_corsica_est.pickle' % data_dir)

    input_df = input_df[(input_df.single_stopping > 0.79) & (input_df.quality < -0.6) & (input_df.zenith < cut) & (
                input_df.energy_stop > 0.0)]

    input_df['range'] = calc_range(input_df.stop_z, input_df.zenith)
    input_df['range_log'] = np.log10(input_df.range)
    input_df['energy_log'] = np.log10(input_df.energy_stop)

    input_df['weight_norm'] = input_df.weight.values / input_df.weight.sum()
    input_df['weight_G4_norm'] = input_df.weight_G4.values / input_df.weight_G4.sum()
    input_df['weight_H_norm'] = input_df.weight_H.values / input_df.weight_H.sum()

    acceptance = area.values / A_gen
    acceptance = np.insert(acceptance, 0, 1.0)
    acceptance = np.append(acceptance, 1.0)

    pulls = []
    for tau in reg_coeff_list:
        pulls += [Parallel(n_jobs=8)(delayed(get_pull)(i, tau, input_df, acceptance, model='G3') for i in range(200))]

if __name__ == "__main__":
    args = docopt(__doc__)
    main()
