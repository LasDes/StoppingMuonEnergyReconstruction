"""
.. module:: global_methods
    :synopsis: A number of useful methods.

.. moduleauthor:: Tobias Hoinka <tobias.hoinka@udo.edu
"""
import numpy as np
from math import sqrt, ceil
from constantDefinitions import DUST_COEFF
from scipy.integrate import simps

"""Calculates weighted mean and std.

Parameters
----------
values : array, shape = [N,]
         Quantity to calculate mean and std of.

weights : array, shape = [N,]
          Weights.
Returns
-------
mean : float
       Mean of values.

std : float
      Standard deviation of values.
"""
def weighted_mean_std(values, weights):
    mean = np.average(values, weights=weights)
    std = sqrt(np.average((values - mean)**2, weights=weights))
    return mean, std

"""Sorts R into M bins of fixed length deltaR

Parameters
----------
R : array, shape = [N,]
    Quantity that gets sorted into bins.

start : float
        Outer edge of the first bin.

end : float
      Outer edge of the last bin.

deltaR : float
         Width of each bin.
Returns
-------
H : array, shape = [M,]
    Content of every bin.
"""
def fixed_width_hist(R, start, end, deltaR, weights=None):
    N = int(ceil(abs(end - start) / deltaR))
    bins = np.linspace(start, N * deltaR + start, N + 1)
    H = np.histogram(R, bins=bins, weights=weights)
    return np.abs(H)[0]

"""Calculates the range of a particle exactly, taking into account that
earth is a sphere.

Parameters
----------
R_E : float
      Earth's radius

T_I : float
      Depth of IceCube detector's zero (1950m)

zenith : float
         Zenith angle.

z : float
    Stopping Depth in IceCube Coordinates.

Returns
-------
range : float
        Range of the muons.
"""
def calc_in_ice_track(R_E, T_I, zenith, z):
    b = R_E - T_I + z
    if R_E**2 - b**2 < 0.0:
        return float("nan")
    return -b * np.cos(zenith) + np.sqrt((R_E**2 - b**2) * np.sin(zenith)**2 + R_E**2 * np.cos(zenith)**2)

"""Calculates quantity that is supposed to be a measure for how
dusty the track is. It basically integrates scattering and absorption co-
efficients from the Ackermann paper over a track.

Parameters
----------
v : array, shape = [3,]
    The directional vector of the track.

w : array, shape = [3,]
    The pivot vector of the track.

entry_z : float
          The z-component of the entry point.

exit_z : float
         The z-component of the exit point.

Returns
-------
B : float
    Dimensionless scattering value

A : float
    Dimensionless absorption value
"""
def dust_coeff(v, w, entry_t, exit_t):
    entry_z = v[2] * entry_t + w[2]
    exit_z = v[2] * exit_t + w[2]
    z = np.linspace(entry_z, exit_z, 20)
    t = np.linspace(entry_t, exit_t, 20)
    b = np.interp(z, DUST_COEFF[:, 0], DUST_COEFF[:, 1])
    a = np.interp(z, DUST_COEFF[:, 0], DUST_COEFF[:, 2])
    B = abs(simps(b, t)) / abs(exit_t - entry_t)
    A = abs(simps(a, t)) / abs(exit_t - entry_t)
    return B, A