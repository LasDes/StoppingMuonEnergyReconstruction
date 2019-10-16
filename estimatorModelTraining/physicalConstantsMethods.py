# # # # # # # # # # # # # # # # # #
# Physical Methods and Constants  #
# # # # # # # # # # # # # # # # # #
#
# Methods for physical stuff.
#
# 2016 T. Hoinka (tobias.hoinka@udo.edu)

import numpy as np

R_EARTH = 6367440 # Radius of earth in meters

def in_ice_range(zenith, stop_z):
    b = R_EARTH - stop_z
    return -b * np.cos(zenith) + np.sqrt((R_EARTH**2 - b**2) * np.sin(zenith)**2 + R_EARTH**2 * np.cos(zenith)**2)