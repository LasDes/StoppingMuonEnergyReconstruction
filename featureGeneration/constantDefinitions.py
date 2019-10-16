"""
.. module:: constant_definitions
    :synopsis: Defintions of useful constants not meant to be changed.

.. moduleauthor:: Tobias Hoinka <tobias.hoinka@udo.edu>
"""
import numpy as np
from scipy.spatial import Delaunay

"""
Detector Properties
-------------------
* ``N_DOMS_PER_STRING``: Number of doms per string.

* ``N_STRINGS``: Number of Strings.

* ``DETECTOR_R``: Radius of idealized cylindrical detector.

* ``DETECTOR_H``: Height of idealized cylindrical detector.

* ``DETECTOR_R_INNER``: Radius of idealized cylindrical inner detector.

* ``DETECTOR_H_INNER``: Height of idealized cylindrical inner detector.

* ``DEEP_CORE_STRINGS``: Strings that belong to the deep core extension.
"""
N_DOMS_PER_STRING = 60
N_STRINGS = 86

DETECTOR_R = 500
DETECTOR_H = 1000
DETECTOR_R_INNER = 400
DETECTOR_H_INNER = 750

DEEP_CORE_STRINGS = np.array([79, 80, 81, 82, 83, 84, 85, 86])
"""
Shell Region Definitions
------------------------
* ``SHELL_0_STRINGS``: Strings belonging to ``V0`` region.

* ``SHELL_0_OMS``: Oms belonging to ``V0`` region.

* ``SHELL_1_STRINGS``: Strings belonging to ``V1`` region.

* ``SHELL_1_OMS``: Oms belonging to ``V1`` region.

* ``SHELL_2_STRINGS``: Strings belonging to ``V2`` region.

* ``SHELL_2_OMS``: Oms belonging to ``V2`` region.

* ``SHELL_3_STRINGS``: Strings belonging to ``V3`` region.

* ``SHELL_3_OMS``: Oms belonging to ``V3`` region.
"""
SHELL_0_STRINGS = np.hstack(
    np.array([
        1, 2, 3, 4, 5, 6, 7, 13, 14, 21, 22, 30, 31, 40, 41, 50, 51, 59, 60,
        67, 68, 72, 73, 74, 75, 76, 77, 78
    ]))
SHELL_0_OMS = np.hstack(np.array([1, 2, 3, 4, 5, 56, 57, 58, 59, 60]))
SHELL_1_STRINGS = np.hstack(
    np.array([
        SHELL_0_STRINGS, 8, 9, 10, 11, 12, 15, 20, 23, 9, 32, 39, 42, 49, 52,
        58, 61, 64, 65, 66, 69, 70, 71
    ]))
SHELL_1_OMS = np.hstack(
    np.array([
        SHELL_0_OMS, 6, 7, 8, 9, 10, 51, 52, 52, 53, 54, 55
    ]))
SHELL_2_STRINGS = np.hstack(
    np.array([
        SHELL_1_STRINGS, 16, 17, 18, 19, 28, 38, 48, 57, 56, 55, 63, 62, 53,
        43, 33, 24
    ]))
SHELL_2_OMS = np.hstack(
    np.array([
        SHELL_1_OMS, 11, 12, 13, 14, 15, 46, 47, 48, 49, 50
    ]))
SHELL_3_STRINGS = np.hstack(
    np.array([
        SHELL_2_STRINGS, 25, 26, 27, 37, 47, 46, 54, 44, 34
    ]))
SHELL_3_OMS = np.hstack(
    np.array([
        SHELL_2_OMS, 16, 17, 18, 19, 20, 41, 42, 43, 44, 45
    ]))

"""
Detector Geometry
-----------------
* ``DET_HULL``: Delaunay triangulation of shape of a slightly (100 m) enlarged detector.

* ``BARE_DET_HULL``: Delaunay triangulation of shape of detector.

* ``CORE_HULL``: Delaunay triangulation of deep core region.
"""
DET_HULL = Delaunay(
    np.array([[-694.06, -137.15, 600.0], [680.69, 213.94, 600.0],
              [-300.93, -637.68, 600.0], [439.82, -519.92, 600.0],
              [373.47, 591.8, 600.0], [147.82, 532.61, 600.0],
              [66.19, 625.76, 600.0], [-436.86, 548.81, 600.0],
              [-694.06, -137.15, -600.0], [680.69, 213.94, -600.0],
              [-300.93, -637.68, -600.0], [439.82, -519.92, -600.0],
              [373.47, 591.8, -600.0], [147.82, 532.61, -600.0],
              [66.19, 625.76, -600.0], [-436.86, 548.81, -600.0]]))

BARE_DET_HULL = Delaunay(
    np.array([
        [-256.14, -521.08, -504.40], [361.00, -422.83, -504.71],
        [576.37, 170.92, -510.18], [338.44, 463.72, -498.50],
        [101.04, 412.79, -500.51], [22.11, 509.50, -499.56],
        [-347.88, 451.52, -502.62], [-570.90, -125.14, -504.62],
        [-256.14, -521.08, 496.03], [361.00, -422.83, 499.51],
        [576.37, 170.92, 494.04], [338.44, 463.72, 505.72],
        [101.04, 412.79, 504.42], [22.11, 509.50, 504.66],
        [-347.88, 451.52, 502.01], [-570.90, -125.14, 499.60]
    ]))

CORE_HULL = Delaunay(
    np.array([
        [-32.96, 62.44, -498.50], [90.49, 82.35, -498.50],
        [194.34, -30.92, -498.50], [124.97, -131.25, -498.50],
        [1.71, -150.63, -498.50], [77.80, -54.33, -498.50],
        [-32.96, 62.44, 191.42], [90.49, 82.35, 191.42],
        [194.34, -30.92, 191.42], [124.97, -131.25, 191.42],
        [1.71, -150.63, 191.42], [77.80, -54.33, 191.42]
    ]))

"""
Physical Constants
------------------
* ``EARTH_RADIUS``: Radius of the earth.

* ``ICECUBE_DEPTH``: Depth of the center of the IceCube detector (the origin of the IceCube coordinate system).

* ``DUST_COEFF``: Absorption and scattering coefficients of the surrounding ice.
"""

EARTH_RADIUS = 6367440  # m
ICECUBE_DEPTH = 1950.0  # m

DUST_COEFF = np.array([[1405, 7.5, 11.3],
                       [1415, 7.2, 11.7],
                       [1425, 7.0, 11.4],
                       [1435, 7.1, 11.5],
                       [1445, 4.9, 9.2],
                       [1455, 5.5, 9.8],
                       [1465, 6.0, 10.3],
                       [1475, 5.0, 9.3],
                       [1485, 4.7, 8.9],
                       [1495, 4.3, 8.5],
                       [1505, 4.0, 8.2],
                       [1515, 3.5, 7.6],
                       [1525, 3.5, 7.7],
                       [1535, 4.2, 8.5],
                       [1545, 4.9, 9.2],
                       [1555, 6.2, 10.5],
                       [1565, 7.4, 11.8],
                       [1575, 6.9, 11.3],
                       [1585, 6.3, 10.7],
                       [1595, 6.3, 10.6],
                       [1605, 5.6, 10.0],
                       [1615, 5.5, 9.8],
                       [1625, 5.4, 9.7],
                       [1635, 5.7, 10.0],
                       [1645, 5.4, 9.7],
                       [1655, 4.7, 9.0],
                       [1665, 4.0, 8.2],
                       [1675, 3.3, 7.5],
                       [1685, 3.2, 7.4],
                       [1695, 3.2, 7.3],
                       [1705, 3.7, 7.9],
                       [1715, 4.5, 8.7],
                       [1725, 5.3, 9.6],
                       [1735, 6.3, 10.7],
                       [1745, 6.7, 11.1],
                       [1755, 6.2, 10.6],
                       [1765, 6.1, 10.5],
                       [1775, 5.4, 9.7],
                       [1785, 4.8, 9.1],
                       [1795, 4.6, 8.8],
                       [1805, 4.1, 8.3],
                       [1815, 3.7, 7.9],
                       [1825, 3.6, 7.8],
                       [1835, 3.5, 7.6],
                       [1845, 3.8, 7.9],
                       [1855, 4.1, 8.3],
                       [1865, 4.9, 9.2],
                       [1875, 5.4, 9.7],
                       [1885, 5.6, 9.9],
                       [1895, 5.0, 9.2],
                       [1905, 4.3, 8.5],
                       [1915, 3.6, 7.7],
                       [1925, 3.2, 7.2],
                       [1935, 3.3, 7.4],
                       [1945, 3.4, 7.6],
                       [1955, 3.7, 7.9],
                       [1965, 4.6, 8.8],
                       [1975, 5.4, 9.7],
                       [1985, 5.5, 9.8],
                       [1995, 6.3, 10.7],
                       [2005, 7.4, 11.8],
                       [2015, 10.5, 15.2],
                       [2025, 10.2, 14.9],
                       [2035, 16.3, 21.4],
                       [2045, 17.5, 22.8],
                       [2055, 15.7, 20.8],
                       [2065, 17.9, 23.2],
                       [2075, 16.3, 21.4],
                       [2085, 10.6, 15.3],
                       [2095, 6.9, 11.3],
                       [2105, 4.2, 8.4],
                       [2115, 3.1, 7.2],
                       [2125, 3.2, 7.3],
                       [2135, 4.2, 8.4],
                       [2145, 3.0, 7.1],
                       [2155, 3.3, 7.5],
                       [2165, 3.7, 7.8],
                       [2175, 3.5, 7.6],
                       [2185, 3.4, 7.5],
                       [2195, 4.5, 8.7],
                       [2205, 3.9, 8.1],
                       [2215, 5.9, 10.2],
                       [2225, 3.8, 8.0],
                       [2235, 2.8, 6.9],
                       [2245, 3.0, 7.1],
                       [2255, 4.5, 8.7],
                       [2265, 3.7, 7.9],
                       [2275, 3.0, 7.1],
                       [2285, 2.9, 7.0],
                       [2295, 2.7, 6.8],
                       [2305, 2.6, 6.7],
                       [2315, 2.5, 6.6],
                       [2325, 4.0, 8.2],
                       [2335, 3.3, 7.4],
                       [2345, 3.5, 7.6],
                       [2355, 3.6, 7.7],
                       [2365, 3.4, 7.6],
                       [2375, 3.2, 7.4],
                       [2385, 3.1, 7.2],
                       [2395, 2.9, 7.0],
                       [2405, 2.8, 6.9],
                       [2415, 2.8, 6.9],
                       [2425, 2.8, 6.9],
                       [2435, 2.7, 6.8],
                       [2445, 2.7, 6.7],
                       [2455, 2.6, 6.7],
                       [2465, 2.7, 6.8],
                       [2475, 2.6, 6.7],
                       [2485, 2.7, 6.8],
                       [2495, 2.8, 6.9],
                       [2505, 3.0, 7.2],
                       [2515, 3.4, 7.5],
                       [2525, 3.6, 7.8],
                       [2535, 3.5, 7.7],
                       [2545, 3.4, 7.5],
                       [2555, 3.3, 7.4],
                       [2565, 2.8, 6.9],
                       [2575, 2.5, 6.6],
                       [2585, 2.4, 6.5],
                       [2595, 2.3, 6.3]])

#DUST_COEFF[:, 0] = ICECUBE_DEPTH - DUST_COEFF[:, 0]
#DUST_COEFF[:, :] = DUST_COEFF[::-1, :]

"""
Computational Constants
-----------------------
* ``PE_THRESHOLD``: Number of photo electrons to be detected from a muon in order to be considered observed.

* ``N_ITER``: Number of iterations to determine the travel length.
"""
PE_THRESHOLD = 5
N_ITER = 20
