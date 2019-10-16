"""
.. module:: class_definitions
    :synopsis: Class Definitions.
.. moduleauthor:: Tobias Hoinka <tobias.hoinka@udo.edu>
"""
import numpy as np
from math import sqrt
import constantDefinitions as CONST
from globalMethods import *
from scipy.spatial import Delaunay

from icecube import icetray, dataclasses, dataio, simclasses, gulliver, linefit, paraboloid, millipede, common_variables
from I3Tray import *
from icecube.weighting import CORSIKAWeightCalculator
from icecube.weighting import fluxes

# # # # # # # # # #
# General Classes #
# # # # # # # # # #

class Feature:
    """This class contains a single feature using a list.

    Attributes
    ----------
    feature : object
              Object stored in this feature.

    name : str
           Name of the feature.

    index : int
            Index number of the feature.

    role : {"attribute", "label"}
           Role of the feature.
    
    Methods
    -------
    get()
        Returns the content of the feature.
    """
    def __init__(self, name, index, data, role):
        self.feature = data
        self.name = name
        self.index = index
        self.role = role

    def get(self):
        return self.feature


class ExtractedFeatures:
    """Concatenation of Feature objects.

    Attributes
    ----------
    feature_dict : dict
                   Dictionary of Feature objectds.

    number_features : int
                      Number of features.

    Methods
    -------
    add(key, data, role="attribute")
        Adds a Feature to the feature dictionary.

    print_stats()
        Prints the contents of this object.

    get(feature)
        Gets the contents of a certain feature inside this object.

    get_row()
        Gets the last row.
    """
    def __init__(self):
        self.feature_dict = {}
        self.number_features = 0

    def __str__(self):
        return "ExtractedFeatures Object"

    def __repr__(self):
        return "<ExtractedFeatures Object>"

    # Adds data to a Feature identified by key.
    def add(self, key, data, role="attribute"):
        self.feature_dict[key] = Feature(key, self.number_features, data,
                                          role)

    # Print the lengths of feature lists with their name. For debugging mainly.
    def print_stats(self):
        for feature in self.feature_dict.values():
            print("%s, %i" % (feature.name, feature.length))

    def get(self, feature):
        return self.feature_dict[feature].get()

    def get_row(self):
        row_label = {}
        row_attr = {}
        for key, feature in self.feature_dict.iteritems():
            if feature.role == "label":
                row_label[key] = float(feature.get())
            if feature.role == "attribute":
                row_attr[key] = float(feature.get())
        return (row_label, row_attr)


class Trajectory:
    """Class that contains a parametrization of a trajectory.

    Attributes
    ----------
    v : array, shape = [3,]
        Directional vector of the trajectory.

    w : array, shape = [3,]
        Pivot vector of the trajectory.

    zenith : float
             Zenith angle corresponding to the trajectory.

    Methods
    -------
    eval(t)
        Evaluates the position for a given position t.

    get_distance(p)
        Calculates the distance of a point p to the trajectory.

    closest_approach()
        Calculates the distance of the origin to the trajectory.

    project_onto(p)
        Projects a point p onto the trajectory.

    get_intersection(start, end)
        Performs a bisection algorithm with start and end as initial values.

    travel_length()
        Calculates the travel length of a trajectory.
    """
    def __init__(self, vx, vy, vz, wx, wy, wz, zenith=0.0):
        self.v = np.array([vx, vy, vz])
        self.w = np.array([wx, wy, wz])
        self.zenith = zenith

    def __str__(self):
        return "Trajectory Object"

    def __repr__(self):
        return "<Trajectory Object>"

    def eval(self, t):
        return self.v * t + self.w

    def get_distance(self, p):
        return sqrt(np.sum(np.cross(p - self.w, self.v)**2))

    def closest_approach(self):
        return self.get_distance(np.array([0, 0, 0]))

    def project_onto(self, p):
        return np.sum((p - self.w) * self.v)

    def get_intersection(self, start, end):
        t1 = start
        t2 = end
        t = (t1 + t2) / 2.0
        for k in range(CONST.N_ITER):
            if CONST.DET_HULL.find_simplex(self.eval(t)) >= 0:
                t2 = t
            else:
                t1 = t
            t = (t1 + t2) / 2.0
        return t

    def travel_length(self):
        start_t = -1000.0
        end_t = 1000.0
        middle_t = 0.0
        while (CONST.BARE_DET_HULL.find_simplex(self.eval(middle_t)) >= 0):
            if (self.v[2] > 0.0):
                middle_t -= 10.0
            else:
                middle_t += 10.0
        return (self.get_intersection(start_t, middle_t),
                self.get_intersection(end_t, middle_t))


class DetectorGeometry:
    """Class that contains the geometry of the detector.

    Attributes
    ----------
    table : array, shape = [n_doms_per_string, n_strings, 3]
            Table that contains the position of every dom.

    G : I3Geometry object
        I3Geometry object.

    Methods
    -------
    get_pos(om, string)
        Returns the position of a dom.
    """
    def __init__(self, file):
        self.table = np.zeros((CONST.N_DOMS_PER_STRING, CONST.N_STRINGS, 3))
        File = dataio.I3File(file)
        frame = File.pop_frame()
        while ('I3Geometry' not in frame.keys()):
            frame = File.pop_frame()
            if frame == None:
                print("Couldn't find proper Geometry frame.")
                break

        self.G = frame['I3Geometry']
        geometry = frame['I3Geometry'].omgeo.items()
        for omkey, geo in geometry:
            if omkey.om < 61:
                self.table[int(omkey.om - 1), int(omkey.string - 1),
                           0] = geo.position.x
                self.table[int(omkey.om - 1), int(omkey.string - 1),
                           1] = geo.position.y
                self.table[int(omkey.om - 1), int(omkey.string - 1),
                           2] = geo.position.z
        self.table += np.random.randn(CONST.N_DOMS_PER_STRING, CONST.N_STRINGS,
                                      3) / 10.0

    def __str__(self):
        return "DetectorGeometry Object"

    def __repr__(self):
        return "<DetectorGeometry Object>"

    def get_pos(self, om, string):
        return self.table[int(om - 1), int(string - 1), :]


# This class contains a bitmask to declare veto (or whatever) regions.
class VetoRegion:
    """Contains a bitmask corresponding to a veto region.

    Attributes
    ----------
    table : array, shape = [n_doms_per_string, n_strings]
            boolean mask.

    name : str
           The name of the veto region.

    Methods
    -------
    make_shell(om_list, string_list, value)
        Changes the doms corresponding to a shell region given by om_list and string_list to value.

    make_corridor(trajectory, geometry, radius, value)
        Generates a corridor region given a trajectory and a radius.

    separate(trajectory, geometry, incoming)
        Isolates either the incoming or outgoing part of the detector depending on whether incoming is set to true or not.

    check_intersect_mask(om, string, mask):
        Checks whether a dom is in the bitmask given by this object AND given by a mask.

    check_mask(om, string)
        Checks whether a dom is in the bitmask given by this object.
    """
    def __init__(self, name, default):
        if default == True:
            self.table = np.ones(
                (CONST.N_DOMS_PER_STRING, CONST.N_STRINGS), dtype=bool)
        else:
            self.table = np.zeros(
                (CONST.N_DOMS_PER_STRING, CONST.N_STRINGS), dtype=bool)

        self.name = name

    # Generates a shell region from a list of oms and strings.
    def make_shell(self, om_list, string_list, value):
        if value == True:
            self.table = np.zeros(
                (CONST.N_DOMS_PER_STRING, CONST.N_STRINGS), dtype=bool)
        else:
            self.table = np.ones(
                (CONST.N_DOMS_PER_STRING, CONST.N_STRINGS), dtype=bool)
        for string in string_list:
            self.table[:, string - 1] = value
        for om in om_list:
            self.table[om - 1, :] = value
        for string in CONST.DEEP_CORE_STRINGS:
            self.table[0:30, string - 1] = False

    # Generates a corridor, meaning the set of all doms which are closer to a
    # given trajectory than radius.
    def make_corridor(self, trajectory, geometry, radius, value):
        if value == True:
            self.table = np.zeros(
                (CONST.N_DOMS_PER_STRING, CONST.N_STRINGS), dtype=bool)
        else:
            self.table = np.ones(
                (CONST.N_DOMS_PER_STRING, CONST.N_STRINGS), dtype=bool)
        for om in range(CONST.N_DOMS_PER_STRING):
            for string in range(CONST.N_STRINGS):
                if trajectory.get_distance(geometry.get_pos(
                        om, string)) < radius:
                    self.table[int(om - 1), int(string - 1)] = value

    # Zeros either the outcoming half of the detector (incoming = True) or the
    # incoming half (incoming = False).
    def separate(self, trajectory, geometry, incoming):
        if incoming == False:
            for om in range(CONST.N_DOMS_PER_STRING):
                for string in range(CONST.N_STRINGS):
                    if np.dot(geometry.table[int(om - 1), int(string - 1)],
                              trajectory.v) / sqrt(np.sum(
                                  trajectory.v**2)) * np.sum(self.table[int(
                                      om - 1), int(string - 1)]**2) < 0.0:
                        self.table[int(om - 1), int(string - 1)] = 0
        else:
            for om in range(CONST.N_DOMS_PER_STRING):
                for string in range(CONST.N_STRINGS):
                    if np.dot(geometry.table[int(om - 1), int(string - 1)],
                              trajectory.v) / sqrt(np.sum(
                                  trajectory.v**2)) * np.sum(self.table[int(
                                      om - 1), int(string - 1)]**2) >= 0.0:
                        self.table[int(om - 1), int(string - 1)] = 0

    def __str__(self):
        return "VetoRegion Object"

    def __repr__(self):
        return "<VetoRegion Object>"

    # Checks whether a dom is inside two veto regions.
    def check_intersect_mask(self, om, string, mask):
        return self.table[om - 1, string - 1] and mask.table[om - 1,
                                                                string - 1]

    # Checks whether a dom is in the veto region.
    def check_mask(self, om, string):
        return self.table[om - 1, string - 1]
