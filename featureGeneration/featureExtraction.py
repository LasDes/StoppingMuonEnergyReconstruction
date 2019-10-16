#!/bin/sh /cvmfs/icecube.opensciencegrid.org/py2-v3.0.1/icetray-start
#METAPROJECT combo

"""Feature Extraction.

Usage: featureExtraction INPUT OUTPUT GEOMETRY ID N [--sim --multi]
"""
from math import sqrt
import os
import sys
import glob

import numpy as np
from scipy.spatial import ConvexHull

from icecube import icetray, dataclasses, dataio, simclasses, gulliver, linefit, paraboloid, millipede, common_variables
from I3Tray import *
from icecube.weighting import CORSIKAWeightCalculator, fluxes

from classDefinitions import Trajectory, Feature, ExtractedFeatures, DetectorGeometry, VetoRegion
import constantDefinitions as CONST
from icetrayModules import *

from docopt import docopt

CUT = True

def isCut(frame):
    if CUT is True:
        return True
    if "BelowCuts" in frame:
        return frame["BelowCuts"] > 0.5
    else:
        return False

# Main extraction method.
def extractFeatures(input_file, output_path, geo_file, dataset_id, n_files,
                    sim=True, multi=False):
    extr_feat = ExtractedFeatures()
    tray = I3Tray()

    if multi == True:
        files = glob.glob(input_file + "*.bz2")
    else:
        files = [input_file]

    # Reads data frame-wise
    tray.AddModule("I3Reader", "reader", FilenameList=files)

    tray.AddModule(PreCuts, "PreCuts", Geometry=geo_file)
    
    if sim is True:
        # Calculates labels from data, puts them into extr_feat
        tray.AddModule(
            GetLabels,
            "getlabels",
            Features=extr_feat,
            If=isCut
            )

    # Calculates further attributes, puts them into extr_feat
    tray.AddModule(
        GetAttributes,
        "getattributes",
        Features=extr_feat,
        PulseKeyList=["InIcePulses", "HVInIcePulses", "TWSRTHVInIcePulses"],
            Geometry=geo_file,
        FitKey="SplineMPE",
        ParameterKey="SplineMPEFitParams",
        If=isCut
        )

    tray.AddModule(
        AddFeaturesToI3, "addfeatures", Features=extr_feat,
        If=isCut
        )
    if multi == False:
        tray.AddModule("I3Writer", "write", Filename=output_path, If=isCut)
    else:
        tray.AddModule("I3MultiWriter", "write",
                       Filename=output_path + "%06u.i3.bz2", If=isCut,
                       SizeLimit=10 * 1024**2)

    tray.AddModule("TrashCan", "trash")
    tray.Execute()
    tray.Finish()


def main(arg): 
    geometry = DetectorGeometry(arguments["GEOMETRY"])
    print(arg["INPUT"])
    extractFeatures(arg["INPUT"],
                    arg["OUTPUT"],
                    DetectorGeometry(arg["GEOMETRY"]),
                    arg["ID"],
                    arg["N"],
                    arg["--sim"],
                    arg["--multi"])

if __name__ == "__main__":
    print("Started.")
    arguments = docopt(__doc__)
    main(arguments)
