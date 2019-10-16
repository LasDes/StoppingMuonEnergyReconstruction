#!/bin/sh /cvmfs/icecube.opensciencegrid.org/py2-v3.0.1/icetray-start
# METAPROJECT combo

"""labelWriterMGS.

Usage: labelWriterMGS INPUT OUTPUT GEOMETRY ID N [--sim --multi]
"""

from math import sqrt
import os
import sys
import glob

import numpy as np
import scipy.spatial as spatial

from icecube import icetray, dataclasses, dataio, simclasses, gulliver
from icecube import linefit, paraboloid, millipede, common_variables, cramer_rao
from I3Tray import *

from classDefinitions import Trajectory, Feature, ExtractedFeatures, DetectorGeometry, VetoRegion
from globalMethods import *
from topologyMethods import get_muon_properties, decide_label, get_coincidence, pe_count, visited_muons

from docopt import docopt

#==============================================================================
# GetLabels
#==============================================================================
class GetLabels(icetray.I3ConditionalModule):
    def __init__(self, context):
        icetray.I3ConditionalModule.__init__(self, context)
        self.AddParameter(
            "Features",
            "Object from class ExtractedFeatures to contain extracted features.",
            ExtractedFeatures())

    def add(self, key, value, role="label"):
        self.feature.add(key, value, role)

    def Configure(self):
        print("GetLabels Starting...")
        self.feature = self.GetParameter("Features")
        self.frame_index = 0
        self.file_index = 0

    def DAQ(self, frame):
        mep_t = dataclasses.get_most_energetic_primary(frame[
            "I3MCTree"])
        traj = Trajectory(
            mep_t.dir.x, mep_t.dir.y,
            mep_t.dir.z, mep_t.pos.x,
            mep_t.pos.y, mep_t.pos.z)
        #self.add("L5Prob", frame["L5Prob"].value)
        self.add("weight", frame["MuonWeight"].value)
        self.add("zenith_true", mep_t.dir.zenith)
        self.add("azimuth_true", mep_t.dir.azimuth)
        self.add("energy_mep", mep_t.energy)

        try:
            pe_counts = pe_count(frame["I3MCPESeriesMap"])
        except:
            pe_counts = 0
        muon_bunches = get_muon_properties(frame["I3MCTree"], pe_counts)

        self.add("coincidence", get_coincidence(muon_bunches))
        visited = visited_muons(muon_bunches)
        stopping = muon_bunches[:, 15] == True
        energy_total = np.mean(muon_bunches[:, 9])
        if np.sum(visited) > 0:
            stopr = np.mean(muon_bunches[visited, 12])
            stopz = np.mean(muon_bunches[visited, 13])
            nmust = np.sum(muon_bunches[visited, 15])
            energy_stop = np.mean(muon_bunches[stopping, 9])
            stop_det, stop_dc = decide_label(muon_bunches)
        else:
            stopr = NaN
            stopz = NaN
            nmust = 0
            energy_stop = NaN
            stop_det = False
            stop_dc = False
        self.add("true_stop_r", stopr)
        self.add("true_stop_z", stopz)
        self.add("energy_stop", energy_stop)
        self.add("energy_total", energy_total)
        self.add("n_mu", len(muon_bunches))
        self.add("n_mu_stop", nmust)
        self.add("label_det", stop_det)
        self.add("label_in", stop_dc)
        self.add("frame_index", self.frame_index)
        self.frame_index += 1
        self.PushFrame(frame)

    def Simulation(self, frame):
        self.PushFrame(frame)

    def Finish(self):
        print("Finished GetLabels.")

#==============================================================================
# AddFeaturesToI3
#==============================================================================
class AddFeaturesToI3(icetray.I3ConditionalModule):
    def __init__(self, context):
        icetray.I3ConditionalModule.__init__(self, context)
        self.AddParameter("Features", "Feature Class Object.")

    def Configure(self):
        self.feature = self.GetParameter("Features")

    def DAQ(self, frame):
        labels, attributes = self.feature.get_row()
        frame["Hoinka_Labels"] = dataclasses.I3MapStringDouble(labels)
        self.PushFrame(frame)

    def Simulation(self, frame):
        self.PushFrame(frame)

    def Finish(self):
        print("AddFeaturesToI3 Finished.")


CUT = True

def isCut(frame):
    if CUT is True:
        return True
    if "BelowCuts" in frame:
        return frame["BelowCuts"] > 0.5
    else:
        return False


# Main extraction method.
def labelMGS(input_file, output_path, geo_file, dataset_id, n_files,
                    sim=True, multi=False):
    extr_feat = ExtractedFeatures()
    tray = I3Tray()

    if multi == True:
        files = glob.glob(input_file + "*.bz2")
    else:
        files = [input_file]

    # Reads data frame-wise
    tray.AddModule("I3Reader", "reader", FilenameList=files)

    if sim is True:
        # Calculates labels from data, puts them into extr_feat
        tray.AddModule(
            GetLabels,
            "getlabels",
            Features=extr_feat,
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
                       SizeLimit=10 * 1024 ** 2)

    tray.AddModule("TrashCan", "trash")
    tray.Execute()
    tray.Finish()


def main(arg):
    geometry = DetectorGeometry(arguments["GEOMETRY"])
    print(arg["INPUT"])
    labelMGS(arg["INPUT"],
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
