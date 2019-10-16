"""HD5Writer.
Usage: HD5Writer.py INPUT OUTPUT [--L3]

-h --help  Show this screen.
INPUT      Input path in bash script.
OUTPUT     Output path to HDF5 file
"""

import sys
import os
import glob
from docopt import docopt

import numpy as np
from icecube import icetray, dataclasses, dataio, simclasses, gulliver, linefit, paraboloid, millipede, common_variables, cramer_rao
from I3Tray import *
from icecube.tableio import I3TableWriter
from icecube.hdfwriter import I3SimHDFWriter
from icecube.hdfwriter import I3HDFTableService


def main(input, output, l3):
    keys = ["Hoinka_Labels", "MuonWeight"]

    tray = I3Tray()

    tray.AddModule("I3Reader", "reader", FilenameList=glob.glob(input))

    print("Reading from %s" % (input))

    if(l3):
        tray.AddModule(
            I3TableWriter,
            "writer",
            tableservice=I3HDFTableService(output),
            keys=keys,
            SubEventStreams=["Final"])
    else:
        tray.Add(I3SimHDFWriter,
        Output=output,
        Keys=keys,
        Types=[],
        )

    tray.AddModule("TrashCan", "trash")
    tray.Execute()
    tray.Finish()


if __name__ == "__main__":
    args = docopt(__doc__, version="HD5 Writer")
    main(args["INPUT"], args["OUTPUT"], args["--L3"])
