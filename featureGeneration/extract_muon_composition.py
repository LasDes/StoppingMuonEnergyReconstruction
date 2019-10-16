"""Extracts event properties from MC truth regarding muon multiplicity.

Usage: extract_muon_composition.py -i <input> -o <output>

-h --help          Show this.
-i <input>         Input files
-o <output>        Output file
"""
import numpy as np
import pandas as pd
from docopt import docopt
from glob import glob
from icecube import icetray, dataclasses, dataio, simclasses, gulliver, linefit, paraboloid, millipede, common_variables
from I3Tray import *
from topologyMethods import get_muon_properties, decide_label, pe_count

def extract_from_frame(frame):
	pe_counts = pe_count(frame["I3MCPESeriesMap"])
	muon_bunches = get_muon_properties(frame["I3MCTree"], pe_counts)
	primaries = np.unique(muon_bunches[:,11])
	_, lab = decide_label(muon_bunches)
	output = np.zeros((len(primaries), 7))
	for p, i  in zip(np.unique(muon_bunches[:,11]), range(len(primaries))):
		sel = muon_bunches[:,11] == p
		output[i,:] = np.array([lab,
			                    np.sum(sel),
			                    np.sum(muon_bunches[sel,9]),
								np.min(muon_bunches[sel,9]),
								np.max(muon_bunches[sel,9]),
								np.mean(muon_bunches[sel,9]),
								np.median(muon_bunches[sel,9])])
	return output
	

def extract_from_file(filename):
	F = dataio.I3File(filename)
	frame = F.pop_physics()
	muon_bunches = None
	while frame:
		if muon_bunches is None:
			muon_bunches = extract_from_frame(frame)
		else:
			muon_bunches = np.vstack((muon_bunches, extract_from_frame(frame)))
		frame = F.pop_physics()
	return muon_bunches

if __name__ == "__main__":
	args = docopt(__doc__)
	input_files = glob(args["-i"])
	muon_list = None
	for f in input_files:
		print(f)
		if muon_list is None:
			muon_list = extract_from_file(f)
		else:
			muon_list = np.vstack((muon_list, extract_from_file(f)))
	cols = ["stopping", "N", "E_tot", "E_min", "E_max", "E_mean", "E_med"]
	df = pd.DataFrame(muon_list,
		              columns=cols)
	df.to_csv(args["-o"], index=False)

