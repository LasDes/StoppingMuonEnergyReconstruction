"""Event Selector.

Usage: eventSelector.py INPUT_DIR OUT_DIR MASK [--score_threshold=<threshold> --reverse --only_fp]

-h                Show help.
--score_threshold Save events with a score above this threshold. [default: 0.5]
--reverse         Save events with a score lower than the threshold.
--only_fp         Save only false positives.
"""
from docopt import docopt
import numpy as np
from glob import glob
from icecube import icetray, dataclasses, dataio, simclasses, gulliver, linefit, paraboloid, millipede, common_variables
from I3Tray import *

def is_event_selected(frame):
	try:
		event_id = frame["I3EventHeader"].event_id
	except:
		return False
	if event_id in ALL_EVENTS:
		return True
	return False

def main(path_dataset, path_output, path_scores, threshold, reverse, only_fp):
	global ALL_EVENTS
	event_ids, scores, labels = np.load(path_scores)

	mask = (scores > threshold)
	if reverse == True:
		mask = (scores < threshold)
	
	if only_fp == True:
		mask &= (labels != 1.0)

	ALL_EVENTS = event_ids[mask]
	print(np.shape(ALL_EVENTS))

	all_files = glob(path_dataset + "*.bz2")

	tray = I3Tray()
	tray.AddModule("I3Reader", "reader", FilenameList=all_files)

	tray.AddModule(
		"I3Writer",
		"write",
		If=is_event_selected,
		Filename=path_output)

	tray.AddModule("TrashCan", "trash")

	tray.Execute()
	tray.Finish()

if __name__ == "__main__":
	args = docopt(__doc__, version="Main Routine")
	main(args["INPUT_DIR"], args["OUT_DIR"], args["MASK"],
		 float(args["--score_threshold"]), args["--reverse"],
		 args["--only_fp"])