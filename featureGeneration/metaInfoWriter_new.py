"""Meta Info Writer.

Usage: metaInfoWriter.py FILENAME SET

-h --help                 Show this.
FILENAME                  Path to file to be altered.
SET						  IDs of used data set.
"""
import h5py
import numpy as np
from docopt import docopt

#dataset_ids = [11057, 11058, 11499]
#dataset_sizes = [1228410, 323531, 575661]

#datasets = {11057: 98717, 11058: 27849, 11499: 97852}
datasets = {11057: 9872, 11058: 27849, 11374: 1620, 11499: 97852}

def main(filename, sets):
	f = h5py.File(filename, "r+")

	if "MetaInfo" in f.keys():
		del f["MetaInfo"]

	set_list = sets.split(",")

	meta_info = f.create_dataset("MetaInfo", (len(set_list),), dtype=np.dtype([("dataset_id", int), ("dataset_size", int)]))

	meta_info["dataset_id"] = [int(i) for i in set_list]
	meta_info["dataset_size"] = [datasets[int(i)] for i in set_list]

	f.close()

if __name__ == "__main__":
	args = docopt(__doc__, version="Meta Info Writer")
	main(args["FILENAME"], args["SET"])
