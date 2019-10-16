"""Meta Info Writer.

Usage: metaInfoWriter.py FILENAME [--dataset_id=<id> --dataset_size=<size>]

-h --help                 Show this.
FILENAME                  Path to file to be altered.
--dataset_id=<id>         ID of the datset used.
--dataset_size=<size>     Number of files processed in datset. 
"""
import h5py
import numpy as np
from docopt import docopt

def main(filename, dataset_id, dataset_size):
	f = h5py.File(filename, "r+")

	if "MetaInfo" in f.keys():
		del f["MetaInfo"]

	meta_info = f.create_dataset("MetaInfo", (1,),
	                             dtype=np.dtype([("dataset_id", int),
	                                             ("dataset_size", int)]))
	meta_info["dataset_id"] = dataset_id
	meta_info["dataset_size"] = dataset_size

	f.close()

if __name__ == "__main__":
	args = docopt(__doc__, version="Meta Info Writer")
	main(args["FILENAME"], args["--dataset_id"], args["--dataset_size"])
