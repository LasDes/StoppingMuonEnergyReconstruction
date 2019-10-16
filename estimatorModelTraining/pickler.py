"""Pickling HDF5-Files (Seriously, that's all that happens here!)

Usage: pickler.py -i <input> -o <output> [--exp]

-h --help                     Show this.
-i <input>                    Input.
-o <output>                   Output.
--exp			      Flag for experimental data
"""
from dataMethods import load_data_beta as load_data
from sklearn.externals import joblib
from docopt import docopt
import glob

if __name__ == "__main__":
    args = docopt(__doc__)
    files = glob.glob(args["-i"])
    if args["--exp"] == False:
        label, attribute, weight, group = load_data(files, weights=False)
        joblib.dump((label, attribute, weight, group), args["-o"])
    else:
        _, attribute, _, _ = load_data(files, weights=False)
        joblib.dump(attribute, args["-o"])
