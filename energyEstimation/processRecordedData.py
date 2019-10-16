"""processRecordedData.
Usage: processRecordedData.py DATA MODELS OUTPUT BATCHES

-h --help  Show this screen.
DATA       Input path to HDF5 file containing (level 4) corsica data.
MODELS     Input path to pickle file containing models.
OUTPUT     Output path.
BATCHES    Number of batches to split up input.
"""

import numpy as np
import pandas as pd
import h5py
from docopt import docopt
from sklearn.externals import joblib
from dataMethods_v3 import load_data


cut = 87.0 / 180 * np.pi


def main(input, model_path, output, n_batches):
    print("*****Reading recorded data from")

    # read data and write to df
    df_list = []

    for f in input.split(","):
        print("Loading data from %s ..." % f)

        file = h5py.File(f)
        n_input_lines = file['Hoinka_Labels'].size

        steps = np.linspace(0, n_input_lines, num=n_batches + 1).astype(int)

        intervals = [(steps[i], steps[i + 1]) for i in range(len(steps) - 1)]

        for n, batch in enumerate(intervals):
            print("...Processing batch %i" % n)
            _, att, _, _ = load_data(file, batch, verbosity=False, weights=False)

            models = joblib.load(model_path)

            proba_s = models['s'][1].predict_proba(att[models['s'][0]])[:, 1]
            estimate_q = models['q'][1].predict(att[models['q'][0]])
            proba_m = models['m'][1].predict_proba(att[models['m'][0]])[:, 1]
            estimate_r = models['r'][1].predict(att[models['r'][0]])

            df = pd.DataFrame({'single_stopping': proba_m,
                                'quality': estimate_q,
                                'zenith': att["Hoinka_zenith_SplineMPE"],
                                'stop_z': estimate_r})

            df_list += [df]

    result = pd.concat(df_list).reset_index()

    result['zenith_cos'] = np.cos(result.zenith)

    # store readout
    joblib.dump(result, "%s/df_data.pickle" % output)

    print("*****Finished Succesfull!")


if __name__ == "__main__":
    args = docopt(__doc__)
    main(args["DATA"], args["MODELS"], args["OUTPUT"], int(args["BATCHES"]))