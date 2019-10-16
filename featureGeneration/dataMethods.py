# # # # # # # # # # # # # # #
# Methods for Data Handling #
# # # # # # # # # # # # # # #
#
# Here are some methods to handle data. Those methods are:
# a) load_data_single(filename)
# Takes a hdf5 file and returns a pandas dataframe
# b) load_data(filename_list)
# Takes a whole filename list, puts each of them into pandas dataframes and if
# compatible, returns a concatenated data frame
#
# 2016 T. Hoinka (tobias.hoinka@udo.edu)

import h5py
import numpy as np
import pandas as pd

def load_data_single(filename, show_status=True,
                     label_list=["HoinkaLabels",
                                 "MCPrimary1",
                                 "MCMostEnergeticInIce"],
                     attribute_list=["HoinkaAttributes"]):
    file = h5py.File(filename)

    n_data = len(file[attribute_list[0]])
    lbl = True
    try:
        n_lbl = len(file[label_list[0]][0]) - 5
    except:
        lbl = False
        n_lbl = 0
    n_att = len(file[attribute_list[0]][0]) - 5
    if show_status is True:
        print("%s: %i frames, %i labels, %i attributes." % (filename,
                                                            n_data,
                                                            n_lbl,
                                                            n_att))
    df_lbl = pd.DataFrame()
    for label in label_list:
        for key in file[label].value[0].dtype.names[5:]:
            df_lbl[key] = file[label][key]

    df_att = pd.DataFrame()
    for attr in attribute_list:
        for key in file[attr].value[0].dtype.names[5:]:
            df_att[key] = file[attr][key]

    return df_lbl.fillna(0), df_att.fillna(0)

def load_data(filename_list,
              label_list=["HoinkaLabels",
                          "MCPrimary1",
                          "MCMostEnergeticInIce"],
              attribute_list=["HoinkaAttributes"]):
    df_lbl_list = []
    df_att_list = []
    for file in filename_list:
        L, D = load_data_single(file, label_list=label_list,
                                attribute_list=attribute_list)
        df_lbl_list += [L]
        df_att_list += [D]

    df_lbl_all = pd.concat(df_lbl_list)
    df_att_all = pd.concat(df_att_list)
    shuffle = np.random.permutation(len(df_lbl_all))

    return df_lbl_att.iloc[shuffle], df_att_all.iloc[shuffle]