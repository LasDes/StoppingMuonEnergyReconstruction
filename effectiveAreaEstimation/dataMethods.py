# # # # # # # # # # # # # # #
# Methods for Data Handling #
# # # # # # # # # # # # # # #
#
# A method to read in hdf5-files. Turns out, using
# pandas makes this incredibly simple
# (And holy shit, fast too)

import numpy as np
import pandas as pd
import h5py

def ensure_safe_float32_cast_(data):
    float32_info = np.finfo("float32")
    data[data > float32_info.max] = float32_info.max
    data[data < float32_info.min] = float32_info.min


def _load_single_name(filename, name):
    clutter = ["Run", "Event", "SubEvent", "SubEventStream", "exists"]
    df = pd.read_hdf(filename, name).drop(clutter, axis=1)
    rename_dict = {}
    for n in df.columns.values:
        if "Hoinka" in n:
            rename_dict[n] = n
        else:
            rename_dict[n] = name + "_" + n
    df = df.rename(index=str, columns=rename_dict)
    return df


def _load_single_file(filename, label_keys, attribute_list=None,
                      verbosity=False):
    if attribute_list is None:
        F = h5py.File(filename)
        names = F.keys()
        F.close()
    else:
        names = attribute_list

    df_att = None
    df_lbl = None
    weight = None
    for name in names[:-1]:
        if verbosity is True:
            print("Reading key %s..." % name)
        if name in label_keys:
            if df_lbl is None:
                df_lbl = _load_single_name(filename, name)
            else:
                df_lbl = df_lbl.join(_load_single_name(filename, name))
        elif name == "MuonWeight":
            weight = pd.read_hdf(filename, name)
        else:
            if df_att is None:
                df_att = _load_single_name(filename, name)
            else:
                df_att = df_att.join(_load_single_name(filename, name))
    return (df_lbl, df_att, weight)


def load_data_beta(filename_list,
                   label_list=["Hoinka_Labels"],
                   attribute_list=None,
                   weights=True,
                   verbosity=False):
    clutter = ["Run", "Event", "SubEvent", "SubEventStream", "exists"]
    if type(filename_list) != list:
        filename_list = [filename_list]

    df_full_list_att = []
    df_full_list_lab = []
    grp = np.empty(0)
    grp_idx = 0

    for filename in filename_list:
        if verbosity is True:
            print("Reading %s..." % filename)
        df_lbl, df_att, weight = _load_single_file(filename,
                                                           label_list,
                                                           attribute_list,
                                                           verbosity)
        df_full_list_att += [df_att]
        df_full_list_lab += [df_lbl]
        grp = np.append(grp, grp_idx * np.ones(len(df_att), dtype=int))
        grp_idx += 1

    lbl = pd.concat(df_full_list_lab).fillna(0)
    att = pd.concat(df_full_list_att).fillna(0)

    if weights is True:
        wgt = weight.value.values
    else:
        wgt = np.ones(len(att))

    ensure_safe_float32_cast_(lbl)
    ensure_safe_float32_cast_(att)

    return lbl, att, wgt, grp
