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


def _load_single_name(filename, name, intervall):
    clutter = ["Run", "Event", "SubEvent", "SubEventStream", "exists"]
    df = pd.DataFrame(filename[name][intervall[0]:intervall[1]]).drop(clutter, axis=1)
    rename_dict = {}
    for n in df.columns.values:
        if "Hoinka" in n:
            rename_dict[n] = n
        else:
            rename_dict[n] = name + "_" + n
    df = df.rename(index=str, columns=rename_dict)
    return df

def _load_single_file(filename, intervall, label_keys, attribute_list=None,
                      verbosity=False):
    if attribute_list is None:
        names = filename.keys()
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
                df_lbl = _load_single_name(filename, name, intervall)
            else:
                df_lbl = df_lbl.join(_load_single_name(filename, name, intervall))
        elif name == "MuonWeight":
            weight = _load_single_name(filename, name, intervall)
        else:
            if df_att is None:
                df_att = _load_single_name(filename, name, intervall)
            else:
                df_att = df_att.join(_load_single_name(filename, name, intervall))
    return (df_lbl, df_att, weight)


def load_data(filename_list, intervall,
                   label_list=["Hoinka_Labels",
                               "CorsikaWeightMap"],
                   attribute_list=None,
                   weights=True,
                   verbosity=False):
    clutter = ["Run", "Event", "SubEvent", "SubEventStream", "exists"]
    if type(filename_list) != list:
        filename_list = [filename_list]

    if verbosity is True:
        print("Reading %s..." % filename_list[0])
    lbl, att, weight = _load_single_file(filename_list[0], intervall,
                                                       label_list,
                                                       attribute_list,
                                                       verbosity)

    grp = np.empty(0)

    lbl.fillna(0, inplace=True)
    ensure_safe_float32_cast_(lbl)

    try:
        att.fillna(0, inplace=True)
        ensure_safe_float32_cast_(att)
    except:
        pass

    if weights is True:
        try:
            wgt = weight.values
        except:
            wgt = np.ones(len(lbl))
    else:
        wgt = np.ones(len(lbl))

    return lbl, att, wgt, grp
