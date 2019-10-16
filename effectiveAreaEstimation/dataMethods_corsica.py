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
from weighting import compoundWeightGenerator

def ensure_safe_float32_cast_(data):
    float32_info = np.finfo("float32")
    data[data > float32_info.max] = float32_info.max
    data[data < float32_info.min] = float32_info.min

def calc_weights(id_list, size_list, energy, ptype):
    """Calculates the weights correctly, i.e. all sets are weighted
    together.

    Parameters
    ----------
    id_list : list(integer)
              A list of IDs of the used datasets.

    size_list : list(integer)
                A list of the number of samples of each dataset.

    energy : array, shape = [n_samples,]
             The primary energies of the frames.

    ptype : array, shape = [n_samples,]
            The pdg ids of the primaries.

    Returns
    -------
    weights : [n_samples,]
              The weights of each event.
    """
    gen = compoundWeightGenerator()
    for i in range(len(id_list)):
        gen.add_generator("./gen_%d.pickle" % id_list[i], id_list[i],
                          size_list[i])
    weights = gen.get_weight(energy, ptype)
    return weights

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
    meta = None
    for name in names[:-1]:
        if verbosity is True:
            print("Reading key %s..." % name)
        if name in label_keys:
            if df_lbl is None:
                df_lbl = _load_single_name(filename, name, intervall)
            else:
                df_lbl = df_lbl.join(_load_single_name(filename, name, intervall))
        elif name == "MetaInfo":
            meta = pd.DataFrame(filename[name].value)
        else:
            if df_att is None:
                df_att = _load_single_name(filename, name, intervall)
            else:
                df_att = df_att.join(_load_single_name(filename, name, intervall))
    if meta is None:
        try:
            meta = pd.DataFrame(filename['MetaInfo'].value)
        except:
            print("You may want to add MetaInfo to your dataset.")
            meta = pd.DataFrame({"dataset_id" :   [-1],
                                 "dataset_size" : [-1]})
    return (df_lbl, df_att,
            meta["dataset_id"].tolist(), meta["dataset_size"].tolist())

def load_data(filename_list,
              intervall,
              label_list=["Hoinka_Labels", "CorsikaWeightMap", "I3MCWeightDict"],
              attribute_list=None,
              weights=True,
              verbosity=False,
              modifier=None):

    if modifier is 'nu':
        print('---nu generator weighting in effect---')

    clutter = ["Run", "Event", "SubEvent", "SubEventStream", "exists"]
    if type(filename_list) != list:
        filename_list = [filename_list]

    if verbosity is True:
        print("Reading %s..." % filename_list[0])
    lbl, att, mc_id, mc_size = _load_single_file(filename_list[0], intervall,
                                                       label_list,
                                                       attribute_list,
                                                       verbosity)

    lbl.fillna(0, inplace=True)
    att.fillna(0, inplace=True)

    float32_info = np.finfo("float32")
    lbl.clip(lower=float32_info.min, upper=float32_info.max, inplace=True)
    att.clip(lower=float32_info.min, upper=float32_info.max, inplace=True)
    # ensure_safe_float32_cast_(lbl)
    # ensure_safe_float32_cast_(att)

    grp = np.zeros(len(att), dtype=int)

    if weights is True:
        if modifier is 'nu':
            wgt = lbl.I3MCWeightDict_TotalWeight.as_matrix()
        else:
            energy = lbl.CorsikaWeightMap_PrimaryEnergy.as_matrix()
            ptype = lbl.CorsikaWeightMap_PrimaryType.as_matrix()
            wgt = calc_weights(mc_id, mc_size, energy, ptype)
    else:
        wgt = np.ones(len(att))

    return lbl, att, wgt, grp
