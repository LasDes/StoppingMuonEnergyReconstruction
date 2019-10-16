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
        gen.add_generator("./gen_%d.pickle" % id_list[i][0], id_list[i][0],
                          size_list[i][0])
    weights = gen.get_weight(energy, ptype)
    return weights

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
    meta = None
    for name in names[:-1]:
        if verbosity is True:
            print("Reading key %s..." % name)
        if name in label_keys:
            if df_lbl is None:
                df_lbl = _load_single_name(filename, name)
            else:
                df_lbl = df_lbl.join(_load_single_name(filename, name))
        elif name == "MetaInfo":
            meta = pd.read_hdf(filename, name)
        else:
            if df_att is None:
                df_att = _load_single_name(filename, name)
            else:
                df_att = df_att.join(_load_single_name(filename, name))
    if meta is None:
        try:
            meta = pd.read_hdf(filename, "MetaInfo")
        except:
            print("You may want to add MetaInfo to your dataset.")
            meta = pd.DataFrame({"dataset_id" :   [-1],
                                 "dataset_size" : [-1]})
    return (df_lbl, df_att,
            meta["dataset_id"].as_matrix(), meta["dataset_size"].as_matrix())

def load_data_alpha(filename_list,
                   label_list=["Hoinka_Labels",
                               "CorsikaWeightMap"],
                   attribute_list=None,
                   weights=True,
                   verbosity=False):

    clutter = ["Run", "Event", "SubEvent", "SubEventStream", "exists"]

    if type(filename_list) != list:
        filename_list = [filename_list]

    if attribute_list is None:
        F = h5py.File(filename_list[0])
        names = F.keys()
        F.close()
    else:
        names = attribute_list

    grp = np.empty(0)

    col_att = []
    col_lbl = []
    meta = None
    lbl = None
    att = None

    for name in names[:-1]:
        if verbosity is True:
            print("Reading key %s..." % name)
        if name in label_list:
            col_lbl = [_load_single_name(filename, name) for filename in filename_list]
            if lbl is None:
                lbl = pd.concat(col_lbl)
            else:
                foo = pd.concat(col_lbl)
                lbl = lbl.join(foo)
        elif name == "MetaInfo":
            pass
        else:
            col_att = [_load_single_name(filename, name) for filename in filename_list]
            if att is None:
                att = pd.concat(col_att)
            else:
                foo = pd.concat(col_att)
                att = att.join(foo)

    if weights is True:
        energy = lbl.CorsikaWeightMap_PrimaryEnergy.as_matrix()
        ptype = lbl.CorsikaWeightMap_PrimaryType.as_matrix()
        wgt = calc_weights(id_list, size_list, energy, ptype)
    else:
        wgt = np.ones(len(att))

    lbl = pd.concat(df_full_list_lab).fillna(0)
    att = pd.concat(df_full_list_att).fillna(0)
    ensure_safe_float32_cast_(lbl)
    ensure_safe_float32_cast_(att)

    return lbl, att, wgt, grp

def load_data_beta(filename_list,
                   label_list=["Hoinka_Labels",
                               "CorsikaWeightMap"],
                   attribute_list=None,
                   weights=True,
                   verbosity=False):
    clutter = ["Run", "Event", "SubEvent", "SubEventStream", "exists"]
    if type(filename_list) != list:
        filename_list = [filename_list]

    df_full_list_att = []
    df_full_list_lab = []
    id_list = []
    grp = np.empty(0)
    size_list = []
    grp_idx = 0

    for filename in filename_list:
        if verbosity is True:
            print("Reading %s..." % filename)
        df_lbl, df_att, mc_id, mc_size = _load_single_file(filename,
                                                           label_list,
                                                           attribute_list,
                                                           verbosity)
        df_full_list_att += [df_att]
        df_full_list_lab += [df_lbl]
        id_list += [mc_id]
        size_list += [mc_size]
        grp = np.append(grp, grp_idx * np.ones(len(df_att), dtype=int))
        grp_idx += 1

    lbl = pd.concat(df_full_list_lab)
    att = pd.concat(df_full_list_att)

    if weights is True:
        energy = lbl.CorsikaWeightMap_PrimaryEnergy.as_matrix()
        ptype = lbl.CorsikaWeightMap_PrimaryType.as_matrix()
        wgt = calc_weights(id_list, size_list, energy, ptype)
    else:
        wgt = np.ones(len(att))

    lbl = pd.concat(df_full_list_lab).fillna(0)
    att = pd.concat(df_full_list_att).fillna(0)
    ensure_safe_float32_cast_(lbl)
    ensure_safe_float32_cast_(att)

    return lbl, att, wgt, grp



def load_data(filename_list,
              label_list=["HoinkaLabels",
                          "MCPrimary1",
                          "MCMostEnergeticInIce"],
              attribute_list=["HoinkaAttributes"],
              exp=False):
    """Loads a list of data files in hdf5 format.

    Parameters
    ----------
    filename_list : list(string)
                    A list of paths to the hdf5 files.

    label_list : list(string)
                 A list of the label keys you want to load.

    attribute_list : list(string)
                     A list of attribute keys you want to load.

    Returns
    -------
    df_lbl : Pandas DataFrame
             Labels

    df_att : Pandas DataFrame
             Attribtues

    weights : array, shape = [n_samples,]
              Weights
    """
    # In case you only want to load a single file
    if type(filename_list) != list:
        filename_list = [filename_list]
    df_att_list = []
    if exp is False:
        df_lbl_list = []
        meta_id_list = []
        meta_size_list = []

    for f in filename_list:
        df_att_list += [pd.concat([pd.read_hdf(f, key) for key in attribute_list], axis=1)]
        if exp is False:
            df_lbl_list += [pd.concat([pd.read_hdf(f, key) for key in label_list], axis=1)]
            meta_id_list += [pd.read_hdf(f, "MetaInfo")["dataset_id"].as_matrix()]
            meta_size_list += [pd.read_hdf(f, "MetaInfo")["dataset_size"].as_matrix()]
    if exp is False:
        df_lbl = pd.concat(df_lbl_list).fillna(0)
    else:
        df_lbl = None
    df_att = pd.concat(df_att_list).fillna(0)
    ensure_safe_float32_cast_(df_att)
    if exp is False:
        weights = calc_weights(meta_id_list, meta_size_list,
                               df_lbl["energy"].as_matrix()[:, 0],
                               df_lbl["pdg_encoding"].as_matrix()[:, 0])
    else:
        weights = None
    groups = []
    grp_idx = 0
    for df in df_att_list:
        groups += [np.full((len(df)), grp_idx, dtype=int)]
        grp_idx += 1
    groups = np.concatenate(groups)
    return df_lbl, df_att, weights, groups
