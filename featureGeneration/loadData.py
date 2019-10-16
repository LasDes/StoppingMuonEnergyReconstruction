#
# Contains main routine to load data as prepared by the featureExtraction
# script.
#
# 2016 T. Hoinka (tobias.hoinka@udo.edu)
#
import numpy as np
import h5py
"""
Loads data from a *.csv and cleans up (usually just replacing NaNs with Zeros).
Returns Names of Attributes and Attributes + Name of Labels + Labels.
"""
def loadDataCSV(filename):
    [names, roles] = np.genfromtxt(
        filename, delimiter=",", max_rows=2, dtype=str, comments=None)
    data = np.loadtxt(filename, delimiter=",", skiprows=2)

    label_names = names[roles == "label"]
    attribute_names = names[roles == "attribute"]
    labels = data[:, roles == "label"]
    attributes = data[:, roles == "attribute"]

    attributes[np.isnan(attributes)] = 0.0

    return label_names, labels, attribute_names, attributes


def loadDataHD5(filename, showStatus=True, exp=False):
    file = h5py.File(filename, "r")
    if exp is True:
        n_labels = len(file["HoinkaLabels"][0]) - 5
        n_data = len(file["HoinkaLabels"])
    else:
        n_labels = 1
        n_data = len(file["HoinkaAttributes"])
    n_attributes = len(file["HoinkaAttributes"][0]) - 5

    if showStatus:
        print("%s: %i frames, %i labels, %i attributes." % (filename,
                                                            n_data,
                                                            n_labels,
                                                            n_attributes))

    label_names = np.zeros(n_labels, dtype="S70")
    attribute_names = np.zeros(n_attributes, dtype="S70")
    labels = np.zeros((n_data, n_labels))
    attributes = np.zeros((n_data, n_attributes))

    index_label = 0
    index_attribute = 0
    if exp is True:
        for key in file["HoinkaLabels"].value[0].dtype.names[5:]:
            label_names[index_label] = str(key)
            labels[:, index_label] = file["HoinkaLabels"][key]
            index_label += 1

    for key in file["HoinkaAttributes"].value[0].dtype.names[5:]:
        attribute_names[index_attribute] = str(key)
        attributes[:, index_attribute] = file["HoinkaAttributes"][key]
        index_attribute += 1

    attributes[np.isnan(attributes)] = 0.0
    return label_names, labels, attribute_names, attributes

def concatenateData(F_list):
    N_F = np.shape(F_list[0])[1]
    L = [0]
    M = 0
    for F in F_list:
        L += len(F)
        L += [M]
    result = np.zeros((L[-1], N_F))
    for i in range(len(F_list)):
        result[L[i]:L[i+1], :] = F_list[i]
    return result 
