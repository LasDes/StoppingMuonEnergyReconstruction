"""
.. module:: featureelectionMethods
    :synopsis: Methods for Feature Selection.
.. moduleauthor:: Tobias Hoinka <tobias.hoinka@udo.edu>
"""
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectFromModel, f_classif

"""Removes low features with low variance.

Parameters
----------
att : array, shape = [n_samples, n_features]
      Features.

variance_threshold : float, optional
                     Variance Threshold.

Returns
-------
picks : array, shape = [n_features,]
        Boolean mask, that is true if feature has survived.
"""
def rm_low_var(att, variance_threshold=0.1):
    std = np.sqrt(np.abs(np.cov(att.T).diagonal()))
    mean = np.abs(np.mean(att, axis=0)) + 1e-3
    return std / mean > variance_threshold

"""Choses the best of strongly correlated features

Parameters
----------
att : array, shape = [n_samples, n_features]
      Features.

label : array, shape = [n_samples,]
        Label.

correlation_threshold : float, optional
                        Correlation Threshold.

test_statistic : method(X, Y) --> stat, p_value, optional
                 Test statistic method to be used.
Returns
-------
picks : array, shape = [n_features,]
        Boolean mask, that is true if feature has survived.
"""
def rm_weaker_correlated_features(att, label, correlation_threshold=0.99,
                                  test_statistic=f_classif):
    att_np = att.as_matrix()
    n_features = np.shape(att_np)[1]
    corr = np.corrcoef(att_np.T) - np.identity(n_features)
    visited = np.array([], dtype=int)
    picks = np.zeros(n_features, dtype=bool)
    for i in range(n_features):
        if i not in visited:
            idx = np.argwhere(corr[i, :] > correlation_threshold).flatten()
            if len(idx) > 1:
                stats = test_statistic(att_np[:, idx], label)[0]
                picks[idx[np.argwhere(stats == np.max(stats))]] = 1
            else:
                picks[i] = True
            visited = np.append(visited, idx)
    return picks

"""Implements an Ad Hoc Feature Selection exploiting SelectFromModel and
BaseEstimator

Parameters
----------
Ranking : array, shape = [n_features,]
          Ranking of features, meaning values that correspond to how good the
          features are relativiely.

k : integer
    Number of features to be selected.

Returns
-------
feature_selection : sklearn.feature_selection.SelectFromModel Object
                    The feature selection object to be employed in a sklearn
                    pipeline or whatever.
"""
def ad_hoc_feature_selection(mask, k):
    ad_hoc = BaseEstimator()
    ad_hoc.feature_importances_ = mask
    if k == "all":
        threshold = np.min(mask[mask > 0.0])
    elif k > np.sum(mask != 0.0):
        k = np.sum(mask != 0.0)
    else:
        threshold = mask[np.argsort(mask)[-k]]
    feature_selection = SelectFromModel(ad_hoc, threshold=threshold,
                                        prefit=True)
    return feature_selection

def rm_nan_features(att, nan_threshold=0.2):
    return 1.0 - np.sum(np.isfinite(att),
                        axis=0) / float(len(att)) < nan_threshold 