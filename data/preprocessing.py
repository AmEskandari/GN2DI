import numpy as np
import copy

def np_random(seed=None):
    rng = np.random.RandomState()
    rng.seed(seed)
    return rng

def mask_data(input_features, missing_rate, mask=None, impute_type=None, rng=None):
    nodes_mean = mean_nodes(input_features)
    nrow, ncol = input_features.shape

    mask = rng.binomial(1, 1 - missing_rate, size=(nrow, ncol)).astype(bool)

    node_attribute = mask * copy.deepcopy(input_features)
    node_label = np.logical_not(mask) * copy.deepcopy(input_features)

    if impute_type == "mean":
        imputed_nodes_attribute = mean_imputation(node_attribute, nodes_mean)
    if impute_type == "zero":
        imputed_nodes_attribute = node_attribute

    return imputed_nodes_attribute, node_label

def mean_imputation(node_attribute, nodes_mean):
    nrow, ncol = node_attribute.shape
    for idx in range(node_attribute.shape[0]):
        node_attribute[idx, np.where(node_attribute[idx] == 0)] = nodes_mean[
            np.where(node_attribute[idx] == 0)
        ]
    return node_attribute

def mean_nodes(node_attribute):
    nodes_mean = np.mean(node_attribute, axis=0)
    return nodes_mean
