import numpy as np
import lightgbm as lgb


def get_knns_features(node, graph, features, nodes_indexing, errors, k=1, remove_error=False):
    """
    Extracts graph features. We return the feature of the nodes, 
    and of the k nearest neighbours considering incomming and outgoing edges
    
    Arguments:
        node {int} -- [description]
        graph {numpy array} -- Adjacency matrix of the graph
        features {numpy array} -- Features corresponding to each text
        nodes_indexing {list} -- Mapping between nodes dataframe and graph nodes
        errors {list} -- Nodes with parsing error
    
    Keyword Arguments:
        k {int} -- Number of neighbours (default: {1})
        remove_error {bool} -- Whether to remove nodes with parsing errors from the neighbours (default: {False})
    
    Returns:
        numpy array -- Features of the node
    """
    if k == 0:
        return features[node]

    idx = nodes_indexing.index(node) 
    
    # Getting k nearest neighbours for incomming edges
    
    in_nn = np.argsort(graph[idx])[::-1]
    in_nn = np.array([nodes_indexing[i] for i in in_nn])
    
    ft_in = []
    i = 0
    while len(ft_in) < k:
        if not errors[in_nn[i]] or not remove_error:
            ft_in.append(features[in_nn[i]])
        i += 1
    ft_in = np.array(ft_in).flatten()

    # Getting k nearest neighbours for outgoing edges

    out_nn = np.argsort(graph[:, idx])[::-1]
    out_nn = np.array([nodes_indexing[i] for i in out_nn])
    
    ft_out = []
    i = 0
    while len(ft_out) < k:
        if not errors[out_nn[i]] or not remove_error:
            ft_out.append(features[out_nn[i]])
        i += 1
    ft_out = np.array(ft_out).flatten()

    return np.concatenate([features[node], ft_in, ft_out], 0).flatten()


def run_lgb(X_train, X_val, y_train, y_val): 
    """
    Trains a lightgbm model on the data
    
    Arguments:
        X_train {numpy array} -- Training features
        X_val {numpy array} -- Validation features
        y_train {numpy array} -- Training labels
        y_val {numpy array} -- Validation labels
    
    Returns:
        lightgbm model -- Trained model
    """
    params = {"objective" : "multiclass",
              "num_class": 8,
              "num_leaves" : 3,
              "min_child_weight" : 0,
              "learning_rate" : 0.005,
              "bagging_fraction" : 1,
              "feature_fraction" : 1,
              "bagging_seed" : 420,
              "verbosity" : -1
             }
    
    lg_train = lgb.Dataset(X_train, label=y_train)
    lg_val = lgb.Dataset(X_val, label=y_val)
    model = lgb.train(params, lg_train, 10000, valid_sets=[lg_val], early_stopping_rounds=200, verbose_eval=100)
    
    return model


