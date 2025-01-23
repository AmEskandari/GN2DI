import numpy as np
from torch_geometric.nn import knn_graph
import torch
from sklearn.metrics.pairwise import rbf_kernel, cosine_similarity

def construct_graph(train_data, num_ngh):
    """
    Construct graph from training data using k-nearest neighbors
    and multiple similarity metrics.
    """
    nrow, ncol = train_data.shape    

    edge_index = knn_graph(
        torch.tensor(train_data.T, dtype=torch.float32), 
        k=num_ngh
    ).numpy()
    
    # Create adjacency matrix
    Adjecency_mat = np.zeros((ncol, ncol))
    Adjecency_mat[edge_index[0], edge_index[1]] = 1
  
    # Calculate different similarity matrices
    weighted_mat_corr = np.cov(train_data.T) 
    weighted_mat_rbf = rbf_kernel(train_data.T, train_data.T)
    weighted_mat_cosine = cosine_similarity(train_data.T, train_data.T)
    
    edge_index = []
    edge_attr = []

    # Create edge list and attributes
    for i in range(ncol):
        for j in range(ncol):
            if Adjecency_mat[i,j] == 1:
                edge_index.append([i,j])
                edge_attr.append([
                    weighted_mat_corr[i,j],
                    weighted_mat_rbf[i,j],
                    weighted_mat_cosine[i,j]
                ])
    
    edge_index = np.array(edge_index, dtype=np.int32).T
    edge_attr = np.array(edge_attr, dtype=np.float32)
    return edge_index, edge_attr
