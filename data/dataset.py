from torch.utils.data import Dataset
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch_geometric_temporal.dataset import (
    METRLADatasetLoader,
    PemsBayDatasetLoader,
    WindmillOutputLargeDatasetLoader,
)

class DataSet(Dataset):
    def __init__(self, data, mask):
        super(DataSet, self).__init__()
        self.data = data
        self.mask = mask

    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        return self.data[idx], self.mask[idx]

def normilize_data(node_attribute):
    scaler = MinMaxScaler()
    node_attribute = scaler.fit_transform(node_attribute)
    return node_attribute

def del_doplicated_row(node_attribute):
    nrow, ncol = node_attribute.shape
    dup_idx_list = []
    for idx in range(nrow):
        if np.all(node_attribute[idx, :] == node_attribute[idx, 0]):
            dup_idx_list.append(idx)
    node_attribute = np.delete(node_attribute, dup_idx_list, axis=0)
    return node_attribute

def get_dataset(dataset_name):
    if dataset_name == "METR-LA":
        data = METRLADatasetLoader()
        node_attribute = data.X[:, 0, :].T.numpy()
    elif dataset_name == "PEMS-BAY":
        data = PemsBayDatasetLoader()
        node_attribute = data.X[:, 0, :].T.numpy()
    elif dataset_name == "Windmill319":
        data = WindmillOutputLargeDatasetLoader()
        node_attribute = np.array(data._dataset["block"])

    normilized_node_attribute = normilize_data(
        node_attribute[:, : int(1 * node_attribute.shape[1])]
    )
    normilized_node_attribute = del_doplicated_row(normilized_node_attribute)
    dataset = normilized_node_attribute

    return dataset
