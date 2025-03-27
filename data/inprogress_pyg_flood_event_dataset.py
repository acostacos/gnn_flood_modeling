import os.path as osp

import torch
from torch_geometric.data import InMemoryDataset, Data
from utils import file_utils, Logger

class FloodEventDataset(InMemoryDataset):
    def __init__(self,
                 root: str,
                 graph_metadata_path: str,
                 feature_metadata_path: str,
                 hec_ras_hdf_path: str,
                 nodes_shp_path: str,
                 edges_shp_path: str,
                 previous_timesteps: int = 0,
                 node_features: dict[str, bool] = {},
                 edge_features: dict[str, bool] = {},
                 debug: bool = False,
                 logger: Logger = None,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter, log=debug)
        self.hec_ras_hdf_path = hec_ras_hdf_path
        self.nodes_shp_path = nodes_shp_path
        self.edges_shp_path = edges_shp_path

        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        # Read data into huge `Data` list.
        data_list = [...]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.save(data_list, self.processed_paths[0])
        # For PyG<2.4:
        # torch.save(self.collate(data_list), self.processed_paths[0])