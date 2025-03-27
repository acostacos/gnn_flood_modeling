import os
import torch
import numpy as np
import pickle

from datetime import datetime
from torch import Tensor
from torch_geometric.data import Dataset, Data
from torch_geometric.transforms import ToUndirected
from typing import Tuple, List, Dict
from utils import file_utils, Logger

class FloodEventDataset(Dataset):
    def __init__(self,
                 root: str,
                 dataset_filename: str,
                 dataset_info_filename: str,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.dataset_path = os.path.join(root, dataset_filename)
        self.dataset_length = self.get_dataset_length(root, dataset_filename, dataset_info_filename)

    def get_dataset_length(self, root, dataset_filename, dataset_info_filename) -> int:
        dataset_info = file_utils.read_yaml_file(os.path.join(root, dataset_info_filename))
        dataset_key = os.path.basename(dataset_filename)
        return dataset_info[dataset_key]['num_data_points']

    def len(self):
        return self.dataset_length

    def load_objects(self):
        with open(self.dataset_path) as f:
            while True:
                try:
                    yield pickle.load(f)
                except EOFError:
                    break

    def get(self, idx):
        # with open(self.dataset_path, 'rb') as f:
        #     while True:
        #         try:
        #             obj = pickle.load(f)
        #             process(obj)
        #         except EOFError:
        #             break

        # return data
        return []