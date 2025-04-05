import time
import torch

from pathlib import Path
from datetime import datetime
from torch import Tensor
from torch_geometric.data import Data
from typing import List, Dict, Callable


FEATURE_CLASS_NODE = "node_features"
FEATURE_CLASS_EDGE = "edge_features"

FEATURE_TYPE_STATIC = "static"
FEATURE_TYPE_DYNAMIC = "dynamic"

class DatasetDebugHelper:
    def __init__(self, logger: Callable):
        self.logger = logger
        self.debug_features = {}
        self.debug_total_size = 0
        self.timer_start_time = None
        self.timer_end_time = None

    def print_file_paths(self, graph_metadata_path: str, feature_metadata_path: str,  dataset_info_path: str, root: str, hdf_filename: str, nodes_shp_filename: str, edges_shp_filename: str):
        self.logger('Loading data from the following files:')
        self.logger(f'\tGraph Metadata Filepath: {graph_metadata_path}')
        self.logger(f'\tFeature Metadata Filepath: {feature_metadata_path}')
        self.logger(f'\tDataset Info Filepath: {dataset_info_path}')
        self.logger(f'\tHEC-RAS HDF Filename: {Path(root) / hdf_filename}')
        self.logger(f'\tNodes SHP Filepath: {Path(root) / nodes_shp_filename}')
        self.logger(f'\tEdges SHP Filepath: {Path(root) / edges_shp_filename}')
    
    def print_timesteps_info(self, timesteps: List[datetime]):
        self.logger(f'Timesteps: {len(timesteps)}')
        if len(timesteps) > 1:
            self.logger(f'Timestep delta: {timesteps[1] - timesteps[0]}')

    def print_graph_properties(self, edge_index: Tensor, pos: Tensor=None):
        self.logger('Graph properties:')
        self.logger(f'\tEdge Index: {edge_index.shape}')
        if pos is not None:
            self.logger(f'\tPos: {pos.shape}')

    def assign_features(self, feature_class: str, feature_type: str, name: str, data: Tensor):
        if feature_class not in self.debug_features:
            self.debug_features[feature_class] = {}
        if feature_type not in self.debug_features[feature_class]:
            self.debug_features[feature_class][feature_type] = {}
        self.debug_features[feature_class][feature_type][name] = data

    def print_loaded_features(self, feature_class: str, static_features: Tensor, dynamic_features: Tensor):
        self.logger(f'Successfully loaded for {feature_class}:')
        shape_map = {
            FEATURE_TYPE_STATIC: static_features.shape,
            FEATURE_TYPE_DYNAMIC: dynamic_features.shape,
        }
        for feat_type, features in self.debug_features[feature_class].items():
            self.logger(f'\t{feat_type} (Shape: {shape_map[feat_type]}):')
            for name, data in features.items():
                self.logger(f'\t\t{name}: {data.shape}')

    def print_data_format(self, previous_timesteps: int):
        for feat_class, feat_type_dict in self.debug_features.items():
            self.logger(f'Expected {feat_class} format:')
            self.logger(f'\t[')
            for feat_type, features in feat_type_dict.items():
                for name in features.keys():
                    if feat_type == FEATURE_TYPE_STATIC:
                        self.logger(f'\t\t{name} ({feat_type})')
                    elif feat_type == FEATURE_TYPE_DYNAMIC:
                        for i in range(previous_timesteps, 0, -1):
                            self.logger(f'\t\t{name} t-{i} ({feat_type})')
                        self.logger(f'\t\t{name} t ({feat_type})')
            self.logger(f'\t]')

    def test_data_format(self, feature_class: str, timestep_idx: int, data: Tensor, previous_timesteps: int):
        curr_idx = 0
        for feat_type, features in self.debug_features[feature_class].items():
            if feat_type == FEATURE_TYPE_STATIC:
                for orig_data in features.values():
                    assert torch.equal(data[:, curr_idx], Tensor(orig_data))
                    curr_idx += 1
            elif feat_type == FEATURE_TYPE_DYNAMIC:
                for orig_data in features.values():
                    for i in range(previous_timesteps, 0, -1):
                        if timestep_idx-i < 0:
                            assert torch.all(data[:, curr_idx] == 0)
                        else:
                            assert torch.equal(data[:, curr_idx], Tensor(orig_data[timestep_idx-i]))
                        curr_idx += 1
                    assert torch.equal(data[:, curr_idx], Tensor(orig_data[timestep_idx]))
                    curr_idx += 1

    def compute_total_size(self, tensors: List[Tensor]):
        for tensor in tensors:
            self.debug_total_size += (tensor.element_size() * tensor.nelement())

    def print_dataset_loaded(self, dataset_len: int, sample: Data, dataset_info: Dict):
        self.logger('Succesfully loaded dateset.')
        self.logger(f'Total size of dataset: {self.debug_total_size} bytes')
        self.logger(f'Number of data points: {dataset_len}')
        self.logger(f'Sample data point: {sample}')
        self.logger('Dataset Info:')
        for key, value in dataset_info.items():
            self.logger(f'\t{key}: {value}')

    def print_dataset_info_saved(self, dataset_info_path: str):
        self.logger(f'Saved dataset info to {dataset_info_path}')
    
    def start_timer(self):
        self.timer_start_time = time.time()
    
    def end_timer(self, title: str = None):
        self.timer_end_time = time.time()
        elapsed_time = self.timer_end_time - self.timer_start_time
        prefix = title or 'Time taken'
        self.logger(f'{prefix}: {elapsed_time:.4f} seconds')

    def clear(self):
        self.debug_features = {}
        self.debug_total_size = 0
