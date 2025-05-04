import time
import torch

from pathlib import Path
from datetime import datetime
from torch import Tensor
from torch_geometric.data import Data
from typing import List, Dict, Callable

class DatasetDebugHelper:
    def __init__(self, logger: Callable):
        self.logger = logger
        self.debug_total_size = 0
        self.timer_start_time = None
        self.timer_end_time = None

    def print_file_paths(self, dataset_info_path: str, root: str, hdf_filename: str, nodes_shp_filename: str, edges_shp_filename: str):
        self.logger('Loading data from the following files:')
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

    def print_loaded_features(self, feature_assignment: str, feature_type: str, features: Dict[str, Tensor]):
        self.logger(f'Successfully loaded {feature_type} features for {feature_assignment}:')
        for name, data in features.items():
            self.logger(f'\t{name}: {data.shape}')

    def print_data_format(self, feature_assignment: str, static_features: List[str], dynamic_features: List[str], previous_timesteps: int):
        self.logger(f'{feature_assignment} features format:')
        self.logger(f'[')
        for feat in static_features:
            self.logger(f'\t{feat} (static)')

        for feat in dynamic_features:
            for i in range(previous_timesteps, -1, -1):
                timestep = 't' if i == 0 else f't-{i}' 
                self.logger(f'\t{feat} {timestep} (dynamic)')
        self.logger(f']')

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
