import os
import torch
import numpy as np

from datetime import datetime
from pathlib import Path
from torch import Tensor
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.transforms import ToUndirected
from typing import Tuple, List
from utils import file_utils, Logger

from .dataset_debug_helper import DebugHelper
from .feature_transform import TRANSFORM_MAP, byte_to_timestamp, to_torch_tensor_w_transpose

FEATURE_CLASS_NODE = "node_features"
FEATURE_CLASS_EDGE = "edge_features"

FEATURE_TYPE_STATIC = "static"
FEATURE_TYPE_DYNAMIC = "dynamic"

class FloodEventDataset(InMemoryDataset):
    def __init__(self,
                 root: str,
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
        log = print
        if logger is not None and hasattr(logger, 'log'):
            log = logger.log
        
        self.debug = debug
        if self.debug:
            self.debug_helper = DebugHelper(log)

        current_dir = Path(__file__).parent
        self.graph_metadata_path = current_dir / 'graph_metadata.yaml'
        self.feature_metadata_path = current_dir / 'feature_metadata.yaml'

        self.hdf_filepath = hec_ras_hdf_path
        self.shp_filepath = { FEATURE_CLASS_NODE: nodes_shp_path, FEATURE_CLASS_EDGE: edges_shp_path }
        if self.debug:
            self.debug_helper.print_file_paths(self.graph_metadata_path, self.feature_metadata_path, self.hdf_filepath, self.shp_filepath)

        included_features = {}
        included_features[FEATURE_CLASS_NODE] = self._get_feature_list(node_features)
        included_features[FEATURE_CLASS_EDGE] = self._get_feature_list(edge_features)
        self.feature_metadata = self._get_feature_metadata(included_features)

        self.previous_timesteps = previous_timesteps
        self.dataset = None
        self.dataset_info = {
            'num_static_node_features': 0,
            'num_dynamic_node_features': 0,
            'num_static_edge_features': 0,
            'num_dynamic_edge_features': 0,
            'previous_timesteps': previous_timesteps,
        }

        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [self.hdf_filepath, self.shp_filepath[0], self.shp_filepath[1]]

    @property
    def processed_file_names(self):
        return ['data.pt', 'data_info.yaml']

    def download(self):
        raise NotImplementedError

    def process(self):
        # Read data into huge `Data` list.
        data_list = [...]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.save(data_list, self.processed_paths[0])
    
    def load(self) -> List[Data]:
        timesteps, edge_index, pos = self._get_graph_properties()

        static_nodes, dynamic_nodes = self._get_features(FEATURE_CLASS_NODE)
        static_edges, dynamic_edges = self._get_features(FEATURE_CLASS_EDGE)

        if self.debug:
            self.debug_helper.print_data_format(self.previous_timesteps)

        self.dataset = []
        for i in range(len(timesteps)-1): # Last time step is only used as a label
            node_features = self._get_timestep_data(i, static_nodes, dynamic_nodes)
            edge_features = self._get_timestep_data(i, static_edges, dynamic_edges)

            if self.debug:
                self.debug_helper.test_data_format(FEATURE_CLASS_NODE, i, node_features, self.previous_timesteps)
                self.debug_helper.test_data_format(FEATURE_CLASS_EDGE, i, edge_features, self.previous_timesteps)

            label_node = dynamic_nodes[i+1][:, [-1]] # Water level
            label_edges = dynamic_edges[i+1] # Velocity
            self.debug_helper.compute_total_size([node_features, edge_index, edge_features, label_node, label_edges])
            data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_features,
                        y=label_node, y_edge=label_edges, pos=pos, timestep=timesteps[i])
            data = ToUndirected()(data) # Transform to undirected graph

            self.dataset.append(data)

        if self.debug:
            self.debug_helper.print_dataset_loaded(self.dataset, self.dataset_info)
            self.debug_helper.clear()


    # =========== Helper Methods ===========

    def _get_feature_list(self, feature: dict[str, bool]) -> List[str]:
        return [name for name, is_included in feature.items() if is_included]
        
    def _get_feature_metadata(self, included_features: dict) -> dict:
        yaml_metadata = file_utils.read_yaml_file(self.feature_metadata_path)

        feature_metadata = {}
        for feature_class in [FEATURE_CLASS_NODE, FEATURE_CLASS_EDGE]:
            if feature_class not in included_features or feature_class not in yaml_metadata:
                continue

            local_metadata = {}
            for name, data in yaml_metadata[feature_class].items():
                if name in included_features[feature_class]:
                    local_metadata |= {name: data}
                    self.dataset_info[f"num_{data['type']}_{feature_class}"] += 1

            feature_metadata[feature_class] = local_metadata

        return feature_metadata

    def _get_graph_properties(self) -> Tuple[List[datetime], torch.Tensor, torch.Tensor]:
        graph_metadata = file_utils.read_yaml_file(self.graph_metadata_path)
        property_metadata = graph_metadata['properties']

        timesteps = self._load_feature_data(**property_metadata['timesteps'])
        timesteps = byte_to_timestamp(timesteps)

        edge_index = self._load_feature_data(feature_class=FEATURE_CLASS_EDGE, **property_metadata['edge_index'])
        edge_index = to_torch_tensor_w_transpose(edge_index)

        pos = self._load_feature_data(feature_class=FEATURE_CLASS_NODE, **property_metadata['pos'])
        pos = to_torch_tensor_w_transpose(pos)

        if self.debug:
            self.debug_helper.print_graph_properties(timesteps, edge_index, pos)

        return timesteps, edge_index, pos
    
    def _get_features(self, feature_class: str) -> Tuple[torch.Tensor, torch.Tensor]:
        features = { FEATURE_TYPE_STATIC: [], FEATURE_TYPE_DYNAMIC: [] }
        for name, metadata in self.feature_metadata[feature_class].items():
            data = self._load_feature_data(feature_class=feature_class, **metadata)

            if name in TRANSFORM_MAP:
                transform_func = TRANSFORM_MAP[name]
                data = transform_func(data)

            if 'type' not in metadata or metadata['type'] not in [FEATURE_TYPE_STATIC, FEATURE_TYPE_DYNAMIC]:
                continue

            features[metadata['type']].append(data)
            if self.debug:
                self.debug_helper.assign_features(feature_class, metadata['type'], name, data)

        static_features, dynamic_features = self._format_features(features)
        if self.debug:
            self.debug_helper.print_loaded_features(feature_class, static_features, dynamic_features)

        return static_features, dynamic_features

    def _load_feature_data(self, file: str, **kwargs) -> np.ndarray:
        if file == 'hdf':
            return file_utils.read_hdf_file_as_numpy(filepath=self.hdf_filepath, property_path=kwargs['path'])
        if file == 'shp':
            filepath = self.shp_filepath[kwargs['feature_class']]
            return file_utils.read_shp_file_as_numpy(filepath=filepath, columns=kwargs['column'])
        raise ValueError('Invalid file type in feature metadata. Valid values are: hdf, shp.')

    def _format_features(self, features: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        # Static features = (num_items, num_features)
        static_features = torch.from_numpy(np.array(features[FEATURE_TYPE_STATIC]))
        if len(static_features) > 0:
            static_features = static_features.transpose(1, 0)

        # Dynamic features = (num_timesteps, num_items, num_features)
        dynamic_features = torch.from_numpy(np.array(features[FEATURE_TYPE_DYNAMIC]))
        if len(dynamic_features) > 0:
            dynamic_features = dynamic_features.permute([1, 2, 0])

        return static_features, dynamic_features
    
    def _get_timestep_data(self, timestep_idx: int, static_features: Tensor, dynamic_features: Tensor) -> Tensor:
        """Returns the data for a specific timestep in the format [static_features, previous_dynamic_features, current_dynamic_features]"""

        if timestep_idx < self.previous_timesteps:
            _, num, num_df = dynamic_features.shape
            num_rows_per_dynamic = self.previous_timesteps+1
            ts_dynamic_features = torch.zeros((num, num_df * (num_rows_per_dynamic)))
            # Add fill for each dynamic feature
            for i in range(num_df):
                fill = torch.zeros((num, self.previous_timesteps-timestep_idx), dtype=dynamic_features.dtype)
                valid = dynamic_features[:timestep_idx+1, :, i].transpose(1, 0)
                combined = torch.cat([fill, valid], dim=1)
                ts_dynamic_features[:, i*num_rows_per_dynamic:(i+1)*num_rows_per_dynamic] = combined
        else:
            ts_dynamic_features = dynamic_features[timestep_idx-self.previous_timesteps:timestep_idx+1]
            ts_dynamic_features = ts_dynamic_features.permute((1, 2, 0)).flatten(start_dim=1)

        return torch.cat([static_features, ts_dynamic_features], dim=1)
