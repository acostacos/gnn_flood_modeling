import torch
import numpy as np

from datetime import datetime
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.transforms import ToUndirected
from typing import Tuple, List
from utils import file_utils, Logger

from .feature_transform import TRANSFORM_MAP, byte_to_timestamp, to_torch_tensor_w_transpose

FEATURE_CLASS_NODE = "node_features"
FEATURE_CLASS_EDGE = "edge_features"

FEATURE_TYPE_STATIC = "static"
FEATURE_TYPE_DYNAMIC = "dynamic"

class FloodingEventDataset():
    def __init__(self,
                 graph_metadata_path: str,
                 feature_metadata_path: str,
                 hec_ras_hdf_path: str = '',
                 nodes_shp_path: str = '',
                 edges_shp_path: str = '',
                 previous_timesteps: int = 0,
                 node_features: dict[str, bool] = {},
                 edge_features: dict[str, bool] = {},
                 debug: bool = False,
                 logger: Logger = None):
        self.debug = debug
        self.debug_features = {}
        self.log = print
        if logger is not None and hasattr(logger, 'log'):
            self.log = logger.log

        self.graph_metadata_path = graph_metadata_path
        self.feature_metadata_path = feature_metadata_path
        self.hdf_filepath = hec_ras_hdf_path
        self.shp_filepath = { FEATURE_CLASS_NODE: nodes_shp_path, FEATURE_CLASS_EDGE: edges_shp_path }
        if self.debug:
            self._debug_print_file_paths()

        self.previous_timesteps = previous_timesteps
        self.dataset_info = {
            'num_static_node_features': 0,
            'num_dynamic_node_features': 0,
            'num_static_edge_features': 0,
            'num_dynamic_edge_features': 0,
            'previous_timesteps': previous_timesteps,
        }

        included_features = {}
        included_features[FEATURE_CLASS_NODE] = self._get_feature_list(node_features)
        included_features[FEATURE_CLASS_EDGE] = self._get_feature_list(edge_features)
        self.feature_metadata = self._get_feature_metadata(included_features)

    def load(self) -> List[Data]:
        timesteps, edge_index, pos = self._get_graph_properties()

        static_nodes, dynamic_nodes = self._get_features(FEATURE_CLASS_NODE)
        static_edges, dynamic_edges = self._get_features(FEATURE_CLASS_EDGE)

        if self.debug:
            self._debug_print_data_format()

        dataset = []
        for i in range(len(timesteps)-1): # Last time step is only used as a label
            node_features = self._get_timestep_data(i, static_nodes, dynamic_nodes)
            edge_features = self._get_timestep_data(i, static_edges, dynamic_edges)

            if self.debug:
                self._debug_test_data_format(FEATURE_CLASS_NODE, i, node_features)
                self._debug_test_data_format(FEATURE_CLASS_EDGE, i, edge_features)

            label_node = dynamic_nodes[i+1][:, :-1] # Water level
            label_edges = dynamic_edges[i+1] # Velocity
            data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_features,
                        y=label_node, y_edge=label_edges, pos=pos)
            data = ToUndirected()(data) # Transform to undirected graph

            dataset.append(data)
        
        if self.debug:
            self._debug_print_dataset_loaded(dataset)

        return dataset, self.dataset_info
    

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
            self._debug_print_graph_properties(timesteps, edge_index, pos)

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
                self._debug_assign_features(feature_class, metadata['type'], name, data)

        static_features, dynamic_features = self._format_features(features)
        if self.debug:
            self._debug_print_loaded_features(feature_class, static_features, dynamic_features)

        return static_features, dynamic_features

    def _load_feature_data(self, file: str, **kwargs) -> np.ndarray:
        if file == 'hdf':
            return file_utils.read_hdf_file_as_numpy(filepath=self.hdf_filepath, property_path=kwargs['path'])
        if file == 'shp':
            filepath = self.shp_filepath[kwargs['feature_class']]
            return file_utils.read_shp_file_as_numpy(filepath=filepath, columns=kwargs['column'])
        raise Exception('Invalid file type in feature metadata. Valid values are: hdf, shp.')

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

    # =========== Debug Methods ===========

    def _debug_print_file_paths(self):
        self.log('Loading data from the following files:')
        self.log(f'\tGraph Metadata Filepath: {self.graph_metadata_path}')
        self.log(f'\tFeature Metadata Filepath: {self.feature_metadata_path}')
        self.log(f'\tHEC-RAS HDF Filepath: {self.hdf_filepath}')
        self.log(f'\tNodes SHP Filepath: {self.shp_filepath[FEATURE_CLASS_NODE]}')
        self.log(f'\tEdges SHP Filepath: {self.shp_filepath[FEATURE_CLASS_EDGE]}')
    
    def _debug_print_graph_properties(self, timesteps: List[datetime], edge_index: Tensor, pos: Tensor):
        self.log('Graph properties:')
        self.log(f'\tTimesteps: {len(timesteps)}')
        if len(timesteps) > 1:
            self.log(f'\tTimestep delta: {timesteps[1] - timesteps[0]}')
        self.log(f'\tEdge Index: {edge_index.shape}')
        self.log(f'\tPos: {pos.shape}')
    
    def _debug_assign_features(self, feature_class: str, feature_type: str, name: str, data: Tensor):
        if feature_class not in self.debug_features:
            self.debug_features[feature_class] = {}
        if feature_type not in self.debug_features[feature_class]:
            self.debug_features[feature_class][feature_type] = {}
        self.debug_features[feature_class][feature_type][name] = data
    
    def _debug_print_loaded_features(self, feature_class: str, static_features: Tensor, dynamic_features: Tensor):
        self.log(f'Successfully loaded for {feature_class}:')
        shape_map = {
            FEATURE_TYPE_STATIC: static_features.shape,
            FEATURE_TYPE_DYNAMIC: dynamic_features.shape,
        }
        for feat_type, features in self.debug_features[feature_class].items():
            self.log(f'\t{feat_type} (Shape: {shape_map[feat_type]}):')
            for name, data in features.items():
                self.log(f'\t\t{name}: {data.shape}')
    
    def _debug_print_data_format(self):
        for feat_class, feat_type_dict in self.debug_features.items():
            self.log(f'Expected {feat_class} format:')
            self.log(f'\t[')
            for feat_type, features in feat_type_dict.items():
                for name in features.keys():
                    if feat_type == FEATURE_TYPE_STATIC:
                        self.log(f'\t\t{name} ({feat_type})')
                    elif feat_type == FEATURE_TYPE_DYNAMIC:
                        for i in range(self.previous_timesteps, 0, -1):
                            self.log(f'\t\t{name} t-{i} ({feat_type})')
                        self.log(f'\t\t{name} t ({feat_type})')
            self.log(f'\t]')
    
    def _debug_test_data_format(self, feature_class: str, timestep_idx: int, data: Tensor):
        curr_idx = 0
        for feat_type, features in self.debug_features[feature_class].items():
            if feat_type == FEATURE_TYPE_STATIC:
                for orig_data in features.values():
                    assert torch.equal(data[:, curr_idx], Tensor(orig_data))
                    curr_idx += 1
            elif feat_type == FEATURE_TYPE_DYNAMIC:
                for orig_data in features.values():
                    for i in range(self.previous_timesteps, 0, -1):
                        if timestep_idx-i < 0:
                            assert torch.all(data[:, curr_idx] == 0)
                        else:
                            assert torch.equal(data[:, curr_idx], Tensor(orig_data[timestep_idx-i]))
                        curr_idx += 1
                    assert torch.equal(data[:, curr_idx], Tensor(orig_data[timestep_idx]))
                    curr_idx += 1
    
    def _debug_print_dataset_loaded(self, dataset: List[Data]):
        self.log('Succesfully loaded dateset.')
        self.log(f'Number of data points: {len(dataset)}')
        self.log(f'Sample data point: {dataset[0]}')
        self.log('Dataset Info:')
        for key, value in self.dataset_info.items():
            self.log(f'\t{key}: {value}')
