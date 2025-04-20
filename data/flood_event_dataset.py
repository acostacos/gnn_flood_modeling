import os
import torch
import numpy as np
import h5py

from pathlib import Path
from torch import Tensor
from torch_geometric.data import Dataset, Data
from typing import Tuple, List, Dict
from utils import convert_utils, file_utils, Logger

from .dataset_debug_helper import DatasetDebugHelper
from .feature_transform import TRANSFORM_MAP, byte_to_timestamp, to_torch_tensor_w_transpose

FEATURE_CLASS_NODE = "node_features"
FEATURE_CLASS_EDGE = "edge_features"

FEATURE_TYPE_STATIC = "static"
FEATURE_TYPE_DYNAMIC = "dynamic"

MAX_CACHE_SIZE_IN_GB = 20

class FloodEventDataset(Dataset):
    def __init__(self,
                 dataset_info_path: str,
                 root_dir: str,
                 hec_ras_hdf_file: str,
                 nodes_shp_file: str,
                 edges_shp_file: str,
                 previous_timesteps: int = 0,
                 node_features: dict[str, bool] = {},
                 edge_features: dict[str, bool] = {},
                 normalize: bool = False,
                 debug: bool = False,
                 logger: Logger = None,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        # Setup debug and logger
        log = print
        if logger is not None and hasattr(logger, 'log'):
            log = logger.log
        
        self.debug = debug
        if self.debug:
            self.debug_helper = DatasetDebugHelper(log)

        # Set file paths to load data from
        current_dir = Path(__file__).parent
        self.graph_metadata_path = current_dir / 'graph_metadata.yaml'
        self.feature_metadata_path = current_dir / 'feature_metadata.yaml'
        self.dataset_info_path = dataset_info_path
        self.hdf_file = hec_ras_hdf_file
        self.nodes_shp_file = nodes_shp_file
        self.edges_shp_file = edges_shp_file
        self.normalize = normalize

        if self.debug:
            self.debug_helper.print_file_paths(self.graph_metadata_path, self.feature_metadata_path, self.dataset_info_path, root_dir, self.hdf_file, self.nodes_shp_file, self.edges_shp_file)

        # Initialize dataset variables
        self.previous_timesteps = previous_timesteps
        self.timesteps = self._get_event_timesteps(root_dir)
        self.dataset_info = {
            'node_features': [],
            'edge_features': [],
            'num_static_node_features': 0,
            'num_dynamic_node_features': 0,
            'num_static_edge_features': 0,
            'num_dynamic_edge_features': 0,
            'previous_timesteps': previous_timesteps,
            'is_normalized': normalize,
        }

        # Set features to load
        included_features = {}
        included_features[FEATURE_CLASS_NODE] = self._get_feature_list(node_features)
        included_features[FEATURE_CLASS_EDGE] = self._get_feature_list(edge_features)
        self.feature_metadata = self._get_feature_metadata(included_features)
        self._enforce_dataset_consistency()

        super().__init__(root_dir, transform, pre_transform, pre_filter, log=debug)

        # In memory cache
        self.cache = {}
        self.estimated_cache_size = 0

    @property
    def raw_file_names(self):
        return [self.hdf_file, self.nodes_shp_file, self.edges_shp_file]

    @property
    def processed_file_names(self):
        return ['processed_data.h5']

    def download(self):
        pass

    def process(self):
        edge_index  = self._get_graph_properties()
        static_nodes, dynamic_nodes = self._get_features(FEATURE_CLASS_NODE)
        static_edges, dynamic_edges = self._get_features(FEATURE_CLASS_EDGE)

        if self.debug:
            self.debug_helper.print_data_format(self.previous_timesteps)

        num_datapoints = self.len()

        with h5py.File(self.processed_paths[0], 'w') as file:
            x_dset = file.create_dataset('x', shape=(num_datapoints, 0, 0), maxshape=(num_datapoints, None, None))
            edge_attr_dset = file.create_dataset('edge_attr', shape=(num_datapoints, 0, 0), maxshape=(num_datapoints, None, None))
            y_dset = file.create_dataset('y', shape=(num_datapoints, 0, 1), maxshape=(num_datapoints, None, 1))
            y_edge_dset = file.create_dataset('y_edge', shape=(num_datapoints, 0, 1), maxshape=(num_datapoints, None, 1))

            for i in range(num_datapoints):
                node_features = self._get_timestep_data(i, static_nodes, dynamic_nodes)
                edge_features = self._get_timestep_data(i, static_edges, dynamic_edges)

                label_node = dynamic_nodes[i+1][:, [-1]] # Water level
                label_edges = dynamic_edges[i+1] # Velocity

                if self.debug:
                    self.debug_helper.test_data_format(FEATURE_CLASS_NODE, i, node_features, self.previous_timesteps)
                    self.debug_helper.test_data_format(FEATURE_CLASS_EDGE, i, edge_features, self.previous_timesteps)
                    self.debug_helper.compute_total_size([node_features, edge_index, edge_features, label_node, label_edges])

                data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_features,
                            y=label_node, y_edge=label_edges, timestep=self.timesteps[i])

                # Apply transforms
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                # Resize datasets if they are empty based on transformations
                if x_dset.shape[1] == 0 and x_dset.shape[2] == 0:
                    x_dset.resize(node_features.shape[0], axis=1)
                    x_dset.resize(node_features.shape[1], axis=2)
                if edge_attr_dset.shape[1] == 0 and edge_attr_dset.shape[2] == 0:
                    edge_attr_dset.resize(edge_features.shape[0], axis=1)
                    edge_attr_dset.resize(edge_features.shape[1], axis=2)
                if y_dset.shape[1] == 0:
                    y_dset.resize(label_node.shape[0], axis=1)
                if y_edge_dset.shape[1] == 0:
                    y_edge_dset.resize(label_edges.shape[0], axis=1)

                x_dset[i] = data.x
                edge_attr_dset[i] = data.edge_attr
                y_dset[i] = data.y
                y_edge_dset[i] = data.y_edge

            file.create_dataset('edge_index', data=edge_index)

        self.save_dataset_info()

        if self.debug:
            self.debug_helper.print_dataset_loaded(self.len(), data, self.dataset_info)
            self.debug_helper.clear()

    def len(self):
        return len(self.timesteps) - 1 # Last time step is only used as a label

    def get(self, idx):
        if self.debug:
            self.debug_helper.start_timer()

        if idx in self.cache:
            return self.cache[idx]

        with h5py.File(self.processed_paths[0], 'r') as file:
            x = torch.from_numpy(file['x'][idx])
            edge_attr = torch.from_numpy(file['edge_attr'][idx])
            y = torch.from_numpy(file['y'][idx])
            y_edge = torch.from_numpy(file['y_edge'][idx])
            edge_index = torch.from_numpy(file['edge_index'][:])
            timestep = self.timesteps[idx]

            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, y_edge=y_edge, timestep=timestep)

        if self.estimated_cache_size < convert_utils.gb_to_bytes(MAX_CACHE_SIZE_IN_GB):
            self.cache[idx] = data

            total_bytes = 0
            for value in data.values():
                if torch.is_tensor(value):
                    total_bytes += value.element_size() * value.nelement()
            self.estimated_cache_size += total_bytes

        if self.debug:
            self.debug_helper.end_timer(title='Time taken to get data point: ')

        return data

    def get_dataset_info(self) -> Dict | None:
        if not os.path.exists(self.dataset_info_path):
            return None

        dataset_info = file_utils.read_yaml_file(self.dataset_info_path)
        return dataset_info

    def save_dataset_info(self):
        if os.path.exists(self.dataset_info_path):
            dataset_info = file_utils.read_yaml_file(self.dataset_info_path)
            if self.root not in dataset_info['included_datasets']:
                dataset_info['included_datasets'].append(self.root)
        else:
            dataset_info = {**self.dataset_info, 'included_datasets': [self.root]}
        file_utils.save_to_yaml_file(self.dataset_info_path, dataset_info)

        if self.debug:
            self.debug_helper.print_dataset_info_saved(self.dataset_info_path)


    # =========== Helper Methods ===========

    def _get_event_timesteps(self, root: str):
        graph_metadata = file_utils.read_yaml_file(self.graph_metadata_path)
        timesteps_kwargs = graph_metadata['properties']['timesteps']

        hdf_filepath = Path(root) / 'raw' / self.hdf_file
        timesteps = file_utils.read_hdf_file_as_numpy(filepath=hdf_filepath, property_path=timesteps_kwargs['path'])
        timesteps = byte_to_timestamp(timesteps)

        if self.debug:
            self.debug_helper.print_timesteps_info(timesteps)

        return timesteps

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
                    self.dataset_info[feature_class].append(name)

            feature_metadata[feature_class] = local_metadata

        return feature_metadata

    def _enforce_dataset_consistency(self):
        dataset_info = self.get_dataset_info()
        if dataset_info is None:
            return
        
        if (set(dataset_info[FEATURE_CLASS_NODE]) != set(self.dataset_info[FEATURE_CLASS_NODE])
            or set(dataset_info[FEATURE_CLASS_EDGE]) != set(self.dataset_info[FEATURE_CLASS_EDGE])
            or dataset_info['previous_timesteps'] != self.dataset_info['previous_timesteps']
            or dataset_info['is_normalized'] != self.dataset_info['is_normalized']):
            raise ValueError(f'Dataset features are inconsistent with previously loaded datasets. See {self.dataset_info_path}')

    def _get_graph_properties(self) -> Tuple[torch.Tensor, torch.Tensor]:
        graph_metadata = file_utils.read_yaml_file(self.graph_metadata_path)
        property_metadata = graph_metadata['properties']

        edge_index = self._load_feature_data(feature_class=FEATURE_CLASS_EDGE, **property_metadata['edge_index'])
        edge_index = to_torch_tensor_w_transpose(edge_index)

        # See if this is still needed
        # pos = self._load_feature_data(feature_class=FEATURE_CLASS_NODE, **property_metadata['pos'])
        # pos = to_torch_tensor_w_transpose(pos)

        if self.debug:
            self.debug_helper.print_graph_properties(edge_index)

        return edge_index
    
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

        if self.normalize:
            static_features, dynamic_features = self._normalize_features(static_features, dynamic_features)

        return static_features, dynamic_features

    def _load_feature_data(self, file: str, **kwargs) -> np.ndarray:
        if file == 'hdf':
            return file_utils.read_hdf_file_as_numpy(filepath=self.raw_paths[0], property_path=kwargs['path'])
        if file == 'shp':
            path_idx = 1 if kwargs['feature_class'] == FEATURE_CLASS_NODE else 2
            return file_utils.read_shp_file_as_numpy(filepath=self.raw_paths[path_idx], columns=kwargs['column'])
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
    
    def _normalize_features(self, static_features: torch.Tensor, dynamic_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Z-score normalization of features"""
        EPS = 1e-7 # Prevent division by zero
        new_static_features = (static_features - static_features.mean(dim=0)) / (static_features.std(dim=0) + EPS)
        new_dynamic_features = (dynamic_features - dynamic_features.mean(dim=1, keepdim=True)) / (dynamic_features.std(dim=1, keepdim=True) + EPS)
        return new_static_features, new_dynamic_features

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
