import numpy as np

from constants import FeatureClass, FeatureType, FeatureSource
from pathlib import Path
from torch_geometric.data import Data
from typing import Tuple
from utils import file_utils

from .feature_transform import TRANSFORM_MAP

class TemporalGraphDataset:
    def __init__(self, hec_result_path='', nodes_shape_path='', edges_shape_path='', node_features={}, edge_features={}):
        self.hdf_filepath = hec_result_path
        self.shp_filepath = {
            FeatureClass.NODE: nodes_shape_path,
            FeatureClass.EDGE: edges_shape_path,
        }

        included_features = {}
        included_features[FeatureClass.GRAPH] = ['timesteps', 'edge_index', 'pos']
        included_features[FeatureClass.NODE] = self.get_feature_list(node_features)
        included_features[FeatureClass.EDGE] = self.get_feature_list(edge_features)
        self.feature_metadata = self.get_feature_metadata(included_features)

    def get_feature_list(self, feature: dict[str, bool]) -> list[str]:
        return [name for name, is_included in feature.items() if is_included]
    
    def get_feature_metadata(self, included_features: dict) -> dict:
        FEATURE_METADATA_FILE = "feature_metadata.json"
        feature_metadata_path = Path(__file__).parent / FEATURE_METADATA_FILE
        json_metadata = file_utils.read_json_file(feature_metadata_path)

        feature_metadata = {}
        for feature_class in FeatureClass:
            if feature_class not in included_features or feature_class not in json_metadata:
                continue

            local_metadata = {}
            for name, data in json_metadata[feature_class].items():
                if name in included_features[feature_class]:
                    local_metadata |= {name: data}
            
            feature_metadata[feature_class] = local_metadata
        
        return feature_metadata

    def load(self) -> list[Data]:
        _, _, raw_graph = self.get_features(FeatureClass.GRAPH)
        static_nodes, dynamic_nodes, _ = self.get_features(FeatureClass.NODE)
        static_edges, dynamic_edges, _ = self.get_features(FeatureClass.EDGE)

        timesteps, edge_index, pos = raw_graph
        dataset = []
        for i, ts in enumerate(timesteps):
            ts_df_nodes = dynamic_nodes[i]
            ts_df_edges = dynamic_edges[i]
            # Convention = [static_features, dynamic_features]
            node_features = np.concatenate([static_nodes, ts_df_nodes], axis=1)
            edge_features = np.concatenate([static_edges, ts_df_edges], axis=1)
            data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_features, pos=pos)
            dataset.append(data)

        return dataset
    
    def get_features(self, feature_class: FeatureClass) -> Tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
        features = { FeatureType.STATIC: [], FeatureType.DYNAMIC: [], FeatureType.RAW: [] }
        for name, metadata in self.feature_metadata[feature_class].items():
            data = self.load_feature_data(source=metadata['file'], feature_class=feature_class, **metadata)

            if name in TRANSFORM_MAP:
                transform_func = TRANSFORM_MAP[name]
                data = transform_func(data)

            if 'type' not in metadata or metadata['type'] not in FeatureType:
                continue

            features[metadata['type']].append(data)

        return self.format_features(features)

    def load_feature_data(self, source: FeatureSource, feature_class: FeatureClass, **kwargs) -> np.ndarray:
        if source == FeatureSource.HDF:
            return file_utils.read_hdf_file_as_numpy(filepath=self.hdf_filepath, property_path=kwargs['path'])
        if source == FeatureSource.SHP:
            filepath_key = kwargs['shp_file'] if 'shp_file' in kwargs else feature_class
            filepath = self.shp_filepath[filepath_key]
            return file_utils.read_shp_file_as_numpy(filepath=filepath, columns=kwargs['column'])
        raise Exception('Invalid file type in feature metadata. Valid values are: hdf, shp.')

    def format_features(self, features: dict) -> Tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
        # Static features = (num_nodes, num_features)
        static_features = np.array(features[FeatureType.STATIC])
        if len(static_features) > 0:
            static_features = static_features.transpose([1, 0])

        # Dynamic features = (num_timesteps, num_nodes, num_features)
        dynamic_features = np.array(features[FeatureType.DYNAMIC])
        if len(dynamic_features) > 0:
            dynamic_features = dynamic_features.transpose([1, 2, 0])

        # Raw features = list of numpy arrays
        raw_features = features[FeatureType.RAW]

        return static_features, dynamic_features, raw_features
