import torch
import numpy as np

from constants import FeatureClass, FeatureType, FeatureSource
from pathlib import Path
from typing import Tuple
from utils import file_utils

from .feature_transform import TRANSFORM_MAP


class BaseDataset:
    def __init__(self,
                 hec_ras_hdf_path: str = '',
                 nodes_shp_path: str = '',
                 edges_shp_path: str = '',
                 previous_timesteps: int = 0,
                 node_features: dict[str, bool] = {},
                 edge_features: dict[str, bool] = {}):
        self.hdf_filepath = hec_ras_hdf_path
        self.shp_filepath = {
            FeatureClass.NODE: nodes_shp_path,
            FeatureClass.EDGE: edges_shp_path,
        }
        self.previous_timesteps = previous_timesteps
        self.dataset_info = {
            'num_static_node_features': 0,
            'num_dynamic_node_features': 0,
            'num_static_edge_features': 0,
            'num_dynamic_edge_features': 0,
            'previous_timesteps': previous_timesteps,
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

                    if feature_class in [FeatureClass.NODE, FeatureClass.EDGE] and \
                          data['type'] in [FeatureType.STATIC, FeatureType.DYNAMIC]:
                        self.dataset_info[f'num_{data['type']}_{feature_class}'] += 1

            feature_metadata[feature_class] = local_metadata

        return feature_metadata

    def get_features(self, feature_class: FeatureClass) -> Tuple[torch.Tensor, torch.Tensor, list]:
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

    def format_features(self, features: dict) -> Tuple[torch.Tensor, torch.Tensor, list]:
        # Static features = (num_items, num_features)
        static_features = torch.from_numpy(np.array(features[FeatureType.STATIC]))
        if len(static_features) > 0:
            static_features = static_features.transpose(1, 0)

        # Dynamic features = (num_timesteps, num_items, num_features)
        dynamic_features = torch.from_numpy(np.array(features[FeatureType.DYNAMIC]))
        if len(dynamic_features) > 0:
            dynamic_features = dynamic_features.permute([1, 2, 0])

        # Raw features
        raw_features = features[FeatureType.RAW]

        return static_features, dynamic_features, raw_features
