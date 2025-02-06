import geopandas as gpd
import h5py
import json
import numpy as np

from datetime import datetime
from pathlib import Path
from torch_geometric.data import Data
from .transform import TRANSFORM_MAP

class FloodDataset:
    def __init__(self, hec_result_path='', nodes_shape_path='', edges_shape_path='', node_features={}, edge_features={}):
        self.hec_result_path = hec_result_path
        self.nodes_shape_path = nodes_shape_path
        self.edges_shape_path = edges_shape_path
        self.feature_metadata_path = Path(__file__).parent / "feature_metadata.json"

        self.included_features = {}
        self.included_features['node_features'] = self.get_feature_list(node_features)
        self.included_features['edge_features'] = self.get_feature_list(edge_features)

    def load(self):
        timesteps = self.get_timedata()
        edge_index = self.get_edge_index()
        pos = self.get_pos()
        sf_nodes, df_nodes = self.get_features('node_features')
        sf_edges, df_edges = self.get_features('edge_features')

        dataset = []
        for i, ts in enumerate(timesteps):
            ts_df_nodes = df_nodes[i]
            ts_df_edges = df_edges[i]
            node_features = np.concatenate([sf_nodes, ts_df_nodes], axis=1)
            edge_features = np.concatenate([sf_edges, ts_df_edges], axis=1)
            data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_features, pos=pos)
            dataset.append(data)

        return dataset

    def get_feature_list(self, feature: dict[str, bool]):
        return [name for name, is_included in feature.items() if is_included]
    
    def get_features(self, feature_type: str) -> list[np.ndarray, np.ndarray]:
        json_metadata = self.load_json_file(self.feature_metadata_path)
        metadata = {name: data for name, data in json_metadata[feature_type].items() if name in self.included_features[feature_type]}

        static_features = []
        dynamic_features = []
        for name, md in metadata.items():
            if md['file'] == 'hdf':
                data = self.load_hdf_file(filepath=self.hec_result_path, propertypath=md['path'])
            elif md['file'] == 'shp':
                filepath = self.nodes_shape_path if feature_type == 'node_features' else self.edges_shape_path
                data = self.load_shp_file(filepath=filepath, columns=md['column'])
            else:
                raise Exception('Invalid file type in feature metadata. Valid values are: hdf, shp.')

            if name in TRANSFORM_MAP:
                transform_func = TRANSFORM_MAP[name]
                data = transform_func(data)

            if md['type'] == 'static':
                static_features.append(data)
            elif md['type'] == 'dynamic':
                dynamic_features.append(data)

        static_features = np.array(static_features).transpose([1, 0]) # [num_nodes, num_features]
        dynamic_features = np.array(dynamic_features).transpose([1, 2, 0]) # [num_timesteps, num_nodes, num_features]
        return static_features, dynamic_features

    def get_timedata(self) -> np.ndarray:
        TIME_STAMP_FORMAT = '%d%b%Y %H:%M:%S'
        with h5py.File(self.hec_result_path, 'r') as hec:
            time_data = hec['Results']['Unsteady']['Output']['Output Blocks']['Base Output']['Unsteady Time Series'] \
                ['Time Date Stamp']

            time_series = []
            for tstep in time_data:
                time_str = tstep.decode('UTF-8')
                time_stamp = datetime.strptime(time_str, TIME_STAMP_FORMAT)
                time_series.append(time_stamp)

        return np.array(time_series)

    def get_edge_index(self) -> np.ndarray:
        edges = gpd.read_file(self.edges_shape_path)
        edge_index = edges[['from_node', 'to_node']].to_numpy()
        return edge_index

    def get_pos(self) -> np.ndarray:
        nodes = gpd.read_file(self.nodes_shape_path)
        pos = nodes[['X', 'Y']].to_numpy()
        return pos
    
    def load_hdf_file(self, filepath: str, propertypath: str):
        with h5py.File(filepath, 'r') as hec:
            data = self.get_property_from_path(hec, propertypath)
            return np.array(data)

    def get_property_from_path(self, dict: dict, dict_path: str, separator: str = '.'):
        keys = dict_path.split(sep=separator)
        d = dict
        for key in keys:
            if key in d:
                d = d[key]
        return d

    def load_shp_file(self, filepath: str, columns: str | list):
        file = gpd.read_file(filepath)
        return file[columns]

    def load_json_file(self, path: str):
        with open(path, 'r') as file:
            data = json.load(file)
        return data





# def load_data_from_file(file_type: str, **kwargs):
#     if file_type == 'hdf':
#         return load_hdf_file(**kwargs)
#     if file_type == 'shp':
#         return load_shp_file(**kwargs)
#     raise Exception('Invalid file type in feature metadata. Valid values are: hdf, shp.')





