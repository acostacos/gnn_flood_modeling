import torch

from constants import FeatureClass
from torch_geometric.data import Data
from torch_geometric.transforms import ToUndirected

from .base_dataset import BaseDataset


class EdgeRegressionDataset(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def load(self) -> list[Data]:
        _, _, raw_graph = self.get_features(FeatureClass.GRAPH)
        static_nodes, dynamic_nodes, _ = self.get_features(FeatureClass.NODE)
        static_edges, dynamic_edges, _ = self.get_features(FeatureClass.EDGE)

        timesteps, edge_index, pos = raw_graph
        _, num_nodes, num_df_nodes = dynamic_nodes.shape
        _, num_edges, num_df_edges = dynamic_edges.shape

        dataset = []
        prev_ts_df_nodes = torch.zeros((num_nodes, num_df_nodes * self.previous_timesteps))
        prev_ts_df_edges = torch.zeros((num_edges, num_df_edges * self.previous_timesteps))
        for i in range(len(timesteps)-1): # Last time step is only used as a label
            ts = timesteps[i]
            ts_df_nodes = dynamic_nodes[i]
            ts_df_edges = dynamic_edges[i]
            next_ts_df_edges = dynamic_edges[i+1]

            # Convention = [static_features, previous_dynamic_features, current_dynamic_features]
            node_features = torch.cat([static_nodes, prev_ts_df_nodes, ts_df_nodes], dim=1)
            edge_features = torch.cat([static_edges, prev_ts_df_edges, ts_df_edges], dim=1)
            data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_features, y=next_ts_df_edges, pos=pos)

            # Transform to undirected graph
            data = ToUndirected()(data)

            dataset.append(data)

            # Replace previous timestep dynamic features
            prev_ts_df_nodes = torch.cat([ts_df_nodes, prev_ts_df_nodes[:, :-num_df_nodes]], dim=1)
            prev_ts_df_edges = torch.cat([ts_df_edges, prev_ts_df_edges[:, :-num_df_edges]], dim=1)

        return dataset, self.dataset_info
