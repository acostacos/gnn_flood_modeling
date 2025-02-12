import torch

from torch import Tensor
from torch.linalg import vector_norm
from torch.nn import Module, ModuleList, Sequential
from torch_geometric.data import Data
from torch_scatter import scatter
from utils.model_utils import make_mlp, get_activation_func

from .swe_gnn import SWEGNNProcessor
from .base_model import BaseModel

class SWEGNN_NED(BaseModel):
    '''
    SWE_GNN with no encoder and decoder
    '''
    def __init__(self,
                 mlp_layers: int = 2,
                 mlp_activation: str = 'prelu',
                 gnn_layers: int = 1,
                 gnn_activation: str = 'prelu',
                 num_message_pass: int = 8,
                 **base_model_kwargs):
        super().__init__(**base_model_kwargs)

        # Processor = GNN
        edge_features = self.static_edge_features + (self.dynamic_edge_features * (self.previous_timesteps+1))
        self.gnn_processors = self._make_gnns(static_node_features=self.static_node_features, dynamic_node_features=self.dynamic_node_features,
                                              edge_features=edge_features, K_hops=num_message_pass,
                                              num_layers=gnn_layers, mlp_layers=mlp_layers, mlp_activation=mlp_activation)

        self.gnn_activations = Sequential(*([get_activation_func(gnn_activation, device=self.device)] * gnn_layers))

    def _make_gnns(self, static_node_features: int, dynamic_node_features: int, edge_features: int, K_hops: int, num_layers: int, mlp_layers: int, mlp_activation: str):
        """Builds GNN module"""
        convs = ModuleList()
        for _ in range(num_layers):
            convs.append(SWEGNNProcessorNED(static_node_features=static_node_features, dynamic_node_features=dynamic_node_features,
                                         edge_features=edge_features, K=K_hops, mlp_layers=mlp_layers,
                                          mlp_activation=mlp_activation, device=self.device))
        return convs

    def forward(self, graph: Data) -> Tensor:
        x = graph.x.clone()
        edge_index = graph.edge_index.clone()
        edge_attr = graph.edge_attr.clone()
        
        x = self._forward_block(x, edge_index, edge_attr)
        
        return x

    def _forward_block(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        """Build encoder-decoder block"""
        x0 = x
        x_s = x[:, :self.static_node_features]
        x_t = x[:, self.static_node_features:]

        # 2. Processor 
        for i, conv in enumerate(self.gnn_processors):
            x = conv(x_s, x_t, edge_index, edge_attr)

            # Add non-linearity
            x = self.gnn_activations[i](x)

            x_t = x

        # Add residual connections
        x = x + x0[:, -self.dynamic_node_features:]

        # Mask very small water depth
        x = self._mask_small_WD(x, epsilon=0.001)

        return x

class SWEGNNProcessorNED(SWEGNNProcessor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        K = kwargs['K']
        dynamic_node_features = kwargs['dynamic_node_features']
        device = kwargs['device']

        total_dynamic = dynamic_node_features * 3
        self.filter_matrix = ModuleList(
            [
                make_mlp(input_size=total_dynamic, output_size=dynamic_node_features, # Initial filter matrix
                     bias=False, device=device),
                *[make_mlp(input_size=dynamic_node_features, output_size=dynamic_node_features,
                        bias=False, device=device) for _ in range(K)],
            ]
        )
