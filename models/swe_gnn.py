import torch

from torch import Tensor
from torch.linalg import vector_norm
from torch.nn import Module, ModuleList, Sequential
from torch_scatter import scatter
from utils.model_utils import make_mlp, get_activation_func

from .base_model import BaseModel

class SWEGNN(BaseModel):
    '''
    SWE_GNN
    https://hess.copernicus.org/articles/27/4227/2023/
    Physics-informed, Encoder-Processor-Decoder
    Based on: https://github.com/RBTV1/SWE-GNN-paper-repository-
    '''
    def __init__(self,
                 hidden_features: int = 64,
                 mlp_layers: int = 2,
                 mlp_activation: str = 'prelu',
                 gnn_layers: int = 1,
                 gnn_activation: str = 'prelu',
                 num_message_pass: int = 8,
                 dropout=0,
                 **base_model_kwargs):
        # TODO: Enforce me
        super().__init__(**base_model_kwargs)

        # Encoder
        edge_features = self.static_edge_features + self.dynamic_edge_features
        self.edge_encoder = make_mlp(input_size=edge_features, output_size=hidden_features,
                                     hidden_size=hidden_features, num_layers=mlp_layers,
                                     activation=mlp_activation, device=self.device)
        self.static_node_encoder = make_mlp(input_size=self.static_node_features, output_size=hidden_features,
                                            hidden_size=hidden_features, num_layers=mlp_layers,
                                            activation=mlp_activation, device=self.device)
        # No bias for dynamic features
        self.dynamic_node_encoder = make_mlp(input_size=self.dynamic_node_features, output_size=hidden_features,
                                             hidden_size=hidden_features, num_layers=mlp_layers,
                                             activation=mlp_activation, bias=False, device=self.device)

        # Processor = GNN
        self.gnn_processors = self._make_gnns(hidden_size=hidden_features, K_hops=num_message_pass,
                                              num_layers=gnn_layers, mlp_layers=mlp_layers, mlp_activation=mlp_activation)

        # TODO: Check if this returns correct list
        self.gnn_activations = Sequential([get_activation_func(gnn_activation, device=self.device)] * gnn_layers)

        # Decoder
        # No bias for dynamic features
        self.node_decoder = make_mlp(input_size=hidden_features, output_size=self.dynamic_node_features,
                                     hidden_size=hidden_features, num_layers=mlp_layers,
                                     activation=mlp_activation, bias=False, device=self.device)

    def _make_gnns(self, hidden_size: int, K_hops: int, num_layers: int, mlp_layers: int, mlp_activation: str):
        """Builds GNN module"""
        convs = ModuleList()
        edge_features = self.static_edge_features + self.dynamic_edge_features
        for _ in range(num_layers):
            convs.append(SWEGNNProcessor(static_node_features=hidden_size, dynamic_node_features=hidden_size, # Because of encoder
                                         edge_features=edge_features, K=K_hops, mlp_layers=mlp_layers,
                                          mlp_activation=mlp_activation, device=self.device,))
        return convs

    def forward(self, graph):
        x = graph.x.clone()
        edge_index = graph.edge_index.clone()
        edge_attr = graph.edge_attr.clone()
        
        x = self._forward_block(x, edge_index, edge_attr)
        
        return x

    def _forward_block(self, x, edge_index, edge_attr):
        """Build encoder-decoder block"""
        # 1. Node and edge encoder
        edge_attr = self.edge_encoder(edge_attr)
        
        x0 = x
        x_s = x[:, :self.static_node_features]
        x_t = x[:, self.static_node_features:]

        x_s = self.static_node_encoder(x_s)
        x = x_t = self.dynamic_node_encoder(x_t)

        # 2. Processor 
        for i, conv in enumerate(self.gnn_processor):
            x = conv(x_s, x_t, edge_index, edge_attr)

            # Add non-linearity
            x = self.gnn_activations[i](x)

            x_t = x

        # 3. Decoder
        x = self.node_decoder(x)

        # Add residual connections
        x = x + x0[:, -self.dynamic_node_features:]

        # Mask very small water depth
        x = self._mask_small_WD(x, epsilon=0.001)

        return x

class SWEGNNProcessor(Module):
    r"""Shallow Water Equations inspired Graph Neural Network

    .. math::
        \mathbf{x}^{\prime}_ti = \mathbf{x}_ti + \sum_{j \in \mathcal{N}(i)} 
        \mathbf{w}_{ij} \cdot (\mathbf{x}_tj - \mathbf{x}_ti)

        \mathbf{w}_{ij} = MLP \left(\mathbf{x}_si, \mathbf{x}_sj,
        \mathbf{x}_ti, \mathbf{x}_tj,
        \mathbf{e}_{ij}\right)
    """
    def __init__(self,
                 static_node_features: int,
                 dynamic_node_features: int,
                 edge_features: int,
                 K: int = 8,
                 mlp_layers: int = 2,
                 mlp_activation: str = 'prelu',
                 device='cpu'):
        super().__init__()
        self.edge_features = edge_features
        self.edge_input_size = edge_features + static_node_features * 2 + dynamic_node_features * 2
        self.edge_output_size = dynamic_node_features
        self.K = K

        hidden_size = self.edge_output_size * 2
        self.edge_mlp = make_mlp(input_size=self.edge_input_size, output_size=self.edge_output_size,
                                hidden_size=hidden_size, num_layers=mlp_layers,
                                activation=mlp_activation, bias=True, device=device)

        # TODO: Check if this only returns list of Linear
        self.filter_matrix = ModuleList([
            make_mlp(input_size=dynamic_node_features, output_size=dynamic_node_features, bias=False) for _ in range(K+1)
        ])

    def forward(self, x_s: Tensor, x_t: Tensor, edge_index: Tensor, 
                edge_attr: Tensor) -> Tensor:
        '''x_s: static node features,
        x_t: dynamic node features'''
        from_node, to_node = edge_index
        num_nodes = x_t.size(0)

        # Why do we multiply with filter_matrix at the start?
        out = self.filter_matrix[0].forward(x_t.clone())

        for k in range(self.K):
            # Filter out zero values
            mask = out.sum(1) != 0
            mask_from_node = mask[from_node]
            mask_to_node = mask[to_node]
            edge_index_mask = mask_from_node + mask_to_node

            # Edge update / Aggregate
            e_ij = torch.cat([x_s[from_node][edge_index_mask], 
                            x_s[to_node][edge_index_mask], 
                            out[from_node][edge_index_mask], 
                            out[to_node][edge_index_mask], 
                            edge_attr[edge_index_mask]], 1)
            w_ij = self.edge_mlp(e_ij)

            # Normalization 
            w_ij = w_ij / vector_norm(w_ij, dim=1, keepdim=True)
            w_ij.masked_fill_(torch.isnan(w_ij), 0)

            # Node update
            # Dynamic node features difference term
            shift_sum = (out[to_node][edge_index_mask] - out[from_node][edge_index_mask]) * w_ij

            scattered = scatter(shift_sum, to_node[edge_index_mask], reduce='sum', 
                          dim=0, dim_size=num_nodes)

            # Multiple with weigthed parameter matrix
            out = out + self.filter_matrix[k+1].forward(scattered)
        
        return out

    def __repr__(self):
        return '{}(node_features={}, edge_features={}, K={})'.format(
            self.__class__.__name__, self.edge_output_size, self.edge_features, self.K)

