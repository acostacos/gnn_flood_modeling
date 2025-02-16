import torch

from constants import Activation
from torch import Tensor
from torch.linalg import vector_norm
from torch.nn import Module, ModuleList, Sequential, Identity
from torch_geometric.data import Data
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
                 num_layers: int = 1,
                 num_hops: int = 8,
                 mlp_layers: int = 2,
                 activation: Activation = Activation.PRELU,
                 residual: bool = True,
                 dropout=0, # TODO: Check if you need this

                 # Encoder Decoder Parameters
                 encoder_layers: int = 2,
                 encoder_activation: Activation = Activation.PRELU,
                 decoder_layers: int = 2,
                 decoder_activation: Activation = Activation.PRELU,
                 **base_model_kwargs):
        super().__init__(**base_model_kwargs)
        self.with_encoder = encoder_layers > 0
        self.with_decoder = decoder_layers > 0

        # Encoder
        if self.with_encoder:
            self.edge_encoder = make_mlp(input_size=self.input_edge_features, output_size=hidden_features,
                                        hidden_size=hidden_features, num_layers=encoder_layers,
                                        activation=encoder_activation, device=self.device)
            self.static_node_encoder = make_mlp(input_size=self.static_node_features, output_size=hidden_features,
                                            hidden_size=hidden_features, num_layers=encoder_layers,
                                            activation=encoder_activation, device=self.device)
            # No bias for dynamic features
            total_dynamic_node_feats = self.dynamic_node_features * (self.previous_timesteps+1)
            self.dynamic_node_encoder = make_mlp(input_size=total_dynamic_node_feats, output_size=hidden_features,
                                                hidden_size=hidden_features, num_layers=encoder_layers,
                                                activation=encoder_activation, bias=False, device=self.device)

        # Processor = GNN
        static_node_features = hidden_features if self.with_encoder else self.static_node_features
        dynamic_node_features = hidden_features if self.with_encoder else self.dynamic_node_features
        edge_features = hidden_features if self.with_encoder else self.input_edge_features
        self.gnn_processors = self._make_gnns(static_node_features=static_node_features, dynamic_node_features=dynamic_node_features,
                                              edge_features=edge_features, K_hops=num_hops,
                                              num_layers=num_layers, mlp_layers=mlp_layers, mlp_activation=activation)
        self.gnn_activations = Sequential(*([get_activation_func(activation, device=self.device)] * num_layers))

        # Decoder
        # No bias for dynamic features
        if self.with_decoder:
            self.node_decoder = make_mlp(input_size=hidden_features, output_size=self.output_node_features,
                                        hidden_size=hidden_features, num_layers=decoder_layers,
                                        activation=decoder_activation, bias=False, device=self.device)
        
        if residual:
            self.residual = Identity()


    def _make_gnns(self, static_node_features: int, dynamic_node_features: int, edge_features: int,
                   K_hops: int, num_layers: int, mlp_layers: int, mlp_activation: str):
        """Builds GNN module"""
        convs = ModuleList()
        for _ in range(num_layers):
            convs.append(SWEGNNProcessor(static_node_features=static_node_features, dynamic_node_features=dynamic_node_features,
                                         edge_features=edge_features, encoded=self.with_encoder, previous_timesteps=self.previous_timesteps,
                                         K=K_hops, mlp_layers=mlp_layers, mlp_activation=mlp_activation, device=self.device))
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

        # 1. Node and edge encoder
        if self.with_encoder:
            edge_attr = self.edge_encoder(edge_attr)
            x_s = self.static_node_encoder(x_s)
            x = x_t = self.dynamic_node_encoder(x_t)

        # 2. Processor 
        for i, conv in enumerate(self.gnn_processors):
            x = conv(x_s, x_t, edge_index, edge_attr)

            # Add non-linearity
            x = self.gnn_activations[i](x)

            x_t = x

        # 3. Decoder
        if self.with_decoder:
            x = self.node_decoder(x)

        # Add residual connections
        if hasattr(self, 'residual'):
            x = x + self.residual(x0[:, -self.dynamic_node_features:])

        # Mask very small water depth
        x = self._mask_small_WD(x, epsilon=0.001)

        return x

    def _mask_small_WD(self, x, epsilon=0.001):        
        x[:,0][x[:,0].abs() < epsilon] = 0

        # Mask velocities where there is no water
        x[:,1:][x[:,0] == 0] = 0

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
                 encoded: bool = True,
                 previous_timesteps: int = 0,
                 K: int = 8,
                 mlp_layers: int = 2,
                 mlp_activation: str = 'prelu',
                 device='cpu'):
        super().__init__()
        self.K = K

        edge_input_size = edge_features + static_node_features * 2 + dynamic_node_features * 2
        edge_output_size = dynamic_node_features
        edge_hidden_size = dynamic_node_features * 2
        self.edge_mlp = make_mlp(input_size=edge_input_size, output_size=edge_output_size,
                                hidden_size=edge_hidden_size, num_layers=mlp_layers,
                                activation=mlp_activation, bias=True, device=device)

        self.filter_matrix = ModuleList([
            make_mlp(input_size=dynamic_node_features, output_size=dynamic_node_features,
                     bias=False, device=device) for _ in range(K+1)
        ])

        filter_input_size = dynamic_node_features if encoded else (dynamic_node_features * (previous_timesteps+1))
        self.filter_matrix = ModuleList(
            [
                make_mlp(input_size=filter_input_size, output_size=dynamic_node_features, # Initial filter matrix
                     bias=False, device=device),
                *[make_mlp(input_size=dynamic_node_features, output_size=dynamic_node_features,
                        bias=False, device=device) for _ in range(K)],
            ]
        )

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

