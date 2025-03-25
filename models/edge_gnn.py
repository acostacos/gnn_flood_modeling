import torch

from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing, Sequential as PygSequential
from utils.model_utils import make_mlp
from .base_model import BaseModel

class EdgeGNN(BaseModel):
    def __init__(self,
                 input_features: int = None,
                 output_features: int = None,
                 hidden_features: int = 32,
                 num_layers: int = 1,
                 activation: str = 'prelu',
                 residual: bool = True,

                 # Encoder Decoder Parameters
                 encoder_layers: int = 2,
                 encoder_activation: str = 'prelu',
                 decoder_layers: int = 2,
                 decoder_activation: str = 'prelu',

                 **base_model_kwargs):
        super().__init__(**base_model_kwargs)
        assert encoder_layers > 0
        assert decoder_layers > 0

        if output_features is None:
            output_features = self.output_edge_features

        # Encoder
        self.node_encoder = make_mlp(input_size=self.input_node_features, output_size=hidden_features,
                                            hidden_size=hidden_features, num_layers=encoder_layers,
                                        activation=encoder_activation, device=self.device)
        self.edge_encoder = make_mlp(input_size=self.input_edge_features, output_size=hidden_features,
                                            hidden_size=hidden_features, num_layers=encoder_layers,
                                        activation=encoder_activation, device=self.device)

        # Processor
        self.convs = self._make_gnn(num_layers=num_layers, hidden_features=hidden_features, device=self.device)

        # Decoder
        self.edge_decoder = make_mlp(input_size=hidden_features, output_size=output_features,
                                    hidden_size=hidden_features, num_layers=decoder_layers,
                                    activation=decoder_activation, bias=False, device=self.device)

        # if residual:
        #     self.residual = Identity()

    def _make_gnn(self, num_layers, hidden_features, device):
        if num_layers == 1:
            return EdgeGNNConv(hidden_features, hidden_features, device=device)

        layers = []
        for _ in range(num_layers):
            layers.append(
                (EdgeGNNConv(hidden_features, hidden_features, device=device), 'x, edge_index, edge_attr -> x, edge_attr')
            )
        return PygSequential('x, edge_index, edge_attr', layers)

    def forward(self, graph: Data) -> Tensor:
        x, edge_index, edge_attr = graph.x.clone(), graph.edge_index.clone(), graph.edge_attr.clone()

        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)

        x, edge_attr = self.convs(x, edge_index, edge_attr)

        out = self.edge_decoder(edge_attr)

        return out

class EdgeGNNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, num_layers: int = 2,
                 activation: str = 'relu', device: str = 'cpu'):
        super().__init__(aggr='sum')

        input_size = in_channels * 2
        hidden_size = input_size * 2
        self.mlp = make_mlp(input_size=input_size, output_size=out_channels,
                            hidden_size=hidden_size, num_layers=num_layers,
                            activation=activation, device=device)
        self.edge_mlp = make_mlp(input_size=input_size, output_size=out_channels,
                                 hidden_size=hidden_size, num_layers=num_layers,
                                 activation=activation, device=device)

    def forward(self, x, edge_index, edge_attr):
        out_nodes = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        out_edges = self.edge_updater(edge_index, x=x)
        return out_nodes, out_edges

    def message(self, x_j, edge_attr):
        features = torch.cat([x_j, edge_attr], dim=-1)
        return self.mlp(features)

    def edge_update(self, x_i, x_j):
        edge_features = torch.cat([x_i, x_j], dim=1)
        return self.edge_mlp(edge_features)
