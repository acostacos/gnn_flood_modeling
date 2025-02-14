from torch import Tensor
from torch.nn import Identity
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, Sequential
from utils.model_utils import make_mlp, get_activation_func

from .base_model import BaseModel

class GAT(BaseModel):
    '''
    GAT
    GNN utlizing attention mechanism for edges
    '''
    def __init__(self,
                 hidden_features: int = 64,
                 encoder_layers: int = 2,
                 encoder_activation: str = 'prelu',
                 gnn_layers: int = 1,
                 gnn_activation: str = 'prelu',
                 **base_model_kwargs):
        super().__init__(**base_model_kwargs)

        # Encoder
        with_encoder_decoder = encoder_layers > 0
        if with_encoder_decoder:
            node_features = self.static_node_features + (self.dynamic_node_features * (self.previous_timesteps+1))
            self.node_encoder = make_mlp(input_size=node_features, output_size=hidden_features,
                                                hidden_size=hidden_features, num_layers=encoder_layers,
                                            activation=encoder_activation, device=self.device)

        self.convs = Sequential('x, edge_index',
            ([
                (GATConv(in_channels=hidden_features, out_channels=hidden_features).to(self.device), 'x, edge_index -> x'),
                get_activation_func(gnn_activation, device=self.device),
            ] * gnn_layers)
        )

        # Decoder
        if with_encoder_decoder:
            self.node_decoder = make_mlp(input_size=hidden_features, output_size=self.dynamic_node_features,
                                        hidden_size=hidden_features, num_layers=encoder_layers,
                                        activation=encoder_activation, bias=False, device=self.device)

        self.residual = Identity()

    def forward(self, graph: Data) -> Tensor:
        x = graph.x.clone()
        edge_index = graph.edge_index.clone()

        x0 = x

        x = self.node_encoder(x)
        x = self.convs(x, edge_index)
        x = self.node_decoder(x)

        if self.residual:
            x = x + self.residual(x0[:, -self.dynamic_node_features:])

        return x

