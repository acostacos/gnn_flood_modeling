from torch_geometric.data import Data
from torch_geometric.nn import Sequential, GCNConv
from utils.model_utils import make_mlp, get_activation_func

from .base_model import BaseModel

class GCN(BaseModel):
    '''
    GCN
    Most Basic Graph Neural Network w/ Encoder-Decoder
    '''
    def __init__(self,
                 hidden_features: int = 64,
                 mlp_layers: int = 2,
                 mlp_activation: str = 'prelu',
                 gnn_layers: int = 1,
                 gnn_activation: str = 'prelu',
                 **base_model_kwargs):
        super().__init__(**base_model_kwargs)

        # Encoder
        node_features = self.static_node_features + self.dynamic_node_features
        self.node_encoder = make_mlp(input_size=node_features, output_size=hidden_features,
                                            hidden_size=hidden_features, num_layers=mlp_layers,
                                            activation=mlp_activation, device=self.device)

        self.convs = Sequential(
            *([
                GCNConv(in_channels=hidden_features, out_channels=hidden_features),
                get_activation_func(gnn_activation, device=self.device),
            ] * gnn_layers)
        )

        # Decoder
        self.node_decoder = make_mlp(input_size=hidden_features, output_size=self.dynamic_node_features,
                                     hidden_size=hidden_features, num_layers=mlp_layers,
                                     activation=mlp_activation, bias=False, device=self.device)

    def forward(self, graph: Data):
        x = graph.x.clone()
        edge_index = graph.edge_index.clone()

        x = self.node_encoder(x)
        x = self.convs(x, edge_index)
        x = self.node_decoder(x)

        return x

