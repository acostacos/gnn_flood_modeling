
from torch import Tensor
from torch.nn import Identity
from torch_geometric.nn import GINConv, Sequential as PygSequential
from torch_geometric.data import Data
from utils.model_utils import make_mlp

from .base_model import BaseModel

class GIN(BaseModel):
    '''
    GIN
    GNN using GINConv which focuses on graph isomorphism
    '''
    def __init__(self,
                 input_features: int = None,
                 output_features: int = None,
                 hidden_features: int = None,
                 num_layers: int = 1,
                 activation: str = 'relu',
                 residual: bool = True,

                 # GINConv Parameters
                 mlp_layers: int = 2,
                 eps: float = 0.0,
                 train_eps: bool = False,

                 # Encoder Decoder Parameters
                 encoder_layers: int = 0,
                 encoder_activation: str = None,
                 decoder_layers: int = 0,
                 decoder_activation: str = None,

                 **base_model_kwargs):
        super().__init__(**base_model_kwargs)
        self.with_encoder = encoder_layers > 0
        self.with_decoder = decoder_layers > 0

        if input_features is None:
            input_features = self.input_node_features
        if output_features is None:
            output_features = self.output_node_features

        input_size = hidden_features if self.with_encoder else input_features
        output_size = hidden_features if self.with_decoder else output_features

        # Encoder
        if self.with_encoder:
            self.node_encoder = make_mlp(input_size=input_features, output_size=hidden_features,
                                                hidden_size=hidden_features, num_layers=encoder_layers,
                                            activation=encoder_activation, device=self.device)

        # Processor
        self.convs = self._make_gnn(num_layers=num_layers, input_size=input_size, output_size=output_size,
                                    hidden_features=hidden_features, activation=activation, device=self.device,
                                    mlp_layers=mlp_layers, eps=eps, train_eps=train_eps)

        # Decoder
        if self.with_decoder:
            self.node_decoder = make_mlp(input_size=hidden_features, output_size=output_features,
                                        hidden_size=hidden_features, num_layers=encoder_layers,
                                        activation=decoder_activation, bias=False, device=self.device)

        if residual:
            self.residual = Identity()

    def _make_gnn(self, num_layers, input_size, output_size, hidden_features, activation, device, mlp_layers, eps, train_eps):
        if num_layers == 1:
            nn = make_mlp(input_size, output_size, hidden_features, mlp_layers, activation, device=self.device)
            return GINConv(nn, eps=eps, train_eps=train_eps).to(device)

        layers = []
        nn = make_mlp(input_size, hidden_features, hidden_features, mlp_layers, activation, device=self.device)
        layers.append((GINConv(nn, eps=eps, train_eps=train_eps).to(device), 'x, edge_index -> x')) # Input Layer

        for _ in range(num_layers-2):
            nn = make_mlp(hidden_features, hidden_features, hidden_features, mlp_layers, activation, device=self.device)
            layers.append((GINConv(nn, eps=eps, train_eps=train_eps).to(device), 'x, edge_index -> x')) # Hidden Layers

        nn = make_mlp(hidden_features, output_size, hidden_features, mlp_layers, activation, device=self.device)
        layers.append((GINConv(nn, eps=eps, train_eps=train_eps).to(device), 'x, edge_index -> x')) # Output Layer
        return PygSequential('x, edge_index', layers)

    def forward(self, graph: Data) -> Tensor:
        x, edge_index = graph.x.clone(), graph.edge_index.clone()
        x0 = x

        if self.with_encoder:
            x = self.node_encoder(x)

        x = self.convs(x, edge_index)

        if self.with_decoder:
            x = self.node_decoder(x)

        if hasattr(self, 'residual'):
            x = x + self.residual(x0[:, -self.output_node_features:])

        return x

