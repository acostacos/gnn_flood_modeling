
from constants import Activation
from torch import Tensor
from torch.nn import Linear, Identity
from torch_geometric.nn import GINEConv, Sequential as PygSequential
from torch_geometric.data import Data
from utils.model_utils import make_mlp

from .base_model import BaseModel

class GINE(BaseModel):
    '''
    GINE
    '''
    def __init__(self,
                 input_features: int = None,
                 output_features: int = None,
                 hidden_features: int = None,
                 num_layers: int = 1,
                 activation: Activation = Activation.PRELU,
                 residual: bool = True,

                 # Encoder Decoder Parameters
                 encoder_layers: int = 2,
                 encoder_activation: Activation = Activation.PRELU,
                 decoder_layers: int = 2,
                 decoder_activation: Activation = Activation.PRELU,

                 **base_model_kwargs):
        super().__init__(**base_model_kwargs)
        if input_features is None:
            input_features = self.input_node_features
        if output_features is None:
            output_features = self.output_node_features

        # Encoder
        self.node_encoder = make_mlp(input_size=input_features, output_size=hidden_features,
                                            hidden_size=hidden_features, num_layers=encoder_layers,
                                        activation=encoder_activation, device=self.device)
        self.edge_encoder = make_mlp(input_size=self.input_edge_features, output_size=hidden_features,
                                            hidden_size=hidden_features, num_layers=encoder_layers,
                                        activation=encoder_activation, device=self.device)

        # Processor
        self.convs = self._make_gnn(num_layers=num_layers, hidden_features=hidden_features, device=self.device)

        # Decoder
        self.node_decoder = make_mlp(input_size=hidden_features, output_size=output_features,
                                    hidden_size=hidden_features, num_layers=decoder_layers,
                                    activation=decoder_activation, bias=False, device=self.device)

        if residual:
            self.residual = Identity()

    def _make_gnn(self, num_layers, hidden_features, device):
        if num_layers == 1:
            nn = Linear(hidden_features, hidden_features)
            return GINEConv(nn, edge_dim=hidden_features).to(device)

        layers = []
        for _ in range(num_layers):
            nn = Linear(hidden_features, hidden_features)
            layers.append(
                (GINEConv(nn, edge_dim=hidden_features).to(device), 'x, edge_index, edge_attr -> x')
            )
        return PygSequential('x, edge_index, edge_attr', layers)

    def forward(self, graph: Data) -> Tensor:
        x, edge_index, edge_attr = graph.x.clone(), graph.edge_index.clone(), graph.edge_attr.clone()
        x0 = x

        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)

        x = self.convs(x, edge_index, edge_attr)

        x = self.node_decoder(x)

        if hasattr(self, 'residual'):
            x = x + self.residual(x0[:, -self.output_node_features:])

        return x

