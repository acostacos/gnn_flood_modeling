from constants import Activation, GNNConvolution
from torch import Tensor
from torch.nn import Identity
from torch_geometric.data import Data
from utils.model_utils import make_mlp, make_gnn

from .base_model import BaseModel

class GAT(BaseModel):
    '''
    GAT
    GNN utlizing attention mechanism for edges
    '''
    def __init__(self,
                 input_features: int = None,
                 output_features: int = None,
                 hidden_features: int = None,
                 num_layers: int = 1,
                 activation: Activation = Activation.PRELU,
                 residual: bool = True,
                 encoding: bool = False, # If encoder, add residual and activation for last layer

                 # Attention Parameters
                 num_heads: int = 1,
                 nhead_out: int = None,
                 concat: bool = True,
                 dropout: float = 0.0,
                 negative_slope: float = 0.2,
                 attn_residual: bool = True,


                 # Encoder Decoder Parameters
                 encoder_layers: int = 0,
                 encoder_activation: Activation = None,
                 decoder_layers: int = 0,
                 decoder_activation: Activation = None,

                 **base_model_kwargs):
        super().__init__(**base_model_kwargs)
        self.with_encoder = encoder_layers > 0
        self.with_decoder = decoder_layers > 0

        if input_features is None:
            input_features = self.input_node_features
        if output_features is None:
            output_features = self.output_node_features
        # if nhead_out is None:
        #     nhead_out = num_heads


        input_size = hidden_features if self.with_encoder else input_features
        output_size = hidden_features if self.with_decoder else output_features

        # Encoder
        if self.with_encoder:
            self.node_encoder = make_mlp(input_size=self.input_node_features, output_size=hidden_features,
                                                hidden_size=hidden_features, num_layers=encoder_layers,
                                            activation=encoder_activation, device=self.device)

        self.convs = make_gnn(input_size=input_size, output_size=output_size,
                              hidden_size=hidden_features, num_layers=num_layers,
                              conv=GNNConvolution.GAT, activation=activation, device=self.device,
                              heads=num_heads, concat=concat, dropout=dropout, negative_slope=negative_slope,
                              residual=attn_residual)

        # Decoder
        if self.with_decoder:
            self.node_decoder = make_mlp(input_size=hidden_features, output_size=self.output_node_features,
                                        hidden_size=hidden_features, num_layers=encoder_layers,
                                        activation=decoder_activation, bias=False, device=self.device)

        if residual:
            self.residual = Identity()

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

