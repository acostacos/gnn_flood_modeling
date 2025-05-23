from torch import Tensor
from torch.nn import Identity, Module
from torch_geometric.nn import GATConv, Sequential as PygSequential
from torch_geometric.data import Data
from utils.model_utils import make_mlp, get_activation_func

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
                 use_edge_features: bool = False,
                 input_edge_features: int = None,
                 num_layers: int = 1,
                 activation: str = 'prelu',
                 residual: bool = True,

                 # Attention Parameters
                 num_heads: int = 1,
                 concat: bool = True,
                 dropout: float = 0.0,
                 add_self_loops: bool = True,
                 negative_slope: float = 0.2,
                 attn_bias: bool = True,
                 attn_residual: bool = True,

                 # Encoder Decoder Parameters
                 encoder_layers: int = 0,
                 encoder_activation: str = None,
                 decoder_layers: int = 0,
                 decoder_activation: str = None,

                 **base_model_kwargs):
        super().__init__(**base_model_kwargs)
        self.with_encoder = encoder_layers > 0
        self.with_decoder = decoder_layers > 0
        self.use_edge_features = use_edge_features

        if input_features is None:
            input_features = self.input_node_features
        if output_features is None:
            output_features = self.output_node_features
        if self.use_edge_features and input_edge_features is None:
            input_edge_features = self.input_edge_features

        input_size = hidden_features if self.with_encoder else input_features
        output_size = hidden_features if self.with_decoder else output_features

        # Encoder
        if self.with_encoder:
            self.node_encoder = make_mlp(input_size=input_features, output_size=hidden_features,
                                                hidden_size=hidden_features, num_layers=encoder_layers,
                                            activation=encoder_activation, device=self.device)
            if self.use_edge_features:
                self.edge_encoder = make_mlp(input_size=input_edge_features, output_size=hidden_features,
                                             hidden_size=hidden_features, num_layers=encoder_layers,
                                             activation=encoder_activation, device=self.device)

        conv_kwargs = {
            'heads': num_heads,
            'concat': concat,
            'dropout': dropout,
            'add_self_loops': add_self_loops,
            'negative_slope': negative_slope,
            'bias': attn_bias,
            'residual': attn_residual,
        }
        if self.use_edge_features:
            edge_dim = hidden_features if self.with_encoder else input_edge_features
            conv_kwargs['edge_dim'] = edge_dim
        self.convs = self._make_gnn(input_size=input_size, output_size=output_size,
                              hidden_size=hidden_features, num_layers=num_layers,
                              activation=activation, use_edge_attr=self.use_edge_features,
                              device=self.device, **conv_kwargs)

        # Decoder
        if self.with_decoder:
            self.node_decoder = make_mlp(input_size=hidden_features, output_size=output_features,
                                        hidden_size=hidden_features, num_layers=encoder_layers,
                                        activation=decoder_activation, bias=False, device=self.device)

        if residual:
            self.residual = Identity()

    def _make_gnn(self, input_size: int, output_size: int, hidden_size: int = None,
                num_layers: int = 1, activation: str = None, use_edge_attr: bool = False,
                heads: int = 1, device: str = 'cpu', **conv_kwargs) -> Module:
        is_multihead = heads > 1

        if num_layers == 1:
            if is_multihead:
                assert conv_kwargs.get('concat', False), 'If multihead attention is used, concat must be True for single layer'
            return GATLayer(input_size, output_size, activation, device, **conv_kwargs)

        layer_schema = 'x, edge_index -> x' if not use_edge_attr else 'x, edge_index, edge_attr -> x'
        input_schema = 'x, edge_index' if not use_edge_attr else 'x, edge_index, edge_attr'
        hidden_size = hidden_size if hidden_size is not None else (input_size * 2)

        layers = []
        layers.append(
            (GATLayer(input_size, hidden_size, activation, use_edge_attr, device,
                      heads=heads, **conv_kwargs), layer_schema)
        ) # Input Layer

        for _ in range(num_layers-2):
            layers.append(
                (GATLayer((hidden_size * heads), hidden_size, activation, use_edge_attr, device,
                          heads=heads, **conv_kwargs), layer_schema)
            ) # Hidden Layers

        concat_kwargs = conv_kwargs.pop('concat', True)
        concat = concat_kwargs if not is_multihead else False
        layers.append(
            (GATLayer((hidden_size * heads), output_size, activation, use_edge_attr, device,
                      heads=heads, **conv_kwargs, concat=concat), layer_schema)
        ) # Output Layer
        return PygSequential(input_schema, layers)

    def forward(self, graph: Data) -> Tensor:
        x, edge_index = graph.x.clone(), graph.edge_index.clone()
        if self.use_edge_features:
            edge_attr = graph.edge_attr.clone()
        x0 = x

        if self.with_encoder:
            x = self.node_encoder(x)
            if self.use_edge_features:
                edge_attr = self.edge_encoder(edge_attr)

        if self.use_edge_features:
            x = self.convs(x, edge_index, edge_attr)
        else:
            x = self.convs(x, edge_index)

        if self.with_decoder:
            x = self.node_decoder(x)

        if hasattr(self, 'residual'):
            x = x + self.residual(x0[:, -self.output_node_features:])

        return x

class GATLayer(Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 activation: str = None,
                 use_edge_attr: bool = False,
                 device: str = 'cpu',
                 **conv_kwargs):
        super().__init__()
        self.conv = GATConv(in_channels=in_features, out_channels=out_features, **conv_kwargs).to(device)
        if activation is not None:
            self.activation = get_activation_func(activation, device=device)
        self.use_edge_attr = use_edge_attr

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor = None) -> Tensor:
        if self.use_edge_attr:
            x = self.conv(x, edge_index, edge_attr)
        else:
            x = self.conv(x, edge_index)

        if hasattr(self, 'activation'):
            x = self.activation(x)
        return x
