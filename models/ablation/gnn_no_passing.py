from torch import Tensor
from torch.nn import Identity
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing, Sequential as PygSequential
from utils.model_utils import make_mlp, get_activation_func

from ..base_model import BaseModel

class GNNNoPassing(BaseModel):
    '''
    Model that uses a GNN with no message passing operation. Essentially an MLP.
    '''
    def __init__(self,
                 hidden_features: int = None,
                 num_layers: int = 1,
                 activation: str = 'prelu',
                 residual: bool = True,
                 mlp_layers: int = 2,

                 # Encoder Decoder Parameters
                 encoder_layers: int = 0,
                 encoder_activation: str = None,
                 decoder_layers: int = 0,
                 decoder_activation: str = None,

                 **base_model_kwargs):
        super().__init__(**base_model_kwargs)
        self.with_encoder = encoder_layers > 0
        self.with_decoder = decoder_layers > 0
        self.with_residual = residual

        # Encoder
        encoder_decoder_hidden = hidden_features*2
        if self.with_encoder:
            self.node_encoder = make_mlp(input_size=self.input_node_features, output_size=hidden_features,
                                                hidden_size=encoder_decoder_hidden, num_layers=encoder_layers,
                                            activation=encoder_activation, device=self.device)

        input_node_size = hidden_features if self.with_encoder else self.input_node_features
        output_node_size = hidden_features if self.with_decoder else self.output_node_features
        self.convs = self._make_gnn(input_node_size=input_node_size, output_node_size=output_node_size,
                                    num_layers=num_layers, mlp_layers=mlp_layers, activation=activation, device=self.device)

        # Decoder
        if self.with_decoder:
            self.node_decoder = make_mlp(input_size=hidden_features, output_size=self.output_node_features,
                                        hidden_size=encoder_decoder_hidden, num_layers=decoder_layers,
                                        activation=decoder_activation, bias=False, device=self.device)

        if self.with_residual:
            self.residual = Identity()
            self.res_activation = get_activation_func(activation, device=self.device)

    def _make_gnn(self, input_node_size: int, output_node_size: int, num_layers: int, mlp_layers: int,
                  activation: str, device: str):
        if num_layers == 1:
            return NoPassConv(input_node_size, output_node_size, num_layers=mlp_layers,
                              activation=activation, device=device)

        layers = []
        for _ in range(num_layers):
            layers.append((
                NoPassConv(input_node_size, output_node_size, num_layers=mlp_layers,
                           activation=activation, device=device),
                'x, edge_index -> x',
            ))
        return PygSequential('x, edge_index', layers)

    def forward(self, graph: Data) -> Tensor:
        x, edge_index = graph.x.clone(), graph.edge_index.clone()
        x0 = x

        if self.with_encoder:
            x = self.node_encoder(x)

        x = self.convs(x, edge_index)

        if self.with_decoder:
            x = self.node_decoder(x)

        if self.with_residual:
            x = self.res_activation(x + self.residual(x0[:, -self.output_node_features:]))

        return x

class NoPassConv(MessagePassing):
    def __init__(self, node_in_channels: int, node_out_channels: int,
                 num_layers: int = 2, activation: str = 'prelu', device: str = 'cpu'):
        super().__init__(aggr='sum')
        update_input_size = node_in_channels
        update_hidden_size = update_input_size * 2
        self.update_mlp = make_mlp(input_size=update_input_size, output_size=node_out_channels,
                                 hidden_size=update_hidden_size, num_layers=num_layers,
                                 activation=activation, device=device)

    def forward(self, x: Tensor, edge_index: Tensor):
        x = self.propagate(edge_index, x=x)
        return x

    def propagate(self, edge_index, **kwargs):
        coll_dict = self._collect(self._user_args, edge_index, [None, None], kwargs)
        update_kwargs = self.inspector.collect_param_data('update', coll_dict)
        out = self.update(None, **update_kwargs) # No aggregation parameter
        return out

    def update(self, _, x: Tensor):
        return self.update_mlp(x)
