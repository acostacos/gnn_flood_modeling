from torch import Tensor
from torch.nn import Module, Sequential, Linear, PReLU, ReLU
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, Sequential as PygSequential

def make_mlp(input_size: int, output_size: int, hidden_size: int = None,
             num_layers: int = 1, activation: str = None, bias: bool = True,
             device: str = 'cpu') -> Module:
    if num_layers == 1:
        return LinearLayer(input_size, output_size, activation, bias, device)

    hidden_size = hidden_size if hidden_size is not None else (input_size * 2)
    layers = []
    layers.append(LinearLayer(input_size, hidden_size, activation, bias, device)) # Input Layer
    for _ in range(num_layers-2):
        layers.append(LinearLayer(hidden_size, hidden_size, activation, bias, device)) # Hidden Layers
    layers.append(LinearLayer(hidden_size, output_size, activation, bias, device)) # Output Layer
    return Sequential(*layers)

def make_gnn(input_size: int, output_size: int, hidden_size: int = None,
             num_layers: int = 1, conv: str = 'gcn',
             activation: str = None, use_edge_attr: bool = False, device: str = 'cpu', **conv_kwargs) -> Module:
    if num_layers == 1:
        return GNNLayer(input_size, output_size, conv, activation, device, **conv_kwargs)

    hidden_size = hidden_size if hidden_size is not None else (input_size * 2)
    layer_schema = 'x, edge_index -> x' if not use_edge_attr else 'x, edge_index, edge_attr -> x'
    input_schema = 'x, edge_index' if not use_edge_attr else 'x, edge_index, edge_attr'
    layers = []
    layers.append(
        (GNNLayer(input_size, hidden_size, conv, activation, use_edge_attr, device, **conv_kwargs), layer_schema)
    ) # Input Layer
    for _ in range(num_layers-2):
        layers.append(
            (GNNLayer(hidden_size, hidden_size, conv, activation, use_edge_attr, device, **conv_kwargs), layer_schema)
        ) # Hidden Layers
    layers.append(
        (GNNLayer(hidden_size, output_size, conv, activation, use_edge_attr, device, **conv_kwargs), layer_schema)
    ) # Output Layer
    return PygSequential(input_schema, layers)

def get_activation_func(name: str, device: str = 'cpu') -> Module:
    if name == 'relu':
        return ReLU()
    if name == 'prelu':
        return PReLU(device=device)
    raise Exception(f'Activation function {name} is not implemented.')

class LinearLayer(Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 activation: str = None,
                 bias: bool = True,
                 device: str = 'cpu'):
        super().__init__()
        self.linear = Linear(in_features=in_features, out_features=out_features, bias=bias, device=device)
        if activation is not None:
            self.activation = get_activation_func(activation, device=device)

    def forward(self, x: Tensor) -> Tensor:
        x = self.linear(x)
        if hasattr(self, 'activation'):
            x = self.activation(x)
        return x

class GNNLayer(Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 conv: str = 'gcn',
                 activation: str = None,
                 use_edge_attr: bool = False,
                 device: str = 'cpu',
                 **conv_kwargs):
        super().__init__()
        self.conv = self._get_conv(conv, in_features, out_features, **conv_kwargs).to(device)
        if activation is not None:
            self.activation = get_activation_func(activation, device=device)
        self.use_edge_attr = use_edge_attr

    def _get_conv(self, conv: str, in_features: int, out_features: int, **conv_kwargs) -> Module:
        if conv == 'gcn':
            return GCNConv(in_channels=in_features, out_channels=out_features, **conv_kwargs)
        if conv == 'gat':
            return GATConv(in_channels=in_features, out_channels=out_features, **conv_kwargs)
        if conv == 'sage':
            in_features = (-1, -1) if in_features is None else in_features
            return SAGEConv(in_channels=in_features, out_channels=out_features, **conv_kwargs)
        raise Exception(f'GNN Convolution {conv} is not implemented.')

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor = None) -> Tensor:
        if self.use_edge_attr:
            x = self.conv(x, edge_index, edge_attr)
        else:
            x = self.conv(x, edge_index)

        if hasattr(self, 'activation'):
            x = self.activation(x)
        return x
