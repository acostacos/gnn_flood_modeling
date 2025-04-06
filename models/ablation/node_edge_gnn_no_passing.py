from torch import Tensor
from torch_geometric.nn import MessagePassing, Sequential as PygSequential
from utils.model_utils import make_mlp

from ..node_edge_gnn import NodeEdgeGNN

class NodeEdgeGNNNoPassing(NodeEdgeGNN):
    '''
    Ablation study with no Message Passing, just a simple MLP for both node and edge features.
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _make_gnn(self, input_node_size: int, output_node_size: int, input_edge_size: int, output_edge_size: int,
                  num_layers: int, mlp_layers: int, activation: str, device: str):
        if num_layers == 1:
            return NoEdgeConv(input_node_size, output_node_size, input_edge_size, output_edge_size,
                                num_layers=mlp_layers, activation=activation, device=device)

        layers = []
        for _ in range(num_layers):
            layers.append((
                NoEdgeConv(input_node_size, output_node_size, input_edge_size, output_edge_size,
                             num_layers=mlp_layers, activation=activation, device=device),
                'x, edge_index, edge_attr -> x, edge_attr',
            ))
        return PygSequential('x, edge_index, edge_attr', layers)

class NoEdgeConv(MessagePassing):
    """
    No message passing, just a simple MLP for both node and edge features.
    """
    def __init__(self, node_in_channels: int, node_out_channels: int, edge_in_channels: int, edge_out_channels: int,
                 num_layers: int = 2, activation: str = 'prelu', device: str = 'cpu'):
        super().__init__(aggr='sum')
        msg_input_size = edge_in_channels
        msg_hidden_size = msg_input_size * 2
        self.msg_mlp = make_mlp(input_size=msg_input_size, output_size=edge_out_channels,
                            hidden_size=msg_hidden_size, num_layers=num_layers,
                            activation=activation, device=device)

        update_input_size = node_in_channels
        update_hidden_size = update_input_size * 2
        self.update_mlp = make_mlp(input_size=update_input_size, output_size=node_out_channels,
                                 hidden_size=update_hidden_size, num_layers=num_layers,
                                 activation=activation, device=device)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor):
        x, edge_attr = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return x, edge_attr

    def propagate(self, edge_index, **kwargs):
        coll_dict = self._collect(self._user_args, edge_index, [None, None], kwargs)

        msg_kwargs = self.inspector.collect_param_data('message', coll_dict)
        msg = self.message(**msg_kwargs)

        update_kwargs = self.inspector.collect_param_data('update', coll_dict)
        out = self.update(None, **update_kwargs)
        return out, msg

    def message(self, edge_attr: Tensor):
        return self.msg_mlp(edge_attr)

    def update(self, aggr_out: Tensor, x: Tensor):
        return self.update_mlp(x)

