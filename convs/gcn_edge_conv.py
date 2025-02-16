import torch

from torch import Tensor
from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

class GCNEdgeConv(MessagePassing):
    '''GCN Convolution with edge features'''
    def __init__(self, in_channels, out_channels, bias=True, device='cpu'):
        super().__init__(aggr='add')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.device = device

        self.lin = Linear(in_channels, out_channels, bias=False, device=device)
        if bias:
            self.bias = Parameter(torch.empty(out_channels)).to(device)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        if hasattr(self, 'bias'):
            self.bias.data.zero_()

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor):
        # Add loops to adjacency matrix
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        # Add empty edge attributes to loops
        padding = torch.zeros(x.size(0), edge_attr.size(1), device=self.device)
        edge_attr = torch.cat([edge_attr, padding], dim=0)

        # Linearly transform node feature matrix
        x = self.lin(x)

        # Normalize with inverse sqrt of degree
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Propagate messages
        out = self.propagate(edge_index, x=x, norm=norm, edge_attr=edge_attr)

        # Add bias
        if hasattr(self, 'bias'):
            out = out + self.bias

        return out

    def message(self, x_j: Tensor, norm: Tensor, edge_attr: Tensor):
        # Add edge features to node features
        return norm.view(-1, 1) * (x_j + edge_attr)

    def __repr__(self):
        return f'{self.__class__.__name__}(in_channels={self.in_channels}, out_channels={self.out_channels})'
