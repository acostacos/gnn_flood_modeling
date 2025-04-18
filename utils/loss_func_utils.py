from torch import Tensor
from torch.nn import Module, MSELoss, L1Loss
from torch_geometric.data import Data
from typing import Callable, Tuple

class CombinedL1Loss(L1Loss):
    def __init__(self, node_weight: float = 0.5, edge_weight: float = 0.5):
        super().__init__()
        self.node_weight = node_weight
        self.edge_weight = edge_weight

        self.node_loss = 0
        self.edge_loss = 0

    def forward(self, node_pred: Tensor, edge_pred: Tensor, orig_graph: Data):
        node_label = orig_graph.y
        edge_label = orig_graph.y_edge

        self.node_loss = self.node_weight * super().forward(node_pred, node_label)
        self.edge_loss = self.edge_weight * super().forward(edge_pred, edge_label)
        return self.node_loss + self.edge_loss

    def get_loss_components(self) -> Tuple[float, float]:
        return self.node_loss.item(), self.edge_loss.item()

class MassConservationL1Loss(CombinedL1Loss):
    def __init__(self, node_weight: float = 0.5, edge_weight: float = 0.5):
        super().__init__(node_weight=node_weight, edge_weight=edge_weight)

    def forward(self, input: Data, target: Data):
        # Assuming input and target are of shape (batch_size, num_nodes)
        # Calculate the sum of input and target along the node dimension
        super().forward(input.x, target.x, input.edge_attr, target.edge_attr)
        wl_loss = self.wl_loss_func(input, target)
        velocity_loss = self.velocity_loss_func(input, target)
        input_sum = input.sum(dim=1)
        target_sum = target.sum(dim=1)

        # Calculate the L1 loss between the sums
        mass_conservation_loss = self.scale * super().forward(input_sum, target_sum)

        return mass_conservation_loss

def get_loss_func(loss_func_name: str, **kwargs) -> Callable | Module:
    if loss_func_name == 'l1':
        return L1Loss()
    if loss_func_name == 'mse':
        return MSELoss()
    if loss_func_name == 'combined_l1':
        return CombinedL1Loss(**kwargs)
    raise ValueError('Invalid loss function name: {}'.format(loss_func_name))
