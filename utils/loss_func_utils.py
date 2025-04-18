import torch

from torch.nn import Module, MSELoss, L1Loss
from torch_geometric.data import Data
from torch_geometric.utils import scatter
from typing import Callable, Tuple

class CombinedL1Loss(L1Loss):
    def __init__(self, node_weight: float = 0.5, edge_weight: float = 0.5):
        super().__init__()
        self.node_weight = node_weight
        self.edge_weight = edge_weight

        self.node_loss = 0
        self.edge_loss = 0

    def forward(self, node_pred: torch.Tensor, edge_pred: torch.Tensor, orig_graph: Data):
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

    def forward(self, node_pred: torch.Tensor, edge_pred: torch.Tensor, orig_graph: Data):
        combined_l1_loss = super().forward(node_pred, edge_pred, orig_graph)

        num_nodes = node_pred.shape[0]
        edge_index = orig_graph.edge_index

        area = orig_graph.x[:, 0]
        water_level = node_pred.squeeze()
        volume = area * water_level

        face_length = orig_graph.edge_attr[:, 0]
        fl_times_v = face_length * edge_pred.squeeze()

        out_total_fl_times_v= scatter(fl_times_v, edge_index[0], reduce='sum', dim_size=num_nodes)
        out_total_flux = out_total_fl_times_v * water_level

        in_total_fl_times_v = scatter(fl_times_v, edge_index[1], reduce='sum', dim_size=num_nodes)
        in_total_flux = in_total_fl_times_v * water_level

        mc_loss = (in_total_flux + volume - out_total_flux).sum() # Want this to be zero

        return combined_l1_loss + mc_loss

def get_loss_func(loss_func_name: str, **kwargs) -> Callable | Module:
    if loss_func_name == 'l1':
        return L1Loss()
    if loss_func_name == 'mse':
        return MSELoss()
    if loss_func_name == 'combined_l1':
        return CombinedL1Loss(**kwargs)
    if loss_func_name == 'mass_conservation_l1':
        return MassConservationL1Loss(**kwargs)
    raise ValueError('Invalid loss function name: {}'.format(loss_func_name))
