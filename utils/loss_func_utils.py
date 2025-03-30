from torch.nn import MSELoss, L1Loss
from typing import Callable

def node_edge_loss_func(node_pred, node_label, edge_pred, edge_label):
    loss_func = L1Loss()
    node_loss = loss_func(node_pred, node_label)
    edge_loss = loss_func(edge_pred, edge_label)
    return node_loss + edge_loss

class ScaledL1Loss(L1Loss):
    def __init__(self, scale: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.scale = scale

    def forward(self, input, target):
        return self.scale * super().forward(input, target)

def get_loss_func(loss_func_name: str, **kwargs) -> Callable:
    if loss_func_name == 'l1':
        return L1Loss()
    if loss_func_name == 'mse':
        return MSELoss()
    if loss_func_name == 'combined_l1':
        return node_edge_loss_func
    if loss_func_name == 'scaled_l1':
        return ScaledL1Loss(**kwargs)
    raise ValueError('Invalid loss function name: {}'.format(loss_func_name))
