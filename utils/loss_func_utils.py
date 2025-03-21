from constants import LossFunction
from torch.nn import MSELoss, L1Loss
from typing import Callable

def node_edge_loss_func(node_pred, node_label, edge_pred, edge_label):
    loss_func = L1Loss()
    node_loss = loss_func(node_pred, node_label)
    edge_loss = loss_func(edge_pred, edge_label)
    return node_loss + edge_loss

def get_loss_func(loss_func_name: LossFunction) -> Callable:
    if loss_func_name == LossFunction.L1:
        return L1Loss()
    if loss_func_name == LossFunction.MSE:
        return MSELoss()
    if loss_func_name == LossFunction.COMBINED_L1:
        return node_edge_loss_func
    raise ValueError('Invalid loss function name: {}'.format(loss_func_name))
