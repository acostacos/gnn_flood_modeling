from torch import Tensor
from torch.nn import Module
from torch_geometric.data import Data
from typing import Callable, Tuple

from .base_trainer import BaseTrainer

class DualRegressionTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_prediction(self, graph: Data) -> Tuple:
        node_pred, edge_pred = self.model(graph)
        return (node_pred, edge_pred)

    def compute_loss(self, loss_func: Callable | Module, graph: Data, output: Tuple) -> Tensor:
        node_pred, edge_pred = output
        node_label = graph.y
        edge_label = graph.y_edge
        return loss_func(node_pred, node_label, edge_pred, edge_label)
