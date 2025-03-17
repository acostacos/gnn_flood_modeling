from torch import Tensor
from torch.nn import Module
from torch_geometric.data import Data
from typing import Callable, Tuple

from .base_trainer import BaseTrainer

class DualRegressionTrainer(BaseTrainer):
    def __init__(self, mode: str = 'dual', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mode = mode

    def get_prediction(self, graph: Data) -> Tuple:
        if self.mode == 'node':
            return self._get_prediction_node(graph)
        return self._get_prediction_dual(graph)
    
    def _get_prediction_node(self, graph: Data) -> Tuple:
        node_pred, _ = self.model(graph)
        return node_pred
    
    def _get_prediction_dual(self, graph: Data) -> Tuple:
        node_pred, edge_pred = self.model(graph)
        return (node_pred, edge_pred)

    def compute_loss(self, loss_func: Callable | Module, graph: Data, output: Tuple) -> Tensor:
        if self.mode == 'node':
            return self._compute_loss_node(loss_func, graph, output)
        return self._compute_loss_dual(loss_func, graph, output)

    def _compute_loss_node(self, loss_func: Callable | Module, graph: Data, node_pred: Tensor) -> Tensor:
        return super().compute_loss(loss_func, graph, node_pred)
    
    def _compute_loss_dual(self, loss_func: Callable | Module, graph: Data, output: Tuple) -> Tensor:
        node_pred, edge_pred = output
        node_label = graph.y
        edge_label = graph.y_edge
        return loss_func(node_pred, node_label, edge_pred, edge_label)
