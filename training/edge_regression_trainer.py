from torch import Tensor
from torch.nn import Module
from torch_geometric.data import Data
from typing import Any, Callable

from .base_trainer import BaseTrainer

class EdgeRegressionTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self, loss_func: Callable | Module, graph: Data, output: Any) -> Tensor:
        label = graph.y_edge
        return loss_func(output, label)
