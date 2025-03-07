import torch

from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch_geometric.data import Data
from typing import Any, Callable, List

from .training_stats import TrainingStats

class BaseTrainer:
    def __init__(self,
                 train_dataset: List[Data],
                 val_dataset: List[Data],
                 model: Module,
                 loss_func: Callable | Module,
                 optimizer: Optimizer,
                 num_epochs: int,
                 device: str):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.model = model
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.device = device
        self.stats = TrainingStats()

    def train(self):
        self.stats.start_train()
        for epoch in range(self.num_epochs):
            self.model.train()
            running_loss = 0.0

            for graph in self.train_dataset:
                self.optimizer.zero_grad()

                graph = graph.to(self.device)
                output = self.get_prediction(graph)

                loss = self.compute_loss(self.loss_func, graph, output)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            epoch_loss = running_loss / len(self.train_dataset)
            self.stats.add_train_loss(epoch_loss)
            print(f'Epoch [{epoch + 1}/{self.num_epochs}], Training Loss: {epoch_loss:.4f}')

        self.stats.end_train()

    def validate(self):
        self.stats.start_val()
        self.model.eval()
        running_loss = 0.0

        with torch.no_grad():
            for graph in self.val_dataset:
                graph = graph.to(self.device)
                output = self.get_prediction(graph)

                loss = self.compute_loss(self.loss_func, graph, output)

                running_loss += loss.item()

        avg_loss = running_loss / len(self.val_dataset)
        self.stats.set_val_loss(avg_loss)

        self.stats.end_val()

    def get_prediction(self, graph: Data) -> Any:
        pred = self.model(graph)
        return pred

    def compute_loss(self, loss_func: Callable | Module, graph: Data, output: Any) -> Tensor:
        label = graph.y
        return loss_func(output, label)

    def get_stats(self):
        return self.stats
