import torch

from torch import Tensor
from torch.nn import Module
from torch_geometric.data import Data
from typing import Callable, Tuple

from .base_trainer import BaseTrainer

class DualRegressionTrainer(BaseTrainer):
    def __init__(self, mode: str = 'dual', edge_loss_func: Callable = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mode = mode
        self.edge_loss_func = edge_loss_func
    
    def train(self):
        if self.mode == 'node':
            super().train()
            return

        self._train_dual()

    def _train_dual(self):
        self.stats.start_train()
        for epoch in range(self.num_epochs):
            self.model.train()
            running_loss = 0.0
            running_node_loss = 0.0
            running_edge_loss = 0.0

            len_training_samples = 0
            for dataset in self.train_datasets:
                len_training_samples += len(dataset)
                for batch in dataset:
                    self.optimizer.zero_grad()

                    batch = batch.to(self.device)
                    output = self.get_prediction(batch)

                    node_loss, edge_loss = self.compute_loss(self.loss_func, batch, output)
                    total_loss = node_loss + edge_loss
                    total_loss.backward()
                    self.optimizer.step()

                    running_loss += total_loss.item()
                    running_node_loss += node_loss.item()
                    running_edge_loss += edge_loss.item()

            epoch_loss = running_loss / len_training_samples
            epoch_node_loss = running_node_loss / len_training_samples
            epoch_edge_loss = running_edge_loss / len_training_samples

            self.stats.add_train_loss(epoch_loss)
            self.log(f'Epoch [{epoch + 1}/{self.num_epochs}]:')
            self.log(f'\tNode Loss: {epoch_node_loss:.4f}')
            self.log(f'\tEdge Loss: {epoch_edge_loss:.4f}')
            self.log(f'\tTotal Loss: {epoch_loss:.4f}')

            if self.debug and epoch > 0 and epoch % 5 == 0:
                self.print_memory_usage(epoch)

        self.stats.end_train()

    def validate(self):
        if self.mode == 'node':
            super().validate()
            return

        self._validate_dual()

    def _validate_dual(self):
        self.stats.start_val()
        self.model.eval()
        running_loss = 0.0
        running_node_loss = 0.0
        running_edge_loss = 0.0

        len_dataset = len(self.val_dataset)
        with torch.no_grad():
            for batch in self.val_dataset:
                batch = batch.to(self.device)
                output = self.get_prediction(batch)

                node_loss, edge_loss = self.compute_loss(self.loss_func, batch, output)
                total_loss = node_loss + edge_loss

                running_loss += total_loss.item()
                running_node_loss += node_loss.item()
                running_edge_loss += edge_loss.item()

        avg_loss = running_loss / len_dataset
        avg_node_loss = running_node_loss / len_dataset
        avg_edge_loss = running_edge_loss / len_dataset
        self.log(f'Validation Node Loss: {avg_node_loss:.4f}')
        self.log(f'Validation Edge Loss: {avg_edge_loss:.4f}')
        self.stats.set_val_loss(avg_loss)

        self.stats.end_val(len_dataset)

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
    
    def _compute_loss_dual(self, loss_func: Callable | Module, graph: Data, output: Tuple) -> Tuple[Tensor, Tensor]:
        node_pred, edge_pred = output

        node_label = graph.y
        node_loss = loss_func(node_pred, node_label)

        edge_label = graph.y_edge
        edge_loss = self.edge_loss_func(edge_pred, edge_label)

        return node_loss, edge_loss
