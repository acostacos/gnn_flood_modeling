import torch

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
        node_pred, _ = self.model(graph)
        return node_pred

    def train(self):
        if self.mode == 'node':
            super().train()
            return

        self._train_dual()
    
    def validate(self):
        if self.mode == 'node':
            super().validate()
            return

        self._validate_dual()

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
                    output = self._get_prediction_dual(batch)

                    loss = self._compute_loss_dual(self.loss_func, batch, output)
                    loss.backward()
                    self.optimizer.step()
                    running_loss += loss.item()

                    node_loss, edge_loss = self.loss_func.get_loss_components()
                    running_node_loss += node_loss
                    running_edge_loss += edge_loss

            epoch_loss = running_loss / len_training_samples
            epoch_node_loss = running_node_loss / len_training_samples
            epoch_edge_loss = running_edge_loss / len_training_samples

            self.stats.add_train_loss(epoch_loss)
            self.log(f'Epoch [{epoch + 1}/{self.num_epochs}]:')
            self.log(f'\tNode Loss: {epoch_node_loss:.4f}')
            self.log(f'\tEdge Loss: {epoch_edge_loss:.4f}')
            self.log(f'\tTotal Loss: {epoch_loss:.4f}')

            if self.debug:
                self.print_memory_usage(epoch)

        additional_info = f'Final Node Loss: {epoch_node_loss:.4f}, Final Edge Loss: {epoch_edge_loss:.4f}'
        self.stats.set_additional_train_info(additional_info)
        self.stats.end_train()

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
                output = self._get_prediction_dual(batch)

                loss = self._compute_loss_dual(self.loss_func, batch, output)

                running_loss += loss.item()

                node_loss, edge_loss = self.loss_func.get_loss_components()
                running_node_loss += node_loss
                running_edge_loss += edge_loss

        avg_loss = running_loss / len_dataset
        avg_node_loss = running_node_loss / len_dataset
        avg_edge_loss = running_edge_loss / len_dataset
        self.stats.set_val_loss(avg_loss)
        additional_info = f'Validation Node Loss: {avg_node_loss:.4f}, Validation Edge Loss: {avg_edge_loss:.4f}'
        self.stats.set_additional_val_info(additional_info)

        self.stats.end_val(len_dataset)

    def _get_prediction_dual(self, graph: Data) -> Tuple:
        node_pred, edge_pred = self.model(graph)
        return (node_pred, edge_pred)

    def _compute_loss_dual(self, loss_func: Callable | Module, orig_graph: Data, output: Tuple) -> Tensor:
        node_pred, edge_pred = output

        return loss_func(node_pred, edge_pred, orig_graph)
