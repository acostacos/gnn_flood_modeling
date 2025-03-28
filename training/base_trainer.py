import torch

from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from typing import Any, Callable, List
from utils import Logger, file_utils

from .training_stats import TrainingStats

class BaseTrainer:
    def __init__(self,
                 train_datasets: List[str],
                 val_dataset: str,
                 model: Module,
                 loss_func: Callable | Module,
                 optimizer: Optimizer,
                 num_epochs: int,
                 batch_size: int,
                 device: str,
                 logger: Logger = None):
        self.train_datasets = train_datasets
        self.val_dataset = val_dataset
        self.model = model
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.device = device

        self.log = print
        if logger is not None and hasattr(logger, 'log'):
            self.log = logger.log
        self.stats = TrainingStats(logger=logger)

    def train(self):
        self.stats.start_train()
        for epoch in range(self.num_epochs):
            self.model.train()
            running_loss = 0.0

            len_training_samples = 0
            for dataset in self.train_datasets:
                len_training_samples += len(dataset)
                dataloader = DataLoader(dataset, batch_size=self.batch_size)

                for batch in dataloader:
                    self.optimizer.zero_grad()

                    batch = batch.to(self.device)
                    output = self.get_prediction(batch)

                    loss = self.compute_loss(self.loss_func, batch, output)
                    loss.backward()
                    self.optimizer.step()

                    running_loss += loss.item()

            epoch_loss = running_loss / len_training_samples
            self.stats.add_train_loss(epoch_loss)
            self.log(f'Epoch [{epoch + 1}/{self.num_epochs}], Training Loss: {epoch_loss:.4f}')

        self.stats.end_train()

    def validate(self):
        self.stats.start_val()
        self.model.eval()
        running_loss = 0.0

        len_dataset = len(self.val_dataset)
        dataloader = DataLoader(self.val_dataset, batch_size=self.batch_size)

        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(self.device)
                output = self.get_prediction(batch)

                loss = self.compute_loss(self.loss_func, batch, output)

                running_loss += loss.item()

        avg_loss = running_loss / len_dataset
        self.stats.set_val_loss(avg_loss)

        self.stats.end_val(len_dataset)

    def get_prediction(self, batch: Data) -> Any:
        pred = self.model(batch)
        return pred

    def compute_loss(self, loss_func: Callable | Module, batch: Data, output: Any) -> Tensor:
        label = batch.y
        return loss_func(output, label)

    def get_stats(self):
        return self.stats
