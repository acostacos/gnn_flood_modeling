import torch

from torch.nn import Module
from torch_geometric.data import Data

from .training_stats import TrainingStats

class Trainer:
    def __init__(self,
                 model: Module,
                 dataset: list[Data],
                 loss_func: str,
                 optimizer: str,
                 num_epochs: int,
                 percentage_train: float,
                 device: str):

        self.device = device
        self.stats = TrainingStats()

    def train(self, model: Module, loss_func, optimizer):
        self.stats.start_train()
        for epoch in range(self.num_epochs):
            model.train()
            running_loss = 0.0

            for graph in train_dataset:
                graph = graph.to(self.device)
                labels = graph.y

                optimizer.zero_grad()

                outputs = model(graph)
                loss = loss_func(outputs, labels)

                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            epoch_loss = running_loss / num_train
            self.stats.add_train_loss(epoch_loss)
            print(f'Epoch [{epoch + 1}/{num_epochs}], Training Loss: {epoch_loss:.4f}')
        self.stats.end_train()
        print(f'Total training time: {end_time - start_time} seconds')
    
    def get_stats(self):
        return self.stats
