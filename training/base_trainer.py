import os
import torch
import psutil

from torch.nn import Module
from torch.optim import Optimizer
from torch_geometric.loader import DataLoader
from typing import Callable, List
from utils import Logger, convert_utils

from .training_stats import TrainingStats

class BaseTrainer:
    def __init__(self,
                 train_datasets: List[DataLoader],
                 model: Module,
                 loss_func: Callable | Module,
                 optimizer: Optimizer,
                 num_epochs: int,
                 device: str,
                 debug: bool = False,
                 logger: Logger = None):
        self.train_datasets = train_datasets
        self.model = model
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.device = device
        self.debug = debug

        self.log = print
        if logger is not None and hasattr(logger, 'log'):
            self.log = logger.log
        self.stats = TrainingStats(logger)

    def train(self):
        raise NotImplementedError("Train method not implemented.")

    def print_stats_summary(self):
        return self.stats.print_stats_summary()

    def plot_train_loss(self):
        self.stats.plot_train_loss()

    def save_training_stats(self, filepath: str):
        self.stats.save_stats(filepath)

    def print_memory_usage(self, epoch: int):
        self.log(f'Usage Statistics (epoch {epoch+1}): ')
        process = psutil.Process(os.getpid())

        ram_used = process.memory_info().rss
        self.log(f"\tRAM Usage: {convert_utils.bytes_to_gb(ram_used)} GB")

        num_cores = psutil.cpu_count()
        self.log(f"\tNum CPU Cores:  {num_cores} cores")

        gpu_usage = torch.cuda.mem_get_info()
        free = convert_utils.bytes_to_gb(gpu_usage[0])
        total = convert_utils.bytes_to_gb(gpu_usage[1])
        self.log(f"\tGPU Usage: {free}GB / {total}GB")

        gpu_allocated = torch.cuda.memory_allocated()
        gpu_cached = torch.cuda.memory_reserved()
        self.log(f"\tCUDA GPU Allocated: {convert_utils.bytes_to_gb(gpu_allocated)}GB")
        self.log(f"\tCUDA GPU Cached: {convert_utils.bytes_to_gb(gpu_cached)}GB")
