import torch

from torch.nn import Module
from torch.nn.functional import relu
from torch_geometric.loader import DataLoader
from typing import Callable
from utils import Logger

from .validation_stats import ValidationStats

class AutoregressionValidator:
    def __init__(self,
                 val_dataset: DataLoader,
                 model: Module,
                 device: str,
                 denormalize_func: Callable = None,
                 debug: bool = False,
                 logger: Logger = None):
        self.model = model
        self.val_dataset = val_dataset
        self.device = device
        self.data_is_normalized = denormalize_func is not None
        self.denormalize_func = denormalize_func
        self.debug = debug

        self.log = print
        if logger is not None and hasattr(logger, 'log'):
            self.log = logger.log
        self.stats = ValidationStats(logger=logger)

    def validate(self, save_stats_path: str = None, use_water_depth: bool = True):
        self.model.eval()
        with torch.no_grad():
            previous_timesteps = self.val_dataset.dataset.previous_timesteps
            sliding_window_length = 1 * (previous_timesteps+1) # Water Level at the end of node features
            wl_sliding_window = self.val_dataset.dataset[0].x.clone()[:, -sliding_window_length:]
            wl_sliding_window = wl_sliding_window.to(self.device)
            self.stats.start_validate()

            for graph in self.val_dataset:
                graph = graph.to(self.device)
                graph.x = torch.concat((graph.x[:, :-sliding_window_length], wl_sliding_window), dim=1)

                pred = self.model(graph)
                wl_sliding_window = torch.concat((wl_sliding_window[:, 1:], pred), dim=1)

                label = graph.y

                if use_water_depth:
                    # Get elevation from the graph data
                    ELEVATION_IDX = 2
                    elevation = graph.x[:, ELEVATION_IDX][:, None].cpu()
                    pred, label = self.convert_water_level_to_water_depth(pred, label, elevation)

                self.stats.update_stats_for_epoch(pred.cpu(),
                                                  label.cpu(),
                                                  water_threshold=0.05)

            self.stats.end_validate()

        self.stats.print_stats_summary()
        if save_stats_path is not None:
            self.stats.save_stats(save_stats_path)

    def convert_water_level_to_water_depth(self, pred: torch.Tensor, target: torch.Tensor, elevation: torch.Tensor):
        if self.data_is_normalized:
            pred = self.denormalize_func('water_level', pred)
            target = self.denormalize_func('water_level', target)
            elevation = self.denormalize_func('elevation', elevation)

        # Ensure water depth is non-negative
        water_depth_pred = relu(pred - elevation)
        water_depth_target = relu(target - elevation)

        return water_depth_pred, water_depth_target
