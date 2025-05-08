import numpy as np
import os
import time

from torch import Tensor
from utils import Logger
from utils.metric_utils import RMSE, MAE, NSE, CSI

class ValidationStats:
    def __init__(self, logger: Logger = None):
        self.val_start_time = None
        self.val_end_time = None
        self.pred_list = []
        self.target_list = []

        # Overall stats
        self.rmse_list = []
        self.mae_list = []
        self.nse_list = []
        self.csi_list = []

        # Flooded cell stats
        self.rmse_flooded_list = []
        self.mae_flooded_list = []
        self.nse_flooded_list = []
        self.csi_flooded_list = []

        self.log = print
        if logger is not None and hasattr(logger, 'log'):
            self.log = logger.log
    
    def start_validate(self):
        self.val_start_time = time.time()

    def end_validate(self):
        self.val_end_time = time.time()
    
    def get_inference_time(self):
        return (self.val_end_time - self.val_start_time) / len(self.pred_list)

    def update_stats_for_epoch(self,
                               pred: Tensor,
                               target: Tensor,
                               water_threshold: float = 0.3):
        self.pred_list.append(pred)
        self.target_list.append(target)

        self.rmse_list.append(RMSE(pred, target))
        self.mae_list.append(MAE(pred, target))
        self.nse_list.append(NSE(pred, target))

        binary_pred = self.convert_water_depth_to_binary(pred, water_threshold=water_threshold)
        binary_target = self.convert_water_depth_to_binary(target, water_threshold=water_threshold)

        self.csi_list.append(CSI(binary_pred, binary_target))

        # Compute metrics for flooded areas only
        flooded_mask = binary_pred | binary_target
        flooded_pred, flooded_target = self.filter_by_water_threshold(pred, target, flooded_mask)

        self.rmse_flooded_list.append(RMSE(flooded_pred, flooded_target))
        self.mae_flooded_list.append(MAE(flooded_pred, flooded_target))
        self.nse_flooded_list.append(NSE(flooded_pred, flooded_target))

        binary_flooded_pred = binary_pred[flooded_mask]
        binary_flooded_target = binary_target[flooded_mask]
        self.csi_flooded_list.append(CSI(binary_flooded_pred, binary_flooded_target))

    def convert_water_level_to_water_depth(self, pred: Tensor, target: Tensor, elevation: Tensor):
        if self.data_is_normalized:
            pred = self.denormalize_func('water_level', pred)
            target = self.denormalize_func('water_level', target)
            elevation = self.denormalize_func('elevation', elevation)

        # water_depth_pred = relu(pred - elevation) # Ensure water depth is non-negative
        water_depth_pred = pred - elevation
        water_depth_target = target - elevation

        return water_depth_pred, water_depth_target
    
    def convert_water_depth_to_binary(self, water_level: Tensor, water_threshold: float) -> Tensor:
        return (water_level > water_threshold)

    def filter_by_water_threshold(self, pred: Tensor, target: Tensor, flooded_mask: Tensor):
        flooded_pred = pred[flooded_mask]
        flooded_target = target[flooded_mask]
        return flooded_pred, flooded_target

    def print_stats_summary(self):
        if len(self.rmse_list) > 0:
            rmse_np = np.array(self.rmse_list)
            self.log(f'Average RMSE: {rmse_np.mean():.4f}')
        if len(self.rmse_flooded_list) > 0:
            rmse_flooded_np = np.array(self.rmse_flooded_list)
            self.log(f'Average RMSE (flooded): {rmse_flooded_np.mean():.4f}')
        if len(self.mae_list) > 0:
            mae_np = np.array(self.mae_list)
            self.log(f'Average MAE: {mae_np.mean():.4f}')
        if len(self.mae_flooded_list) > 0:
            mae_flooded_np = np.array(self.mae_flooded_list)
            self.log(f'Average MAE (flooded): {mae_flooded_np.mean():.4f}')
        if len(self.nse_list) > 0:
            nse_np = np.array(self.nse_list)
            self.log(f'Average NSE: {nse_np.mean():.4f}')
        if len(self.nse_flooded_list) > 0:
            nse_flooded_np = np.array(self.nse_flooded_list)
            self.log(f'Average NSE (flooded): {nse_flooded_np.mean():.4f}')
        if len(self.csi_list) > 0:
            csi_np = np.array(self.csi_list)
            self.log(f'Average CSI: {csi_np.mean():.4f}')
        if len(self.csi_flooded_list) > 0:
            csi_flooded_np = np.array(self.csi_flooded_list)
            self.log(f'Average CSI (flooded): {csi_flooded_np.mean():.4f}')

        if self.val_start_time is not None and self.val_end_time is not None:
            self.log(f'Inference time for one timestep: {self.get_inference_time():.4f} seconds')

    def save_stats(self, filepath: str):
        dirname = os.path.dirname(filepath)
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        stats = {
            'pred': np.array(self.pred_list),
            'target': np.array(self.target_list),
            'rmse': np.array(self.rmse_list),
            'mae': np.array(self.mae_list),
            'nse': np.array(self.nse_list),
            'csi': np.array(self.csi_list),
            'rmse_flooded': np.array(self.rmse_flooded_list),
            'mae_flooded': np.array(self.mae_flooded_list),
            'nse_flooded': np.array(self.nse_flooded_list),
            'csi_flooded': np.array(self.csi_flooded_list)
        }
        np.savez(filepath, **stats)

        self.log(f'Saved metrics to: {filepath}')
