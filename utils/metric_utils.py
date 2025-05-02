import torch

from torch import Tensor
from torch.nn.functional import mse_loss, l1_loss

def RMSE(pred: Tensor, target: Tensor) -> float:
    return torch.sqrt(mse_loss(pred, target))

def MAE(pred: Tensor, target: Tensor):
    return l1_loss(pred, target)

def CSI(pred: Tensor, target: Tensor, threshold: float = 0.3):
    # TP / (TP + FN + FP)
    pass

def NSE(y_true, y_pred):
    # 1 - (sum((y_true - y_pred)^2) / sum((y_true - mean(y_true))^2))
    pass
