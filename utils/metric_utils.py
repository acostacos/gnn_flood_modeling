import torch

from torch import Tensor
from torch.nn.functional import mse_loss, l1_loss

def RMSE(pred: Tensor, target: Tensor) -> Tensor:
    return torch.sqrt(mse_loss(pred, target))

def MAE(pred: Tensor, target: Tensor) -> Tensor:
    return l1_loss(pred, target)

def NSE(pred: Tensor, target: Tensor) -> Tensor:
    '''Nash Sutcliffe Efficiency'''
    model_sse = torch.sum((target - pred)**2)
    mean_model_sse = torch.sum((target - target.mean())**2)
    return 1 - (model_sse / mean_model_sse)

def CSI(pred: Tensor, target: Tensor, threshold: float = 0.3):
    binary_pred = convert_water_depth_to_binary(pred, water_threshold=threshold)
    binary_target = convert_water_depth_to_binary(target, water_threshold=threshold)
    TP, _, FP, FN = get_confusion_matrix(binary_pred, binary_target)
    return TP / (TP + FN + FP)

def convert_water_depth_to_binary(water_level: Tensor, water_threshold: float = 0.3) -> Tensor:
    return (water_level > water_threshold).int()

def get_confusion_matrix(pred: Tensor, target: Tensor):
    dim = 0
    TP = (pred & target).sum(dim=dim) #true positive
    TN = (~pred & ~target).sum(dim=dim) #true negative
    FP = (pred & ~target).sum(dim=dim) #false positive
    FN = (~pred & target).sum(dim=dim) #false negative

    return TP, TN, FP, FN
