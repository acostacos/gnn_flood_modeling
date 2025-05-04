import torch

from torch import Tensor
from torch.nn.functional import mse_loss, l1_loss

def RMSE(pred: Tensor, target: Tensor) -> Tensor:
    return torch.sqrt(mse_loss(pred, target))

def MAE(pred: Tensor, target: Tensor) -> Tensor:
    return l1_loss(pred, target)

def CSI(pred: Tensor, target: Tensor, threshold: float = 0.3):
    # TP / (TP + FN + FP)
    pass

def NSE(y_true, y_pred):
    # 1 - (sum((y_true - y_pred)^2) / sum((y_true - mean(y_true))^2))
    pass




# ================= TODO: FIX CSI ====================
def get_binary_rollouts(predicted_rollout, real_rollout, water_threshold=0):
    '''Converts flood simulation into a binary map (1=flood, 0=no flood) for classification purposes
    ------
    water_threshold: float
        Threshold for the binary map creation, i.e., 'flood' if WD>threshold
    '''
    if predicted_rollout.dim() == 4:
        predicted_rollout_flood = predicted_rollout[:,:,0,:]>water_threshold
        real_roll_flood = real_rollout[:,:,0,:]>water_threshold
    elif predicted_rollout.dim() == 3:
        predicted_rollout_flood = predicted_rollout[:,0,:]>water_threshold
        real_roll_flood = real_rollout[:,0,:]>water_threshold

    return predicted_rollout_flood, real_roll_flood

def get_rollout_confusion_matrix(predicted_rollout, real_rollout, water_threshold=0):
    predicted_rollout_flood, real_roll_flood = get_binary_rollouts(predicted_rollout, real_rollout, water_threshold=water_threshold)

    if predicted_rollout.dim() == 4:
        nodes_dim = 1
    elif predicted_rollout.dim() == 3:
        nodes_dim = 0

    TP = (predicted_rollout_flood & real_roll_flood).sum(nodes_dim) #true positive
    TN = (~predicted_rollout_flood & ~real_roll_flood).sum(nodes_dim) #true negative
    FP = (predicted_rollout_flood & ~real_roll_flood).sum(nodes_dim) #false positive
    FN = (~predicted_rollout_flood & real_roll_flood).sum(nodes_dim) #false negative

    return TP, TN, FP, FN

def get_CSI(predicted_rollout, real_rollout, water_threshold=0):
    '''Returns the Critical Success Index (CSI) in time for a given water_threshold'''
    TP, TN, FP, FN = get_rollout_confusion_matrix(predicted_rollout, real_rollout, water_threshold=water_threshold)

    CSI = TP / (TP + FN + FP)

    return CSI
