from enum import StrEnum

class GNNConvolution(StrEnum):
    GCN = 'gcn'
    GAT = 'gat'
    SAGE = 'sage'
    GIN = 'gin'

class Activation(StrEnum):
    RELU = 'relu'
    PRELU = 'prelu'

class LossFunction(StrEnum):
    L1 = 'l1'
    MSE = 'mse'
    COMBINED_L1 = 'combined_l1'
