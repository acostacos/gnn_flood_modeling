from enum import StrEnum

class GNNConvolution(StrEnum):
    GCN = 'gcn'
    GAT = 'gat'
    SAGE = 'sage'
    GIN = 'gin'

class Activation(StrEnum):
    RELU = 'relu'
    PRELU = 'prelu'
