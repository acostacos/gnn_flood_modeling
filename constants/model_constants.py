from enum import StrEnum

class GNNConvolution(StrEnum):
    GCN = 'gcn'
    GAT = 'gat'

class Activation(StrEnum):
    PRELU = 'prelu'
