from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from typing import Literal

class ToUndirectedFlipFeatures(BaseTransform):
    def __init__(self, feature_indices: list[int], positive_dir: Literal['from', 'to'] = 'from'):
        self.feature_indices = feature_indices
        self.positive_dir = positive_dir

    def forward(self, data: Data) -> Data:
        # Get all edge index which has a corresponding flip
        # Only get one of those edges - depending on self.positive_dir
        # Get index of edge_attrs for those edges
        # Multiply by -1

        return data
