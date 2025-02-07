from torch.nn import Module

class BaseModel(Module):
    def __init__(self,
                 static_node_features: int,
                 dynamic_node_features: int,
                 static_edge_features: int,
                 dynamic_edge_features: int,
                 device: str = 'cpu'):
        super().__init__()
        self.device = device
        self.static_node_features = static_node_features
        self.dynamic_node_features = dynamic_node_features
        self.static_edge_features = static_edge_features
        self.dynamic_edge_features = dynamic_edge_features

    def _mask_small_WD(self, x, epsilon=0.001):        
        x[:,0][x[:,0].abs() < epsilon] = 0

        # Mask velocities where there is no water
        x[:,1:][x[:,0] == 0] = 0

        return x
