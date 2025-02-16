from torch.nn import Module

class BaseModel(Module):
    def __init__(self,
                 static_node_features: int,
                 dynamic_node_features: int,
                 static_edge_features: int,
                 dynamic_edge_features: int,
                 previous_timesteps: int,
                 device: str = 'cpu'):
        super().__init__()
        self.device = device
        self.previous_timesteps = previous_timesteps

        self.static_node_features = static_node_features
        self.dynamic_node_features = dynamic_node_features
        self.input_node_features = self.static_node_features + (self.dynamic_node_features * (self.previous_timesteps+1))
        self.output_node_features = self.dynamic_node_features

        self.static_edge_features = static_edge_features
        self.dynamic_edge_features = dynamic_edge_features
        self.input_edge_features = self.static_edge_features + (self.dynamic_edge_features * (self.previous_timesteps+1))
        self.output_edge_features = self.dynamic_edge_features
