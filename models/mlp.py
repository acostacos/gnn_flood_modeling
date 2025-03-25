from torch import Tensor
from torch.nn import Identity
from torch_geometric.data import Data
from utils.model_utils import make_mlp

from .base_model import BaseModel

class MLP(BaseModel):
    '''
    MLP
    Multi-Layer Perceptron model as baseline for non-GNN appraoches
    '''
    def __init__(self,
                 num_nodes: int,
                 num_layers: int = 1,
                 activation: str = 'prelu',
                 bias: bool = True,
                 residual: bool = True,

                 **base_model_kwargs):
        super().__init__(**base_model_kwargs)
        self.num_nodes = num_nodes

        input_size = self.input_node_features * num_nodes
        output_size = self.output_node_features * num_nodes
        hidden_size = input_size * 2
        self.mlp = make_mlp(input_size=input_size, output_size=output_size,
                                            hidden_size=hidden_size, num_layers=num_layers,
                                        activation=activation, bias=bias, device=self.device)

        if residual:
            self.residual = Identity()

    def forward(self, graph: Data) -> Tensor:
        x = graph.x.clone()
        x0 = x[:, -self.output_node_features:].flatten()
        x = x.flatten()

        x = self.mlp(x)

        if hasattr(self, 'residual'):
            x = x + self.residual(x0)
        
        x = x.reshape((self.num_nodes, self.output_node_features))

        return x

