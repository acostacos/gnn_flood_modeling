# Taken from https://github.com/snap-stanford/pretrain-gnns/blob/master/bio/model.py

import torch
import torch.nn.functional as F

from torch_geometric.nn import GINEConv

from utils.model_utils import make_mlp
from models.base_model import BaseModel

class GINE(BaseModel):
    """
    GINE
    Self-supervised learning with node and graph self-supervised tasks.
    """
    def __init__(self,
                 input_features: int = None,
                 output_features: int = None,
                 hidden_features: int = None,
                 num_layers: int = 2,
                 activation: str = 'prelu',

                 # GINEConv Parameters
                 mlp_layers: int = 2,
                 edge_features: int = None,
                 eps: float = 0.0,
                 train_eps: bool = False,
                 JK: str = "last",
                 drop_ratio: float = 0,

                 **base_model_kwargs):
        super().__init__(**base_model_kwargs)
        self.JK = JK
        self.num_layer = num_layers
        self.drop_ratio = drop_ratio

        if input_features is None:
            input_features = self.input_node_features
        if output_features is None:
            output_features = self.output_node_features
        if edge_features is None:
            edge_features = self.input_edge_features

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")
        
        self.x_embedding1 = torch.nn.Embedding(num_atom_type, emb_dim)
        self.x_embedding2 = torch.nn.Embedding(num_chirality_tag, emb_dim)

        ###List of message-passing GNN convs
        self.gnns = torch.nn.ModuleList()
        for layer in range(mlp_layers):
            self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, emb_dim))
            self.gnns.append(GINEConv(nn, eps=eps, train_eps=train_eps, edge_dim=edge_features))
        
        ###List of batchnorms
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(self.num_layer):
            self.batch_norms.append(torch.nn.BatchNorm1d(num_features=output_features))

    #def forward(self, x, edge_index, edge_attr):
    def forward(self, x, edge_index, edge_attr):
        # TODO: See if we need to add encoder
        # x = self.x_embedding1(x[:,0]) + self.x_embedding2(x[:,1])

        h_list = [x]
        for layer in range(self.num_layer):
            print(h_list[layer].shape)
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            print(h.shape)
            h = self.batch_norms[layer](h)
            #h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            if layer == self.num_layer - 1:
                #remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "concat":
            node_representation = torch.cat(h_list, dim = 1)
        elif self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "max":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.max(torch.cat(h_list, dim = 0), dim = 0)[0]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim = 0), dim = 0)[0]

        return node_representation