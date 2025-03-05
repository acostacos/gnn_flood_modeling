import torch
import numpy as np
import networkx as nx
import random

from torch_geometric.data import Data


def graph_data_obj_to_nx_simple(data):
    """
    Converts graph Data object required by the pytorch geometric package to
    network x data object. NB: Uses simplified atom and bond features,
    and represent as indices. NB: possible issues with recapitulating relative
    stereochemistry since the edges in the nx object are unordered.
    :param data: pytorch geometric Data object
    :return: network x object
    """
    G = nx.Graph()

    # node features
    atom_features = data.x.cpu().numpy()
    num_atoms = atom_features.shape[0]
    for i in range(num_atoms):
        area, roughness, elevation, water_level_t, water_level_t1, water_level_t2 = atom_features[i]
        G.add_node(i, area=area, roughness=roughness, elevation=elevation, water_level_t=water_level_t,
                   water_level_t1=water_level_t1, water_level_t2=water_level_t2)
        pass

    # edge features
    edge_index = data.edge_index.cpu().numpy()
    edge_attr = data.edge_attr.cpu().numpy()
    num_bonds = edge_index.shape[1]
    for j in range(0, num_bonds, 2):
        begin_idx = int(edge_index[0, j])
        end_idx = int(edge_index[1, j])
        direction_x, direction_y, face_length, length, slope, velocity_t, velocity_t1, velocity_t2 = edge_attr[j]
        if not G.has_edge(begin_idx, end_idx):
            G.add_edge(begin_idx, end_idx, direction_x=direction_x, direction_y=direction_y, face_length=face_length,
                       length=length, slope=slope, velocity_t=velocity_t, velocity_t1=velocity_t1, velocity_t2=velocity_t2)

    return G

def nx_to_graph_data_obj_simple(G):
    """
    Converts nx graph to pytorch geometric Data object. Assume node indices
    are numbered from 0 to num_nodes - 1. NB: Uses simplified atom and bond
    features, and represent as indices. NB: possible issues with
    recapitulating relative stereochemistry since the edges in the nx
    object are unordered.
    :param G: nx graph obj
    :return: pytorch geometric Data object
    """
    # node features
    node_features_list = []
    for _, node in G.nodes(data=True):
        atom_feature = [node['area'], node['roughness'], node['elevation'], node['water_level_t'], node['water_level_t1'], node['water_level_t2']]
        node_features_list.append(atom_feature)
    x = torch.tensor(np.array(node_features_list), dtype=torch.float32)

    # edge features
    num_bond_features = 8  # direction_x, direction_y, face_length, length, slope, velocity_t, velocity_t1, velocity_t2
    if len(G.edges()) > 0:  # mol has bonds
        edges_list = []
        edge_features_list = []
        for i, j, edge in G.edges(data=True):
            edge_feature = [edge['direction_x'], edge['direction_y'], edge['face_length'], edge['length'], edge['slope'], edge['velocity_t'], edge['velocity_t1'], edge['velocity_t2']]
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.int64)

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = torch.tensor(np.array(edge_features_list),
                                 dtype=torch.float32)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.int64)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.float32)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    return data

def reset_idxes(G):
    """
    Resets node indices such that they are numbered from 0 to num_nodes - 1
    :param G:
    :return: copy of G with relabelled node indices, mapping
    """
    mapping = {}
    for new_idx, old_idx in enumerate(G.nodes()):
        mapping[old_idx] = new_idx
    new_G = nx.relabel_nodes(G, mapping, copy=True)
    return new_G, mapping

class ExtractSubstructureContextPair:
    def __init__(self, k, l1, l2):
        """
        Randomly selects a node from the data object, and adds attributes
        that contain the substructure that corresponds to k hop neighbours
        rooted at the node, and the context substructures that corresponds to
        the subgraph that is between l1 and l2 hops away from the
        root node.
        :param k:
        :param l1:
        :param l2:
        """
        self.k = k
        self.l1 = l1
        self.l2 = l2

        # for the special case of 0, addresses the quirk with
        # single_source_shortest_path_length
        if self.k == 0:
            self.k = -1
        if self.l1 == 0:
            self.l1 = -1
        if self.l2 == 0:
            self.l2 = -1

    def __call__(self, data, root_idx=None):
        """

        :param data: pytorch geometric data object
        :param root_idx: If None, then randomly samples an atom idx.
        Otherwise sets atom idx of root (for debugging only)
        :return: None. Creates new attributes in original data object:
        data.center_substruct_idx
        data.x_substruct
        data.edge_attr_substruct
        data.edge_index_substruct
        data.x_context
        data.edge_attr_context
        data.edge_index_context
        data.overlap_context_substruct_idx
        """
        num_atoms = data.x.size()[0]
        if root_idx == None:
            root_idx = random.sample(range(num_atoms), 1)[0]

        G = graph_data_obj_to_nx_simple(data)  # same ordering as input data obj

        # Get k-hop subgraph rooted at specified atom idx
        substruct_node_idxes = nx.single_source_shortest_path_length(G,
                                                                     root_idx,
                                                                     self.k).keys()
        if len(substruct_node_idxes) > 0:
            substruct_G = G.subgraph(substruct_node_idxes)
            substruct_G, substruct_node_map = reset_idxes(substruct_G)  # need
            # to reset node idx to 0 -> num_nodes - 1, otherwise data obj does not
            # make sense, since the node indices in data obj must start at 0
            substruct_data = nx_to_graph_data_obj_simple(substruct_G)
            data.x_substruct = substruct_data.x
            data.edge_attr_substruct = substruct_data.edge_attr
            data.edge_index_substruct = substruct_data.edge_index
            data.center_substruct_idx = torch.tensor([substruct_node_map[
                                                          root_idx]])  # need
            # to convert center idx from original graph node ordering to the
            # new substruct node ordering

        # Get subgraphs that is between l1 and l2 hops away from the root node
        l1_node_idxes = nx.single_source_shortest_path_length(G, root_idx,
                                                              self.l1).keys()
        l2_node_idxes = nx.single_source_shortest_path_length(G, root_idx,
                                                              self.l2).keys()
        context_node_idxes = set(l1_node_idxes).symmetric_difference(
            set(l2_node_idxes))
        if len(context_node_idxes) > 0:
            context_G = G.subgraph(context_node_idxes)
            context_G, context_node_map = reset_idxes(context_G)  # need to
            # reset node idx to 0 -> num_nodes - 1, otherwise data obj does not
            # make sense, since the node indices in data obj must start at 0
            context_data = nx_to_graph_data_obj_simple(context_G)
            data.x_context = context_data.x
            data.edge_attr_context = context_data.edge_attr
            data.edge_index_context = context_data.edge_index

        # Get indices of overlapping nodes between substruct and context,
        # WRT context ordering
        context_substruct_overlap_idxes = list(set(
            context_node_idxes).intersection(set(substruct_node_idxes)))
        if len(context_substruct_overlap_idxes) > 0:
            context_substruct_overlap_idxes_reorder = [context_node_map[old_idx]
                                                       for
                                                       old_idx in
                                                       context_substruct_overlap_idxes]
            # need to convert the overlap node idxes, which is from the
            # original graph node ordering to the new context node ordering
            data.overlap_context_substruct_idx = \
                torch.tensor(context_substruct_overlap_idxes_reorder)

        return data


class BatchSubstructContext(Data):
    r"""A plain old python object modeling a batch of graphs as one big
    (dicconnected) graph. With :class:`torch_geometric.data.Data` being the
    base class, all its methods can also be used here.
    In addition, single graphs can be reconstructed via the assignment vector
    :obj:`batch`, which maps each node to its respective graph identifier.
    """

    """
    Specialized batching for substructure context pair!
    """

    def __init__(self, batch=None, **kwargs):
        super(BatchSubstructContext, self).__init__(**kwargs)
        self.batch = batch

    @staticmethod
    def from_data_list(data_list):
        r"""Constructs a batch object from a python list holding
        :class:`torch_geometric.data.Data` objects.
        The assignment vector :obj:`batch` is created on the fly."""
        #keys = [set(data.keys) for data in data_list]
        #keys = list(set.union(*keys))
        #assert 'batch' not in keys

        batch = BatchSubstructContext()
        keys = ["center_substruct_idx", "edge_attr_substruct", "edge_index_substruct", "x_substruct", "overlap_context_substruct_idx", "edge_attr_context", "edge_index_context", "x_context"]

        for key in keys:
            #print(key)
            batch[key] = []

        #batch.batch = []
        #used for pooling the context
        batch.batch_overlapped_context = []
        batch.overlapped_context_size = []

        cumsum_main = 0
        cumsum_substruct = 0
        cumsum_context = 0

        i = 0
        
        for data in data_list:
            #If there is no context, just skip!!
            if hasattr(data, "x_context"):
                num_nodes = data.num_nodes
                num_nodes_substruct = len(data.x_substruct)
                num_nodes_context = len(data.x_context)

                #batch.batch.append(torch.full((num_nodes, ), i, dtype=torch.long))
                batch.batch_overlapped_context.append(torch.full((len(data.overlap_context_substruct_idx), ), i, dtype=torch.long))
                batch.overlapped_context_size.append(len(data.overlap_context_substruct_idx))

                ###batching for the main graph
                #for key in data.keys:
                #    if not "context" in key and not "substruct" in key:
                #        item = data[key]
                #        item = item + cumsum_main if batch.cumsum(key, item) else item
                #        batch[key].append(item)
                
                ###batching for the substructure graph
                for key in ["center_substruct_idx", "edge_attr_substruct", "edge_index_substruct", "x_substruct"]:
                    item = data[key]
                    item = item + cumsum_substruct if batch.cumsum(key, item) else item
                    batch[key].append(item)
                

                ###batching for the context graph
                for key in ["overlap_context_substruct_idx", "edge_attr_context", "edge_index_context", "x_context"]:
                    item = data[key]
                    item = item + cumsum_context if batch.cumsum(key, item) else item
                    batch[key].append(item)

                cumsum_main += num_nodes
                cumsum_substruct += num_nodes_substruct   
                cumsum_context += num_nodes_context
                i += 1

        for key in keys:
            batch[key] = torch.cat(
                batch[key], dim=batch.cat_dim(key))
        #batch.batch = torch.cat(batch.batch, dim=-1)
        batch.batch_overlapped_context = torch.cat(batch.batch_overlapped_context, dim=-1)
        batch.overlapped_context_size = torch.LongTensor(batch.overlapped_context_size)

        return batch.contiguous()

    def cat_dim(self, key):
        return -1 if key in ["edge_index", "edge_index_substruct", "edge_index_context"] else 0

    def cumsum(self, key, item):
        r"""If :obj:`True`, the attribute :obj:`key` with content :obj:`item`
        should be added up cumulatively before concatenated together.
        .. note::
            This method is for internal use only, and should only be overridden
            if the batch concatenation process is corrupted for a specific data
            attribute.
        """
        return key in ["edge_index", "edge_index_substruct", "edge_index_context", "overlap_context_substruct_idx", "center_substruct_idx"]

    @property
    def num_graphs(self):
        """Returns the number of graphs in the batch."""
        return self.batch[-1].item() + 1


def temp_func(data_list):
    return BatchSubstructContext.from_data_list(data_list)

class DataLoaderSubstructContext(torch.utils.data.DataLoader):
    r"""Data loader which merges data objects from a
    :class:`torch_geometric.data.dataset` to a mini-batch.
    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How may samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch (default: :obj:`True`)
    """

    def __init__(self, dataset, batch_size=1, shuffle=True, **kwargs):
        super().__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=temp_func,
            **kwargs)
