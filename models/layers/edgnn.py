"""
edGNN layer (add link to the paper)
"""
import torch
import torch.nn as nn

from utils.inits import reset, init_weights
from utils.utils import reset_graph_features

from utils.constants import GNN_MSG_KEY, GNN_NODE_FEAT_IN_KEY, GNN_NODE_FEAT_OUT_KEY, GNN_EDGE_FEAT_KEY, GNN_AGG_MSG_KEY


class edGNNLayer(nn.Module):
    """
    Simple GNN layer
    """

    def __init__(self,
                 g,
                 node_dim,
                 edge_dim,
                 out_feats,
                 activation=None,
                 dropout=None,
                 bias=None,
                 is_cuda=True):
        """
        edGNN Layer constructor.

        Args:
            g (dgl.DGLGraph): instance of DGLGraph defining the topology for message passing
            node_dim (int): node dimension
            edge_dim (int): edge dimension (if 1-hot, edge_dim = n_rels)
            out_feats (int): hidden dimension
            activation: pyTorch functional defining the nonlinearity to use
            dropout (float or None): dropout probability
            bias (bool): if True, a bias term will be added before applying the activation
        """
        super(edGNNLayer, self).__init__()

        #? 1. set parameters
        self.g = g
        self.node_dim = node_dim
        self.out_feats = out_feats
        self.activation = activation
        self.dropout = dropout
        self.edge_dim = edge_dim
        self.bias = bias
        self.is_cuda = is_cuda

        #? 2. create variables
        self._build_parameters()

        #? 3. initialize variables
        self.apply(init_weights)
        self.reset_parameters()


    def reset_parameters(self):
        reset(self.linear)


    def _build_parameters(self):
        """
        Build parameters and store them in a dictionary.
        The keys are the same keys of the node features to which we are applying the parameters.
        """
        input_dim = 2 * self.node_dim
        # print('self.node_dim', self.node_dim)
        # print('self.edge_dim', self.edge_dim)
        if self.edge_dim is not None:
            input_dim = input_dim + self.edge_dim
        print('input_dim', input_dim)
        print('self.out_feats', self.out_feats)
        self.linear = nn.Linear(input_dim, self.out_feats, bias=self.bias)

        # Dropout module
        if self.dropout:
            self.dropout = nn.Dropout(p=self.dropout)


    def gnn_msg(self, edges):
        """
        If edge features: for each edge u->v, return as msg: MLP(concat([h_u, h_uv]))
        """
        if self.g.edata is not None:
            # print('GNN_NODE_FEAT_IN_KEY', GNN_NODE_FEAT_IN_KEY)
            # print('edges.src[GNN_NODE_FEAT_IN_KEY]', edges.src[GNN_NODE_FEAT_IN_KEY])
            # print('edges.data[GNN_EDGE_FEAT_KEY]', edges.data[GNN_EDGE_FEAT_KEY])
            # print(edges.src[GNN_NODE_FEAT_IN_KEY].shape)
            # print(edges.data[GNN_EDGE_FEAT_KEY].shape)
            msg = torch.cat([edges.src[GNN_NODE_FEAT_IN_KEY], edges.data[GNN_EDGE_FEAT_KEY]], dim=1)
            if self.dropout:
                msg = self.dropout(msg)
        else:
            msg = edges.src[GNN_NODE_FEAT_IN_KEY]
            if self.dropout:
                msg = self.dropout(msg)
        # print('edges.src[GNN_NODE_FEAT_IN_KEY]', edges.src[GNN_NODE_FEAT_IN_KEY].shape)
        # print('edges.data[GNN_EDGE_FEAT_KEY]', edges.data[GNN_EDGE_FEAT_KEY].shape)
        return {GNN_MSG_KEY: msg}


    def gnn_reduce(self, nodes):
        accum = torch.sum((nodes.mailbox[GNN_MSG_KEY]), 1)
        # print('nodes.mailbox[GNN_MSG_KEY]', nodes.mailbox[GNN_MSG_KEY])
        return {GNN_AGG_MSG_KEY: accum}


    def node_update(self, nodes):
        # print('nodes', nodes)
        # print('nodes.data[GNN_NODE_FEAT_IN_KEY].shape', nodes.data[GNN_NODE_FEAT_IN_KEY].shape)
        # print('nodes.data[GNN_AGG_MSG_KEY].shape', nodes.data[GNN_AGG_MSG_KEY].shape)
        h = torch.cat([nodes.data[GNN_NODE_FEAT_IN_KEY],
                       nodes.data[GNN_AGG_MSG_KEY]],
                      dim=1)
        # h = h.type(torch.cuda.LongTensor)
        # print('h.shape', h.shape)
        h = self.linear(h)

        if self.activation:
            h = self.activation(h)

        if self.dropout:
            h = self.dropout(h)

        return {GNN_NODE_FEAT_OUT_KEY: h}


    def forward(self, node_features, edge_features, g):

        if g is not None:
            self.g = g

        #? 1. clean graph features
        reset_graph_features(self.g)

        #? 2. set current iteration features
        self.g.ndata[GNN_NODE_FEAT_IN_KEY] = node_features
        self.g.edata[GNN_EDGE_FEAT_KEY] = edge_features

        #? 3. aggregate messages
        self.g.update_all(self.gnn_msg,
                          self.gnn_reduce,
                          self.node_update)

        h = self.g.ndata.pop(GNN_NODE_FEAT_OUT_KEY)

        return h
