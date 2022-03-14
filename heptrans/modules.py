import torch
import torch.nn as nn
import dgl
import dgl.function as fn
from dgl.nn.pytorch.conv.gatv2conv import GATv2Conv

class GNNConv(GATv2Conv):
    def forward(self, graph, feat, get_attention=False):
        h_src = h_dst = feat
        feat_src = self.fc_src(h_src).view(-1, self._num_heads, self._out_feats)

        graph.srcdata.update({'el': feat_src})
        graph.update_all(fn.copy_u('el', 'm'), fn.mean('m', 'ft'))
        rst = graph.dstdata['ft']

        if self.res_fc is not None:
            resval = self.res_fc(h_dst).view(h_dst.shape[0], -1, self._out_feats)
            rst = rst + resval

        return rst

class DeepGraphConvLayer(nn.Module):
    def __init__(self, inputs, nodes, num_heads, gat=False):
        super(DeepGraphConvLayer, self).__init__()

        gat_nodes = nodes // num_heads

        self.nodes = nodes
        self.num_heads = num_heads

        if gat:
            self.conv = GATv2Conv(inputs, 
                                  gat_nodes, 
                                  num_heads=num_heads,
                                  residual=True,
                                  bias=False,
                                  share_weights=True,
                                  allow_zero_in_degree=True)
        else:
            self.conv = GNNConv(inputs,
                                gat_nodes,
                                num_heads=num_heads,
                                residual=True,
                                bias=False,
                                share_weights=True,
                                allow_zero_in_degree=True)

        self.bn = nn.BatchNorm1d(nodes)
        self.relu = nn.ReLU()

    def forward(self, g, attn=False):
        in_feat = g.ndata['features']

        batch_size = g.batch_size
        node_size = in_feat.size(0) // batch_size

        if attn:
            in_feat, rtn_attn = self.conv(g, in_feat, get_attention=attn)
            num_edges = g.num_edges() // batch_size
            rtn_attn = torch.squeeze(rtn_attn)
            rtn_attn = rtn_attn.reshape(batch_size, num_edges, self.num_heads)
            rtn_attn = rtn_attn.permute(0, 2, 1)
        else:
            in_feat = self.conv(g, in_feat)

        in_feat = in_feat.reshape(batch_size*node_size, self.nodes)
        in_feat = self.bn(in_feat)
        in_feat = self.relu(in_feat)

        g.ndata['features'] = in_feat

        if attn:
            return [g, rtn_attn]
        else:
            return g

class DeepGraphConvModel(nn.Module):
    def __init__(self, layers, nodes, num_heads, gat=False):
        super(DeepGraphConvModel, self).__init__()

        convs = [DeepGraphConvLayer(5, nodes, num_heads, gat)]
        for ii in range(layers-1):
            convs.append(DeepGraphConvLayer(nodes, nodes, num_heads, gat))

        self.convs = nn.Sequential(*convs)  

    def forward(self, g, attn=False):
        rtn_attn = []

        for conv in self.convs:
            if attn:
                g, g_attn = conv(g, attn)
                rtn_attn.append(g_attn)
            else:
                g = conv(g)

        g = dgl.mean_nodes(g, feat='features')    

        if attn:
            return [g] + rtn_attn
        else:
            return g

class TransferModel(nn.Module):
    def __init__(self, features=None, layers=2, nodes=16, num_heads=4, attn=False):
        super(TransferModel, self).__init__()
      
        self.attn = attn
        self.layers = layers
        
        if features == 'gnn':
            self.features = DeepGraphConvModel(layers,
                                               nodes,
                                               num_heads)
        elif features == 'gat':
            self.features = DeepGraphConvModel(layers,
                                               nodes,
                                               num_heads,
                                               gat=True)

        self.fc = nn.Linear(nodes, 2)

    def set_attn(self, conf=True):
        self.attn = conf

    def forward(self, g):
        if self.attn:
            g, *attn  = self.features(g, self.attn)
        else:
            g = self.features(g)
            attn = [g]*self.layers # dummy data

        g = self.fc(g)
        return [g] + attn
