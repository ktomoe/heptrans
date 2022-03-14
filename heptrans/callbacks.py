import torch
import dgl

edges_src5 = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
              2, 2, 2, 2, 2, 3, 3, 3, 3, 3,
              4, 4, 4, 4, 4,]
edges_dst5 = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4,
              0, 1, 2, 3, 4, 0, 1, 2, 3, 4,
              0, 1, 2, 3, 4,]

edges_src6 = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1,
              2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3,
              4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5]
edges_dst6 = [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5,
              0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5,
              0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5]

edges_src8 = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
              2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
              4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5,
              6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7]
edges_dst8 = [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7,
              0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7,
              0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7,
              0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7]

graph_dict = {
    5: dgl.graph((edges_src5, edges_dst5), num_nodes=5),
    6: dgl.graph((edges_src6, edges_dst6), num_nodes=6),
    8: dgl.graph((edges_src8, edges_dst8), num_nodes=8),
}

def get_dgl(data, target):
    num_batch = data.shape[0]
    num_nodes = data.shape[1]

    data = data.reshape(num_batch*num_nodes, 5)
    data = torch.tensor(data, dtype=torch.float32)

    graphs = [graph_dict[num_nodes] for ii in range(num_batch)]
    graphs = dgl.batch(graphs)

    graphs.ndata['features'] = data

    return graphs, target
