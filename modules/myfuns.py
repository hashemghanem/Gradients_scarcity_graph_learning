import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torch_geometric
import networkx as nx
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.pylab as pylab
from matplotlib.ticker import SymmetricalLogLocator




def train_to_initialize_g2g(g2g, data, batch_size, lr, epochs, save_dir,
                            sparse_adj=True):
    # first define optimizer
    optim = torch.optim.Adam(g2g.parameters(), lr=lr)
    # create a dataset of "just" edge_index
    dataset = TensorDataset((data.edge_index[:, :-100]).transpose(0, 1))
    dataloader = DataLoader(dataset, batch_size, shuffle=True, drop_last=True)
    # define the loss fun
    bce = nn.BCELoss()
    # start learning
    best_val_acc = 0.
    # start training
    for epoch in range(epochs):
        g2g.train()
        for i, (edges,) in enumerate(dataloader):
            edges = edges.transpose(0, 1)
            non_edges = torch.randint(
                0, data.x.shape[0], (2, batch_size), device=edges.device)
            edge_index = torch.cat((edges, non_edges), dim=1)
            # target for given edges: 1s.
            target1 = torch.ones(batch_size, device=edges.device)
            # target for non-edges: 0s.
            target2 = torch.zeros(batch_size, device=edges.device)
            target = torch.cat((target1, target2))
            if sparse_adj:
                edge_atr = g2g(data.x, edge_index)
            else:
                adj_matrix = g2g(data.x)
                edge_atr = adj_matrix[edge_index[0], edge_index[1]]
            loss = bce(edge_atr, target)
            optim.zero_grad()
            loss.backward()
            optim.step()
            # computing the valid loss and accuracy.
            val_edges = data.edge_index[:, -100:]
            val_non_edges = torch.randint(0, data.x.shape[0],
                                          (2, val_edges.shape[1]),
                                          device=edges.device)
            val_edge_index = torch.cat((val_edges, val_non_edges), dim=1)
            target1 = torch.ones(val_edges.shape[1], device=edges.device)
            target2 = torch.zeros(val_edges.shape[1], device=edges.device)
            target = torch.cat((target1, target2))
            if sparse_adj:
                val_edge_atr = g2g(data.x, val_edge_index)
            else:
                adj_matrix = g2g(data.x)
                val_edge_atr = adj_matrix[val_edge_index[0], val_edge_index[1]]
            val_loss = bce(val_edge_atr, target)
            val_acc = (((val_edge_atr > .5).float())
                       == target).sum()/len(target)
            if i % 10 == 0:
                print(f"Epoch = {epoch:03d}, Batch= {i:03d}/{len(dataloader):03d}, "
                      f"loss = {loss.item():.4f}, valid loss = {val_loss.item():.4f}"
                      f" , valid acc = {val_acc.item():.3f}")
            if val_acc > best_val_acc:
                best_val_acc = val_acc.item()
                torch.save(g2g, os.path.join(save_dir, "g2g.pth"))
    print(f"Best validation accuracy: {best_val_acc:.4f} ")


def sparsify_dense(matrix, k):
    r"""Sparsifies the given dense matrix.

    Args:
        matrix (Tensor): Matrix to sparsify.
        k (int): num of coefficients to leave per col/row.
    :rtype: (:class:`LongTensor`, :class:`Tensor`)
    """
    N = matrix.shape[1]
    sort_idx = torch.argsort(matrix, dim=0, descending=True)
    top_idx = sort_idx[:k]
    edge_weight = torch.gather(matrix, dim=0,
                               index=top_idx).flatten()

    row_idx = torch.arange(0, N, device=matrix.device).repeat(k)
    edge_index = torch.stack([top_idx.flatten(), row_idx], dim=0)
    adj = torch.zeros_like(matrix)
    adj[edge_index[0], edge_index[1]] = edge_weight
    adj = (adj + adj.T)/2
    index = adj.nonzero(as_tuple=True)
    edge_attr = adj[index]
    return torch.stack(index, dim=0), edge_attr


def sample_edges_for_agr_model_gam_star(data, labels,
                                        mask, batch_size, verbose=False):
    """Sample edges to train the GAM* model not the gcn."""
    device = data.x.device
    # number of unique labels
    nlabels = data.y.unique().shape[0]
    # sample size to guarantee 50% -50% pos-neg agreement.
    sample_size = batch_size*nlabels
    labelled_nodes = torch.arange(end=labels.shape[0],
                                  device=device)[mask]
    # sample indices then nodes from the labelled pool.
    sampled_ind = torch.randint(low=0, high=labelled_nodes.shape[0],
                                size=(2, sample_size), device=device)
    sampled_nodes1 = labelled_nodes[sampled_ind[0]]
    sampled_nodes2 = labelled_nodes[sampled_ind[1]]
    labels1 = labels[sampled_nodes1]
    labels2 = labels[sampled_nodes2]
    pos_agr = labels1 == labels2
    if verbose:
        print(f"The number of found pos agreements is "
              f"{pos_agr.long().sum().item():}")
    sz_pos_agr = min(pos_agr.long().sum().item(), batch_size//2)
    # now get sz_pos_agr edges with positive agreement.
    pos_agr_ind = torch.arange(end=pos_agr.shape[0], device=device)[pos_agr]
    pos_agr_ind = pos_agr_ind[:sz_pos_agr]
    positives = torch.stack(
        (sampled_nodes1[pos_agr_ind], sampled_nodes2[pos_agr_ind]))
    pos_target = torch.ones_like(positives[0])
    # find negative agreements
    neg_agr = labels1 != labels2
    sz_neg_agr = batch_size - sz_pos_agr
    neg_agr_ind = torch.arange(end=neg_agr.shape[0], device=device)[neg_agr]
    neg_agr_ind = neg_agr_ind[:sz_neg_agr]
    negatives = torch.stack(
        (sampled_nodes1[neg_agr_ind], sampled_nodes2[neg_agr_ind]))
    neg_target = torch.zeros_like(negatives[0])
    # return a batch (edges, target weights)
    edge_index = torch.cat((positives, negatives), dim=1)
    target = torch.cat((pos_target, neg_target), dim=0)
    # return positives, negatives
    # print(neg_target.dtype, pos_target.dtype, target.dtype)
    return edge_index, target.float()


def sample_edges_for_agr_model(data, labels, mask, batch_size, verbose=False):
    """Sample edges to train the GAM model not the gcn."""
    device = data.x.device
    # Checking edges with both endpoints labelled.
    is_labelled1 = mask[data.edge_index[0]]
    is_labelled2 = mask[data.edge_index[1]]
    is_labelled = torch.logical_and(is_labelled1, is_labelled2)
    if verbose:
        print('The number of edges connecting labelled nodes is',
              is_labelled.long().sum().item())
    # get the edges with both ends labelled.
    labelled_edges = data.edge_index[:, is_labelled]
    labels1, labels2 = labels[labelled_edges[0]], labels[labelled_edges[1]]
    pos_agr_ind = labels1 == labels2
    positives = labelled_edges[:, pos_agr_ind]
    negatives = labelled_edges[:, ~pos_agr_ind]
    if verbose:
        print(f"Number of positive agreement: {pos_agr_ind.sum().item():}")
    # check if the given graph affords batch_size of labelled edges.
    greater_cardinal = max(pos_agr_ind.sum(), (~pos_agr_ind).sum())
    batch_size = min(batch_size, 2*greater_cardinal)
    # decide what to sample from the pool of all pairs of nodes.
    # negative agreements of postivie ones.
    sample_negative = True
    sample_size = pos_agr_ind.sum() - (~pos_agr_ind).sum()
    if pos_agr_ind.sum() < (~pos_agr_ind).sum():
        sample_negative = False
        sample_size = (~pos_agr_ind).sum() - pos_agr_ind.sum()
    extra_edges, _ = sample_edges_for_agr_model_gam_star(
        data, labels, mask, 2*sample_size)
    if sample_negative:
        negatives = torch.cat((negatives, extra_edges[:, -sample_size:]),
                              dim=1)
    else:
        positives = torch.cat((positives, extra_edges[:, :sample_size]),
                              dim=1)
    pos_target = torch.ones_like(positives[0])
    neg_target = torch.zeros_like(negatives[0])
    edge_index = torch.cat((positives, negatives), dim=1)
    target = torch.cat((pos_target, neg_target), dim=0)
    return edge_index, target.float()


def sample_edges_for_agr_reg_term_gam_star(data, labels, mask,
                                           batch_size, verbose=False):
    """Sample edges for the regularizer to train GCN with GAM*."""
    device = data.x.device
    # sample the LU edges, first sample labeled nodes
    # get labelled indices
    labelled_nodes = torch.arange(end=labels.shape[0], device=device)[mask]
    # sample labelled nodes
    sampled_nodes_l = torch.randint(
        high=labelled_nodes.shape[0], size=(batch_size,), device=device)
    sampled_nodes_l = labelled_nodes[sampled_nodes_l]
    # sample unlabelled nodes
    unlabelled_nodes = torch.arange(end=labels.shape[0], device=device)[~mask]
    sampled_nodes_u = torch.randint(
        high=unlabelled_nodes.shape[0], size=(batch_size,), device=device)
    sampled_nodes_u = unlabelled_nodes[sampled_nodes_u]
    edges_lu = torch.stack((sampled_nodes_l, sampled_nodes_u))
    # sample the UU edges
    unlabelled_nodes = torch.arange(end=labels.shape[0], device=device)[~mask]
    sampled_nodes_u = torch.randint(
        high=unlabelled_nodes.shape[0], size=(2, batch_size), device=device)
    sampled_nodes_u1 = unlabelled_nodes[sampled_nodes_u[0]]
    sampled_nodes_u2 = unlabelled_nodes[sampled_nodes_u[1]]
    edges_uu = torch.stack((sampled_nodes_u1, sampled_nodes_u2))
    return edges_lu, edges_uu


def sample_edges_for_agr_reg_term(data, labels, mask,
                                  batch_size, verbose=False):
    """Sample edges for the regularizer to train GCN with GAM."""
    device = data.x.device
    # sample the LU edges, first sample labeled nodes
    is_labelled1 = mask[data.edge_index[0]]
    is_labelled2 = mask[data.edge_index[1]]
    # if you have (labelled, unlabelled) in the edge_index, flip it.
    ind_labelled_unlabelled = torch.logical_and(is_labelled1, ~is_labelled2)
    edges_labelled_unlabelled = data.edge_index[:, ind_labelled_unlabelled]
    # now sample batch_size  labelled-2-unlabelled edges.
    ind_lu = torch.multinomial(torch.ones_like(edges_labelled_unlabelled[0]).float(),
                               num_samples=batch_size, replacement=False)
    edges_labelled_unlabelled = edges_labelled_unlabelled[:, ind_lu]

    # do the same with unlabelled to labelled edges.
    ind_unlabelled_unlabelled = torch.logical_and(~is_labelled1, ~is_labelled2)
    edges_unlabelled_unlabelled = data.edge_index[:, ind_unlabelled_unlabelled]
    # now sample batch_size  labelled-2-unlabelled edges.
    ind_uu = torch.multinomial(torch.ones_like(edges_unlabelled_unlabelled[0]).float(),
                               num_samples=batch_size, replacement=False)
    edges_unlabelled_unlabelled = edges_unlabelled_unlabelled[:, ind_uu]
    return edges_labelled_unlabelled, edges_unlabelled_unlabelled


def loss_gcn_supervised_gam(output, labels, mask, batch_size, verbose=False):
    """Calculate the classification loss for gcns in GAM."""
    device = labels.device
    labelled_nodes = torch.arange(end=labels.shape[0], device=device)[mask]
    # sample labelled nodes
    sampled_nodes_l = torch.randint(
        high=labelled_nodes.shape[0], size=(batch_size,), device=device)
    sampled_nodes_l = labelled_nodes[sampled_nodes_l]
    loss = F.cross_entropy(output[sampled_nodes_l], labels[sampled_nodes_l])
    return loss


def get_out_directory(is_main, gpu_or_cpu, current_dir):
    out_dir = os.path.join(current_dir, "saved_data")
    if is_main == '__main__':
        if len(gpu_or_cpu) > 1:
            if gpu_or_cpu[1] == "gpu":
                out_dir = out_dir+"_gpu"
            if gpu_or_cpu[1] == "gpu1":
                out_dir = out_dir+"_gpu1"
            elif gpu_or_cpu[1] == "batch":
                out_dir = out_dir+"_batch"
    return out_dir


def delete_files_in_directory(directory_path):
    try:
        files = os.listdir(directory_path)
        for file in files:
            file_path = os.path.join(directory_path, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
        print("All files deleted successfully.")
    except OSError:
        print("Error occurred while deleting files.")


def spath(data):
    """Return shortes path from labelled nodes."""
    Isolated_node_distance = np.inf
    G = nx.Graph()
    npoints = data.x.shape[0]
    G.add_nodes_from([i for i in range(npoints)])
    G.add_edges_from([(i[0], i[1]) for i in data.edge_index.T.tolist()])
    paths = dict(nx.shortest_path_length(G))
    dijkstra = Isolated_node_distance*torch.ones(npoints, npoints)
    for i in paths:
        for j in paths[i]:
            dijkstra[i, j] = paths[i][j]
    # distance of nodes to V_out+V_in
    dist_inout = dijkstra[data.train_in_mask +
                          data.train_out_mask].min(dim=0).values
    dist_in = dijkstra[data.train_in_mask].min(dim=0).values
    # distance of nodes to V_out
    dist_out = dijkstra[data.train_out_mask].min(dim=0).values
    return dist_in, dist_out, dist_inout


def weight_reset(m):
    if isinstance(m, torch_geometric.nn.GraphConv):
        m.reset_parameters()
    else:
        raise Exception("Not a graph neural network")



def bfs_layers(G, sources):
    """Returns an iterator of all the layers in breadth-first search traversal.

    Parameters
    ----------
    G : NetworkX graph
        A graph over which to find the layers using breadth-first search.

    sources : node in `G` or list of nodes in `G`
        Specify starting nodes for single source or multiple sources breadth-first search

    Yields
    ------
    layer: list of nodes
        Yields list of nodes at the same distance from sources

    Examples
    --------
    >>> G = nx.path_graph(5)
    >>> dict(enumerate(nx.bfs_layers(G, [0, 4])))
    {0: [0, 4], 1: [1, 3], 2: [2]}
    >>> H = nx.Graph()
    >>> H.add_edges_from([(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)])
    >>> dict(enumerate(nx.bfs_layers(H, [1])))
    {0: [1], 1: [0, 3, 4], 2: [2], 3: [5, 6]}
    >>> dict(enumerate(nx.bfs_layers(H, [1, 6])))
    {0: [1, 6], 1: [0, 3, 4, 2], 2: [5]}
    """
    if sources in G:
        sources = [sources]

    current_layer = list(sources)
    visited = set(sources)

    for source in current_layer:
        if source not in G:
            raise nx.NetworkXError(f"The node {source} is not in the graph.")

    # this is basically BFS, except that the current layer only stores the nodes at
    # same distance from sources at each iteration
    while current_layer:
        yield current_layer
        next_layer = []
        for node in current_layer:
            for child in G[node]:
                if child not in visited:
                    visited.add(child)
                    next_layer.append(child)
        current_layer = next_layer
def plot_hypergradient_GNNs_case(args, itrout=9):
    # Set the pyplot fonts, color cycles and markers etc.
    fontsize = 20
    plt.style.use('default')
    plt.rcParams.update({'font.size': fontsize})
    plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    # plt.rc('text', usetex=True)
    matplotlib.rc('text', usetex=True)
    matplotlib.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]

    line_width, marker_size = 2, 4
    cb_clr_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                    '#f781bf', '#a65628', '#984ea3',
                    '#999999', '#e41a1c', '#dede00']

    # Colorblind friendly color cycles
    ccycles = ['tableau-colorblind10', 'seaborn-colorblind']

    # markers in case needed
    mrk = ['s', 'v', 'o', 'x', '3', 'p', '|']
    fontsize = 20
    # making the labels/ticks' font larger
    params = {'legend.fontsize': fontsize,
            'axes.labelsize': fontsize,
            'xtick.labelsize': 16,
            'ytick.labelsize': 16,
            "axes.titlesize": 16,
            "axes.labelsize": 16}
    pylab.rcParams.update(params)

    # Set the color style (cycle + map) you want.
    plt.style.use(ccycles[0])

    # load the graph dataset and compute the edge distance.
    data = torch.load(os.path.join(args.output_dir, 'dataset.pt'),map_location=torch.device('cpu'))

    G = nx.Graph()
    npoints = data.x.shape[0]
    G.add_nodes_from([i for i in range(npoints)])
    G.add_edges_from([(i[0], i[1]) for i in data.edge_index.T.tolist()])

    dist_in, dist_out, dist_inout = -1 * torch.ones(npoints), -1 * torch.ones(npoints), -1 * torch.ones(npoints)
    # computing node distance to inner/outer training nodes
    for dist, nodes in enumerate(bfs_layers(G, torch.arange(data.x.shape[0])[data.train_in_mask].tolist())):
        dist_in[nodes] = dist
    for dist, nodes in enumerate(bfs_layers(G, torch.arange(data.x.shape[0])[data.train_out_mask].tolist())):
        dist_out[nodes] = dist
    for dist, nodes in enumerate(bfs_layers(G, torch.arange(data.x.shape[0])[data.train_out_mask+data.train_in_mask].tolist())):
        dist_inout[nodes] = dist
    print(args.dataset+" statistics:")
    print('furthest node from the union of both labelled node subsets', dist_inout.max())
    print('furthest node from outer labelled nodes', dist_out.max())
    print('furthest node from inner labelled nodes', dist_in.max())
    # Compute edge distance to inner/outer training nodes
    dist_in_edge = dist_in[data.edge_index.flatten()]
    dist_in_edge = dist_in_edge.reshape(data.edge_index.size()).min(dim=0).values
    dist_out_edge = dist_out[data.edge_index.flatten()]
    dist_out_edge = dist_out_edge.reshape(data.edge_index.size()).min(dim=0).values
    print("max dist_out_edge", dist_out_edge.max())
    print("max dist_in_edge", dist_in_edge.max())
    if args.inner_model =='Laplacian':
        dist_edge = dist_out_edge + dist_in_edge
    else:
        dist_edge = dist_inout[data.edge_index.flatten()]
        dist_edge = dist_edge.reshape(data.edge_index.size()).min(dim=0).values
        # In case we have nodes with no path to a labelled node, put their distance to 15 for plotting purposes:
        dist_edge[dist_edge == -1] = 15
    print("max dist_edge", dist_edge.max())
    # print percentage of edges with distance less than 2
    print("percentage of edges with distance less than 3",(dist_edge < 3).sum()/dist_edge.shape[0])
    # print number of edges with distance less than 2
    print ("number of edges with distance less than 3", (dist_edge < 3).sum())
    # print num of edges
    print ("number of edges", data.edge_index[0].shape[0])

    # load the hypergradient data
    gradstr = 'GradScreenshot_Iter'+str(itrout)+".pt"
    grad_no_fix = torch.load(os.path.join(args.output_dir, gradstr), map_location=torch.device('cpu'))
    # Plot the data on each subplot
    fig, axs = plt.subplots(constrained_layout=True)
    axs.scatter(dist_edge, grad_no_fix, marker=mrk[0], facecolors='none', edgecolors=cb_clr_cycle[0])
    axs.set_yscale('symlog', linthresh=1e-6)
    axs.set_title(args.dataset)
    xlabel = r'ECD to $V_{in}, V_{out}$' if args.inner_model =='Laplacian' else r'ED to $V_{in}\cup V_{out}$'
    axs.set_xlabel(xlabel)
    axs.set_ylabel(r"Hypergradient(" + args.inner_model + " case)")
    axs.grid()
    yticks = SymmetricalLogLocator(base=1000, linthresh=1e-6)
    axs.yaxis.set_major_locator(yticks)

    plt.savefig(os.path.join(args.output_dir,'figs/hypergradientscarcity'+args.dataset+args.inner_model+'.pdf'),bbox_inches='tight')
    plt.show()
