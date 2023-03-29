import os.path as osp
import os
import sys
import higher
from numpy import indices
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid


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
