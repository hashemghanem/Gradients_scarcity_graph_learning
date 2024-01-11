
import os.path as osp
import os
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
import torch
import networkx as nx
import math
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse


def Gaussian_pdf(mu, std):
    def pdf(x):
        dif = ((x-mu)**2).sum(len(x.shape)-1)
        return torch.exp(-dif/2/std**2)
    return pdf


def Gaussians_sum_labeling(x):
    npoints = x.shape[-2]
    rang = x.max()
    y = torch.zeros(x.shape[0:2], device=x.device)
    for ind in range(x.shape[0]):
        g = [Gaussian_pdf(rang*torch.rand(x.shape[-1], device=x.device), .2)
             for _ in range(3)]
        g = [gi(x[ind:ind+1]) for gi in g]
        for gi in g:
            y[ind:ind+1] = y[ind:ind+1] + gi
    y = y - y.min(dim=1, keepdim=True).values.expand(-1, npoints)
    y = y/(y.max(dim=1, keepdim=True).values.expand(-1, npoints))
    return y


def cheaters_network(nsamples=1000, npoints=32, ncoordinates=10,
                     passing_threshold=1.5, epsilon=0.23, aggregation='max',
                     scale_adj=1, train_in_size=10, train_out_size=10,
                     val_size=10, test_size=10, sigma=0.03):

    pointset = torch.rand((nsamples, npoints, ncoordinates))
    # sorting points by their x
    pointset[:, :, 0] = pointset[:, :, 0].sort()[0]
    marks = pointset[:, :, 1:].clone()
    xy = pointset[:, :, :1].clone()
    dist = (xy[:, None, :, :]).expand(-1, npoints, -1, -1)
    dist = -((dist - dist.transpose(1, 2))**2).sum(dim=3)/epsilon**2
    adj_star = torch.exp(dist)
    triu = torch.triu_indices(npoints, npoints)
    discrete_edges = torch.bernoulli(adj_star[:, triu[0], triu[1]])
    adj = torch.zeros_like(adj_star)
    adj[:, triu[0], triu[1]] = discrete_edges
    adj[:, triu[1], triu[0]] = discrete_edges
    if aggregation == 'max':
        marks = marks[:, None, :, :]
        marks = marks.expand(-1, npoints, -1, -1)
        repeat_adj = adj_star[:, :, :, None].repeat(1, 1, 1, marks.size(3))
        propag = scale_adj*(marks*repeat_adj).max(dim=2).values
    elif aggregation == 'sum':
        propag = torch.bmm(scale_adj*adj_star, marks)

    before = marks.sum(dim=2)
    labels = (propag.sum(dim=2))
    labels[labels < before] = before[labels < before]
    labels = labels > passing_threshold

    before = before > passing_threshold
    nchanged_labels = (before != labels).float().sum().item()/nsamples
    print('Avergae #nodes passing without cheating:',
          before.float().sum().item()/nsamples)
    print('Average #nodes whose labels changed with propagation: ', nchanged_labels)
    print('Average #number of points who pass:',
          labels.int().sum().item()/nsamples)
    print('Average #edges per graph:',
          ((adj != 0).sum().item()-nsamples*npoints)/nsamples/2)
    print("Target num of edges (n log n):", math.log(npoints)*npoints)
    train_in_mask = torch.zeros(npoints, dtype=torch.bool)
    train_in_mask[:train_in_size//2] = True
    train_in_mask[-train_in_size//2:] = True
    train_out_mask = torch.zeros(npoints, dtype=torch.bool)
    train_out_mask[npoints//2-train_out_size //
                   2:npoints//2+train_out_size//2] = True
    val_indices = torch.multinomial(torch.ones(val_size+test_size), val_size)
    div_val_test = torch.zeros(val_size+test_size, dtype=torch.bool)
    div_val_test[val_indices] = True
    val_mask = torch.zeros(npoints, dtype=torch.bool)
    val_mask[~(train_in_mask+train_out_mask)] = div_val_test
    test_mask = torch.zeros(npoints, dtype=torch.bool)
    test_mask[~(train_in_mask+train_out_mask)] = ~div_val_test
    if nsamples > 1:
        return Data(x=pointset, y=labels.long(), adj_star=adj_star,
                    train_in_mask=train_in_mask,
                    train_out_mask=train_out_mask,
                    val_mask=val_mask, test_mask=test_mask,
                    num_features=ncoordinates,
                    num_classes=2)
    else:
        edge_index, edge_attr = dense_to_sparse(adj[0])
        data = Data(x=pointset[0], edge_index=edge_index, edge_attr=edge_attr,
                    adj_star=adj_star[0], y=labels.flatten().long(),
                    train_in_mask=train_in_mask,
                    train_out_mask=train_out_mask,
                    val_mask=val_mask, test_mask=test_mask,
                    num_features=ncoordinates,
                    num_classes=2)
        return data


def fetch_Planetoid_dataset(dataset='Cora'):
    path = osp.join(os.getcwd(), '..', 'data', dataset)
    dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
    data = dataset[0]

    # We change the train_mask to train_in_mask, as this is how it is used.
    data.train_in_mask = data.train_mask.clone()
    # To save memory.
    del data.train_mask
    # Divide the valid set evenly at random into outer training and valid sets.
    # First, pick indices to pick for the outer training set.
    ind = torch.randint(low=0, high=2, size=(
        data.val_mask.sum(),), dtype=torch.bool)
    # Initialize mask.
    data.train_out_mask = data.val_mask.clone()
    # Set non-chosen indices to zero in train_out_mask.
    data.train_out_mask[data.train_out_mask == True] = ind
    # Set chosen indices to zero in validation mask.
    data.val_mask[data.val_mask == True] = torch.logical_not(ind)
    # Determine the output directory
    data.num_features = dataset.num_features
    data.num_classes = dataset.num_classes
    return data


def synthetic1(nsamples=100, npoints=200, ncoordinates=2,
               sigma=1.,  rang=None, train_in_size=10,
               train_out_size=10, val_size=10):
    """G2G+laplacian Gaussian kernel."""
    if rang is None:
        rang = 1.
    x = rang*torch.rand((nsamples, npoints, ncoordinates))
    x[:, :, 0] = x[:, :, 0].sort()[0]
    dist = (x[:, None, :, :]).expand(-1, npoints, -1, -1)
    dist = -((dist - dist.transpose(1, 2))**2).sum(dim=3)/sigma**2
    adj_star = -dist < 1
    adj_star[0].fill_diagonal_(False)
    triu = torch.triu_indices(npoints, npoints)
    discrete_edges = (adj_star[:, triu[0], triu[1]])
    adj = torch.zeros_like(adj_star)
    adj[:, triu[0], triu[1]] = discrete_edges
    adj[:, triu[1], triu[0]] = discrete_edges
    print("Num of edges is:", (adj[0].sum().item() - npoints)/2)
    print("Target num of edges:", math.log(npoints)*npoints)
    train_in_mask = torch.zeros(npoints, dtype=torch.bool)
    train_in_mask[:train_in_size//2] = True
    train_in_mask[-train_in_size//2:] = True
    train_out_mask = torch.zeros(npoints, dtype=torch.bool)
    train_out_mask[npoints//2-train_out_size //
                   2:npoints//2+train_out_size//2] = True
    val_mask = torch.zeros(npoints, dtype=torch.bool)
    val_mask[train_in_size//2:train_in_size//2 + val_size] = True
    # Labeling function. y dim at the end is [nsamples, npoints].
    y = Gaussians_sum_labeling(x)
    if nsamples > 1:
        return Data(x=x, y=y, adj_star=adj_star, adj=adj,
                    train_in_mask=train_in_mask,
                    train_out_mask=train_out_mask,
                    val_mask=val_mask, num_features=ncoordinates)
    else:
        edge_index, edge_attr = dense_to_sparse(adj[0])
        data = Data(x=x[0], edge_index=edge_index, adj_star=adj_star[0],
                    edge_attr=edge_attr, y=y.flatten(),
                    train_in_mask=train_in_mask,
                    train_out_mask=train_out_mask,
                    val_mask=val_mask, num_features=ncoordinates)
        return data

    data.train_in_mask = torch.zeros(npoints, dtype=torch.bool)
    data.train_in_mask[((2*data.x-1)**2).sum(axis=1) < .08] = True
    size_vtr = int(data.train_in_mask.sum())

    if high_freq_in:
        data.train_in_mask = torch.zeros(npoints, dtype=torch.bool)
        data.train_in_mask[np.random.choice(
            npoints, size=size_vtr, replace=False)] = True

    data.train_out_mask = torch.zeros(npoints, dtype=torch.bool)
    data.train_out_mask[((2*data.x-1)**2).sum(axis=1) < .02] = True
    size_vtout = int(data.train_out_mask.sum())
    if high_freq_out:
        data.train_out_mask = torch.zeros(npoints, dtype=torch.bool)
        data.train_out_mask[np.random.choice(
            npoints, size=size_vtout, replace=False)] = True


def fetch_dataset(dataset):
    # Load the dataset
    if dataset in ["Cora", "CiteSeer", "PubMed"]:
        return fetch_Planetoid_dataset(dataset)
    elif dataset == 'Cheaters':
        return cheaters_network(nsamples=1, npoints=256, ncoordinates=10,
                                passing_threshold=59., epsilon=.027,
                                aggregation='sum', scale_adj=1.,
                                train_in_size=256//4, train_out_size=256//4,
                                val_size=256//4, test_size=256//4)
    elif dataset == "Synthetic1HighFrequency":
        return synthetic1()
    elif dataset == "Synthetic1LowFrequency":
        return synthetic1()

def make_graph_connected(data):
    G = nx.Graph()
    npoints = data.x.shape[0]
    G.add_nodes_from([i for i in range(npoints)])
    G.add_edges_from([(i[0], i[1]) for i in data.edge_index.T.tolist()])
    comps = list(nx.connected_components(G))
    counter = 0
    for i in range(len(comps)-1):
        G.add_edge(list(comps[i])[0], list(comps[i+1])[0])
        counter += 1
    data.edge_index = torch.tensor(list(G.edges)).T
    data.edge_index = torch.cat(
        (data.edge_index, data.edge_index[[1, 0]]), dim=1)
    print("The number of connected componenets is:", counter)

    return data
