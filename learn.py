import argparse

import os.path as osp
import os
import sys
import math
import higher
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm, LogNorm
import torch
import networkx as nx
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.utils import dense_to_sparse
from torch_geometric.datasets import Planetoid
from modules.datasets import fetch_dataset, get_hub_graph, cheaters_network, fetch_Planetoid_dataset
from modules.models import MlpG2g, Net, Alearner
from modules.myfuns import delete_files_in_directory
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def set_hyperparameters(args):
    g2g_hid_dim = GNN_hid_dim = graph_reg_mag = MAX_IN_ITER = MAX_OUT_ITER = lr_out = lr_in = None
    # Models' hyperparameters, first G2G hidden dimension:
    if args.method == "BO+G2G":
        g2g_hid_dim = 16 if args.dataset == 'Cheaters' else 32
    # GNN hidden dimension:
    if args.classifier == "GNN":
        GNN_hid_dim = 8 if args.dataset == 'Cheaters' else 128
    # graph_regularization_magnitude: regularization magnitude parameter.
    if args.method == 'BO+regularization':
        graph_reg_mag = 1.
    else:
        graph_reg_mag = 0.

    # Training hyperparameters, first number of inner iterations:
    if args.classifier == 'Laplacian':
        MAX_IN_ITER = 500
    elif args.classifier == 'GNN' and args.dataset == 'Cheaters':
        MAX_IN_ITER = 200
    elif args.classifier == 'GNN' and args.dataset in ['Cora', 'CiteSeer', 'PubMed']:
        MAX_IN_ITER = 100
    # Number of outer iterations:
    MAX_OUT_ITER = 150
    # Outer learning rate:
    if args.method == 'G2G' and args.classifier == 'Laplacian' and args.dataset in ['Cora', 'CiteSeer', 'PubMed']:
        lr_out = 0.001
    elif args.method == 'G2G' and args.classifier == 'GNN' and args.dataset in ['Cora', 'CiteSeer', 'PubMed']:
        lr_out = 0.0001
    elif args.method == 'G2G' and args.classifier == 'GNN' and args.dataset == 'Cheaters':
        lr_out = 0.001
    else:
        lr_out = 0.01
    # Inner learning rate:
    lr_in = 0.01 if args.classifier == 'GNN' else 0.1
    return g2g_hid_dim, GNN_hid_dim, graph_reg_mag, MAX_IN_ITER, MAX_OUT_ITER, lr_out, lr_in


def train(data, graph_gener, classifier, outer_optim, inner_optim, MAX_IN_ITER, MAX_OUT_ITER, lr_in, lr_out, graph_reg_mag, args):

    best_val_acc, best_test_acc = 0., 0.
    tr_in_loss, tr_out_loss = torch.zeros(
        MAX_OUT_ITER), torch.zeros(MAX_OUT_ITER)
    tr_in_acc, tr_out_acc = torch.zeros(
        MAX_OUT_ITER), torch.zeros(MAX_OUT_ITER)
    val_acc, test_acc = torch.zeros(MAX_OUT_ITER), torch.zeros(MAX_OUT_ITER)

    # Set the graph generator's hyperparameters
    graph_gener.train()
    # start training
    for itrout in range(0, MAX_OUT_ITER):
        classifier.train()
        edge_attr = graph_gener(data.x, data.edge_index)
        inner_optim.zero_grad()
        # compute edge_attr output
        with higher.innerloop_ctx(classifier, inner_optim) as (fmodel, diffopt):
            for itrin in range(MAX_IN_ITER-unrolled, MAX_IN_ITER):
                # compute predicted classes
                out = fmodel(data.x, data.edge_index, edge_attr)
                loss = F.cross_entropy(
                    out[data.train_in_mask], data.y[data.train_in_mask])
                diffopt.step(loss)
                tr_in_loss[itrout] = loss.item()
                accs = []
                for _, mask in data('train_in_mask', 'train_out_mask', 'val_mask', 'test_mask'):
                    pred = out[mask].max(1)[1]
                    acc = pred.eq(data.y[mask]).sum(
                    ).item() / mask.sum().item()
                    accs.append(acc)
                tr_in_acc[itrout], tr_out_acc[itrout], val_acc[itrout], test_acc[itrout] = accs
                if val_acc[itrout] > best_val_acc:
                    best_val_acc = val_acc[itrout].item()
                    best_test_acc = test_acc[itrout].item()
                    if False is True:
                        torch.save(graph_gener, os.path.join(
                            out_dir, "graph_gener.pth"))
                        torch.save(edge_attr, os.path.join(
                            out_dir, "edge_attr.pt"))
                        torch.save(classifier, os.path.join(
                            out_dir, "gnn.pth"))

                if itrin % 20 == 0:
                    print(f'Outer iteration: {itrout:03d}, Inner iteration: {itrin:03d}, '
                          f'Inner: {tr_in_acc[itrout]:.4f}, Outer: {tr_out_acc[itrout]:.4f}, '
                          f'Validation: {val_acc[itrout]: .4f}, Test: {test_acc[itrout]: .4f}, '
                          f' InLoss: {loss.item():.4f}')
            # Do one outer iteration, first:
            # compute predicted classes
            out = fmodel(data.x, data.edge_index, edge_attr)
            loss = F.cross_entropy(out[data.train_out_mask],
                                   data.y[data.train_out_mask])
            print(f"Outer loss:  {loss.item():.6f}, "
                  f"Best valid acc: {best_val_acc:.4f}, "
                  f"Best test acc: {best_test_acc:.4f}")
            tr_out_loss[itrout] = loss.item()
            # regularizing the graph
            adj_dense = torch.zeros((len(data.y), len(data.y)), device=device)
            adj_dense[data.edge_index[0], data.edge_index[1]] = edge_attr
            graph_reg = - graph_reg_mag * \
                torch.log(adj_dense.sum(dim=1)).mean()
            loss = loss + graph_reg
        outer_optim.zero_grad()
        loss.backward()
        # saved_grad = edge_attr.grad.clone()
        if itrout % 3 == 0:
            adj_to_save = torch.zeros((len(data.y), len(data.y)))
            grad_to_save = torch.zeros((len(data.y), len(data.y)))
            adj_to_save[data.edge_index[0], data.edge_index[1]] = edge_attr
            # grad_to_save[data.edge_index[0], data.edge_index[1]] = saved_grad
            gradstr = 'GradScreenshot_Iter'+str(itrout)+".pt"
            adjstr = 'AdjScreenshot_Iter'+str(itrout)+".pt"
            torch.save(grad_to_save.detach(), os.path.join(out_dir, gradstr))
            torch.save(adj_to_save.detach(), os.path.join(out_dir, adjstr))
            # print(edge_attr.grad)
        #     print("number of non zero grad", (edge_attr.grad != 0).sum().item())
        # print('is there nan in grad', torch.isnan(edge_attr.grad).sum().item())
        # if itrout % 10 == 0:
        #     plt.imshow(edge_attr.detach())
        #     plt.colorbar()
        #     plt.show()
        #     plt.imshow(edge_attr.grad.detach())
        #     plt.colorbar()
        #     plt.show()
        outer_optim.step()
        with torch.no_grad():
            edge_attr[edge_attr < 0] = 1e-10
            adj_to_save = torch.zeros((len(data.y), len(data.y)))
            adj_to_save[data.edge_index[0], data.edge_index[1]] = edge_attr
            adj_to_save = (adj_to_save+adj_to_save.T)/2
            edge_attr[:] = adj_to_save[data.edge_index[0], data.edge_index[1]]
        if False is True:
            torch.save(classifier, os.path.join(out_dir, "GNN_last.pth"))
            torch.save(edge_attr, os.path.join(out_dir, "g2g_last.pth"))
            torch.save(tr_in_loss[:itrout], os.path.join(
                out_dir, "tr_in_loss.pt"))
            torch.save(tr_out_loss[:itrout], os.path.join(
                out_dir, "tr_out_loss.pt"))
            torch.save(tr_in_acc[:itrout], os.path.join(
                out_dir, "tr_in_acc.pt"))
            torch.save(tr_out_acc[:itrout], os.path.join(
                out_dir, "tr_out_acc.pt"))
            torch.save(val_acc[:itrout], os.path.join(out_dir, "val_acc.pt"))
            torch.save(test_acc[:itrout], os.path.join(out_dir, "test_acc.pt"))
        # with torch.no_grad():
        #     new_params = dict(fmodel.named_parameters())
        #     for name, params in classifier.named_parameters():
        #         params.data.copy_(new_params[name])
    print(f"Best validation accuracy: {best_val_acc:.4f}, "
          f"Best test accuracy: {best_test_acc:.4f}")


if __name__ == "__main__":
    # Define/collect arguments
    parser = argparse.ArgumentParser(
        description='This is the graph learning script')
    parser.add_argument('dataset', type=str,
                        help='options: Cora, CiteSeer, PubMed, Cheaters, Synthetic1HighFrequency, Synthetic1LowFrequency')
    parser.add_argument('classifier', type=str, help='GNN or Laplacian')
    parser.add_argument('method', type=str,
                        help='The version of the bilevel framework when optimizing the graph, or A_obs when using the observed one. Options: BO, BO+G2G, BO+regularization, BO+A_obs^6, A_obs')
    args = parser.parse_args()

    if args.dataset in ['CiteSeer', 'PubMed'] and args.method != 'BO+G2G':
        msg = 'This code cannot handle the memory demand of learning from '\
            + args.dataset + " with " + args.method +\
            ". To make it do so,"
        if args.method == "BO+A_obs^6":
            msg += " change the method from BO+A_obs^6, and then "
        msg += "modify the code by reducing the number of inner iterations itrin."
        raise Exception(msg)

    # Set hyperparameters based on dataset and models
    g2g_hid_dim, GNN_hid_dim, graph_reg_mag, MAX_IN_ITER,\
        MAX_OUT_ITER, lr_out, lr_in = set_hyperparameters(args)

    # Delete previous output files in dir /outputs/
    out_dir = os.path.dirname(__file__)
    delete_files_in_directory(os.path.join(out_dir, "outputs"))

    # Fetch and save the dataset
    data = fetch_dataset(args.dataset)
    torch.save(data, os.path.join(out_dir, "dataset.pt"))

    # Migrate to GNN if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)

    # set the outer model: G2G or directly learning the adjacency matrix A.
    if args.method == 'BO+G2G':
        graph_gener = MlpG2g(data.x.shape[1], g2g_hid_dim).to(device)
    else:
        graph_gener = Alearner(data.edge_attr.shape[0]).to(device)

    outer_optim = torch.optim.Adam(graph_gener.parameters(), lr=lr_out)

    # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$S
    # to do, choose the classifier based on the argument passed by user.
    # classifier should be defined inside the outer loop maybe its optimizer too?
    # also loss function should be nn.CrossEntropyLoss() or nn.MSELoss() based on the classifier
    ###################
    classifier = Net(data.num_features, GNN_hid_dim,
                     data.num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_in)
    # Set the classifier's hyperparameters
    if args.classifier == 'GNN':
        classifier = classifier.to(device)
        classifier.train()
        inner_optim = torch.optim.Adam(classifier.parameters(), lr=lr_in)
        criterion = torch.nn.CrossEntropyLoss()
    elif args.classifier == 'Laplacian':
        classifier = classifier.to(device)
        classifier.train()
        inner_optim = torch.optim.Adam(classifier.parameters(), lr=lr_in)
        criterion = torch.nn.CrossEntropyLoss()
