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
from modules.models import MlpG2gCora, Net
from modules.myfuns import delete_files_in_directory
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

if __name__ == "__main__":

    # Define/collect arguments
    parser = argparse.ArgumentParser(
        description='This is the graph learning script')
    parser.add_argument('dataset', type=str,
                        help='options: Cora, CiteSeer, PubMed, Cheaters, Synthetic1HighFrequency, Synthetic1LowFrequency')
    parser.add_argument('classifier', type=str, help='GNN or Laplacian')
    parser.add_argument(
        'method', type=str, help='the choice of the outer loss. Options: BO, BO+G2G, BO+regularization, BO+A_obs^6')
    args = parser.parse_args()

    if args.dataset in ['CiteSeer', 'PubMed'] and args.method != 'BO+G2G':
        msg = 'This code cannot handle the memory demand of learning from '\
            + args.dataset + " with " + args.method +\
            ". To make it do so,"
        if args.method == "BO+A_obs^6":
            msg += " change the method from BO+A_obs^6, and then "
        msg += "modify the code by reducing the number of inner iterations itrin."
        raise Exception(msg)

    # Models' hyperparameters, first G2G hidden dimension:
    if args.method == "BO+G2G":
        g2g_hid_dim = 16 if args.dataset == 'Cheaters' else 32
    # GNN hidden dimension:
    if args.classifier == "GNN":
        GNN_hid_dim = 8 if args.dataset == 'Cheaters' else 128
    # whether to regularize the learned graph or not:
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
    lr_gnn, lr_laplacian = .01, 0.1

    # Delete previous output files in dir /outputs/
    out_dir = os.path.dirname(__file__)
    delete_files_in_directory(os.path.join(out_dir, "outputs"))

    # Fetch and save the dataset
    data = fetch_dataset(args.dataset)
    torch.save(data, os.path.join(out_dir, "dataset.pt"))

    # Migrate to GNN if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)

    graph_gener = MlpG2gCora(
        inp_dim=data.x.shape[1], hid_dim=g2g_hid_dim).to(device)
    with torch.no_grad():
        graph_gener.fc3.bias[:] = 1e-5
        graph_gener.fc3.weight[:] = 1e-5
    g2g_optim = torch.optim.Adam(graph_gener.parameters(), lr=lr_out)
    # g2g = .00001*torch.rand(data.edge_attr.size(), device=device)
    # g2g.requires_grad_(True)
    # g2g_optim = torch.optim.Adam([g2g], lr=lr_out)

    best_val_acc, best_test_acc = 0., 0.
    tr_in_loss, tr_out_loss = torch.zeros(
        MAX_OUT_ITER), torch.zeros(MAX_OUT_ITER)
    tr_in_acc, tr_out_acc = torch.zeros(
        MAX_OUT_ITER), torch.zeros(MAX_OUT_ITER)
    val_acc, test_acc = torch.zeros(MAX_OUT_ITER), torch.zeros(MAX_OUT_ITER)

    model = Net().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_gnn)
    # # start training
    # for itrout in range(0, MAX_OUT_ITER):

    #     graph_gener.train()
    #     model.train()
    #     g2g = graph_gener(data.x, data.edge_index)
    #     print(g2g[:10])
    #     for itrin in range(MAX_IN_ITER - unrolled):
    #         out = model(data.x, data.edge_index, g2g)
    #         loss = F.cross_entropy(
    #             out[data.train_in_mask], data.y[data.train_in_mask])
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #         tr_in_loss[itrout] = loss.item()
    #         accs = []
    #         for _, mask in data('train_in_mask', 'train_out_mask', 'val_mask', 'test_mask'):
    #             pred = out[mask].max(1)[1]
    #             acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
    #             accs.append(acc)
    #         tr_in_acc[itrout], tr_out_acc[itrout], val_acc[itrout], test_acc[itrout] = accs
    #         if val_acc[itrout] > best_val_acc:
    #             best_val_acc = val_acc[itrout].item()
    #             best_test_acc = test_acc[itrout].item()
    #             if save is True:
    #                 torch.save(graph_gener, os.path.join(
    #                     out_dir, "graph_gener.pth"))
    #                 torch.save(g2g, os.path.join(out_dir, "g2g.pth"))
    #                 torch.save(model, os.path.join(out_dir, "gnn.pth"))
    #         if itrin % 10 == 0:
    #             print(f'Outer iteration: {itrout:03d}, Inner iteration: {itrin:03d}, '
    #                   f'Inner: {tr_in_acc[itrout]:.4f}, Outer: {tr_out_acc[itrout]:.4f}, '
    #                   f'Validation: {val_acc[itrout]: .4f}, Test: {test_acc[itrout]: .4f}, '
    #                   f' InLoss: {loss.item():.4f}')

    #     # model.eval()
    #     optimizer.zero_grad()
    #     # compute g2g output
    #     with higher.innerloop_ctx(model, optimizer) as (fmodel, diffopt):
    #         for itrin in range(MAX_IN_ITER-unrolled, MAX_IN_ITER):
    #             # compute predicted classes
    #             out = fmodel(data.x, data.edge_index, g2g)
    #             loss = F.cross_entropy(
    #                 out[data.train_in_mask], data.y[data.train_in_mask])
    #             diffopt.step(loss)
    #             tr_in_loss[itrout] = loss.item()
    #             accs = []
    #             for _, mask in data('train_in_mask', 'train_out_mask', 'val_mask', 'test_mask'):
    #                 pred = out[mask].max(1)[1]
    #                 acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
    #                 accs.append(acc)
    #             tr_in_acc[itrout], tr_out_acc[itrout], val_acc[itrout], test_acc[itrout] = accs
    #             if val_acc[itrout] > best_val_acc:
    #                 best_val_acc = val_acc[itrout].item()
    #                 best_test_acc = test_acc[itrout].item()
    #                 if save is True:
    #                     torch.save(graph_gener, os.path.join(
    #                         out_dir, "graph_gener.pth"))
    #                     torch.save(g2g, os.path.join(out_dir, "g2g.pt"))
    #                     torch.save(model, os.path.join(out_dir, "gnn.pth"))

    #             if itrin % 20 == 0:
    #                 print(f'Outer iteration: {itrout:03d}, Inner iteration: {itrin:03d}, '
    #                       f'Inner: {tr_in_acc[itrout]:.4f}, Outer: {tr_out_acc[itrout]:.4f}, '
    #                       f'Validation: {val_acc[itrout]: .4f}, Test: {test_acc[itrout]: .4f}, '
    #                       f' InLoss: {loss.item():.4f}')
    #         # Do one outer iteration, first:
    #         # compute predicted classes
    #         out = fmodel(data.x, data.edge_index, g2g)
    #         loss = F.cross_entropy(out[data.train_out_mask],
    #                                data.y[data.train_out_mask])
    #         print(f"Outer loss:  {loss.item():.6f}, "
    #               f"Best valid acc: {best_val_acc:.4f}, "
    #               f"Best test acc: {best_test_acc:.4f}")
    #         tr_out_loss[itrout] = loss.item()
    #         # regularizing the graph
    #         adj_dense = torch.zeros((len(data.y), len(data.y)), device=device)
    #         adj_dense[data.edge_index[0], data.edge_index[1]] = g2g
    #         graph_reg = - graph_reg_mag * torch.log(adj_dense.sum(dim=1)).mean()
    #         loss = loss + graph_reg
    #     g2g_optim.zero_grad()
    #     loss.backward()
    #     # saved_grad = g2g.grad.clone()
    #     if itrout % 3 == 0:
    #         adj_to_save = torch.zeros((len(data.y), len(data.y)))
    #         grad_to_save = torch.zeros((len(data.y), len(data.y)))
    #         adj_to_save[data.edge_index[0], data.edge_index[1]] = g2g
    #         # grad_to_save[data.edge_index[0], data.edge_index[1]] = saved_grad
    #         gradstr = 'GradScreenshot_Iter'+str(itrout)+".pt"
    #         adjstr = 'AdjScreenshot_Iter'+str(itrout)+".pt"
    #         torch.save(grad_to_save.detach(), os.path.join(out_dir, gradstr))
    #         torch.save(adj_to_save.detach(), os.path.join(out_dir, adjstr))
    #         # print(g2g.grad)
    #     #     print("number of non zero grad", (g2g.grad != 0).sum().item())
    #     # print('is there nan in grad', torch.isnan(g2g.grad).sum().item())
    #     # if itrout % 10 == 0:
    #     #     plt.imshow(g2g.detach())
    #     #     plt.colorbar()
    #     #     plt.show()
    #     #     plt.imshow(g2g.grad.detach())
    #     #     plt.colorbar()
    #     #     plt.show()
    #     g2g_optim.step()
    #     with torch.no_grad():
    #         g2g[g2g < 0] = 1e-10
    #         adj_to_save = torch.zeros((len(data.y), len(data.y)))
    #         adj_to_save[data.edge_index[0], data.edge_index[1]] = g2g
    #         adj_to_save = (adj_to_save+adj_to_save.T)/2
    #         g2g[:] = adj_to_save[data.edge_index[0], data.edge_index[1]]
    #     if save is True:
    #         torch.save(model, os.path.join(out_dir, "GNN_last.pth"))
    #         torch.save(g2g, os.path.join(out_dir, "g2g_last.pth"))
    #         torch.save(tr_in_loss[:itrout], os.path.join(out_dir, "tr_in_loss.pt"))
    #         torch.save(tr_out_loss[:itrout], os.path.join(
    #             out_dir, "tr_out_loss.pt"))
    #         torch.save(tr_in_acc[:itrout], os.path.join(out_dir, "tr_in_acc.pt"))
    #         torch.save(tr_out_acc[:itrout], os.path.join(out_dir, "tr_out_acc.pt"))
    #         torch.save(val_acc[:itrout], os.path.join(out_dir, "val_acc.pt"))
    #         torch.save(test_acc[:itrout], os.path.join(out_dir, "test_acc.pt"))
    #     # with torch.no_grad():
    #     #     new_params = dict(fmodel.named_parameters())
    #     #     for name, params in model.named_parameters():
    #     #         params.data.copy_(new_params[name])
    # print(f"Best validation accuracy: {best_val_acc:.4f}, "
    #       f"Best test accuracy: {best_test_acc:.4f}")
