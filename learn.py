
# %%
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
from modules.datasets import get_hub_graph, cheaters_network, fetch_Planetoid_dataset
from modules.models import MlpG2gCora, Net
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

dataset = sys.argv[1]
classifier = sys.argv[2]
fix = sys.argv[3]
# Load the dataset
if dataset == 'Cora':
    data = fetch_Planetoid_dataset(dataset)
elif dataset == 'Cheaters':
    data = cheaters_network(nsamples=1, npoints=256, ncoordinates=10,
                            passing_threshold=59., epsilon=.027,
                            aggregation='sum', scale_adj=1.,
                            train_in_size=256//4, train_out_size=256//4,
                            val_size=256//4, test_size=256//4)

# Set hyperparameters
if dataset == 'Cora':
    g2g_hid_dim, GCN_hid_dim = 32, 128
    if classifier == 'Laplacian':
        MAX_IN_ITER = 500
    elif classifier == 'GCN':
        MAX_IN_ITER = 100
elif classifier == 'GCN' and dataset == 'Cheaters':
    MAX_IN_ITER = 200
    g2g_hid_dim, GCN_hid_dim = 16, 8
MAX_OUT_ITER = 150
if fix == 'Regularization':
    graph_reg_mag = 1.
else:
    graph_reg_mag = 0.

if fix = 'G2G' and classifier == 'Laplacian' and dataset == 'Cora':
    lr_out = 0.001
elif fix = 'G2G' and classifier == 'GCN' and dataset == 'Cora':
    lr_out = 0.0001
elif fix = 'G2G' and classifier == 'GCN' and dataset == 'Cheaters':
    lr_out = 0.001
else:
    lr_out = 0.01
lr_gcn, lr_laplacian = .01, 0.1
save = True

if save is True:
    torch.save(data, os.path.join(out_dir, "dataset.pt"))


out_dir = os.path.join(os.getcwd(), "saved_data")
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
tr_in_loss, tr_out_loss = torch.zeros(MAX_OUT_ITER), torch.zeros(MAX_OUT_ITER)
tr_in_acc, tr_out_acc = torch.zeros(MAX_OUT_ITER), torch.zeros(MAX_OUT_ITER)
val_acc, test_acc = torch.zeros(MAX_OUT_ITER), torch.zeros(MAX_OUT_ITER)

model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr_gcn)
# start training
for itrout in range(0, MAX_OUT_ITER):

    graph_gener.train()
    model.train()
    g2g = graph_gener(data.x, data.edge_index)
    print(g2g[:10])
    for itrin in range(MAX_IN_ITER - unrolled):
        out = model(data.x, data.edge_index, g2g)
        loss = F.cross_entropy(
            out[data.train_in_mask], data.y[data.train_in_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        tr_in_loss[itrout] = loss.item()
        accs = []
        for _, mask in data('train_in_mask', 'train_out_mask', 'val_mask', 'test_mask'):
            pred = out[mask].max(1)[1]
            acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
            accs.append(acc)
        tr_in_acc[itrout], tr_out_acc[itrout], val_acc[itrout], test_acc[itrout] = accs
        if val_acc[itrout] > best_val_acc:
            best_val_acc = val_acc[itrout].item()
            best_test_acc = test_acc[itrout].item()
            if save is True:
                torch.save(graph_gener, os.path.join(
                    out_dir, "graph_gener.pth"))
                torch.save(g2g, os.path.join(out_dir, "g2g.pth"))
                torch.save(model, os.path.join(out_dir, "gcn.pth"))
        if itrin % 10 == 0:
            print(f'Outer iteration: {itrout:03d}, Inner iteration: {itrin:03d}, '
                  f'Inner: {tr_in_acc[itrout]:.4f}, Outer: {tr_out_acc[itrout]:.4f}, '
                  f'Validation: {val_acc[itrout]: .4f}, Test: {test_acc[itrout]: .4f}, '
                  f' InLoss: {loss.item():.4f}')

    # model.eval()
    optimizer.zero_grad()
    # compute g2g output
    with higher.innerloop_ctx(model, optimizer) as (fmodel, diffopt):
        for itrin in range(MAX_IN_ITER-unrolled, MAX_IN_ITER):
            # compute predicted classes
            out = fmodel(data.x, data.edge_index, g2g)
            loss = F.cross_entropy(
                out[data.train_in_mask], data.y[data.train_in_mask])
            diffopt.step(loss)
            tr_in_loss[itrout] = loss.item()
            accs = []
            for _, mask in data('train_in_mask', 'train_out_mask', 'val_mask', 'test_mask'):
                pred = out[mask].max(1)[1]
                acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
                accs.append(acc)
            tr_in_acc[itrout], tr_out_acc[itrout], val_acc[itrout], test_acc[itrout] = accs
            if val_acc[itrout] > best_val_acc:
                best_val_acc = val_acc[itrout].item()
                best_test_acc = test_acc[itrout].item()
                if save is True:
                    torch.save(graph_gener, os.path.join(
                        out_dir, "graph_gener.pth"))
                    torch.save(g2g, os.path.join(out_dir, "g2g.pt"))
                    torch.save(model, os.path.join(out_dir, "gcn.pth"))

            if itrin % 20 == 0:
                print(f'Outer iteration: {itrout:03d}, Inner iteration: {itrin:03d}, '
                      f'Inner: {tr_in_acc[itrout]:.4f}, Outer: {tr_out_acc[itrout]:.4f}, '
                      f'Validation: {val_acc[itrout]: .4f}, Test: {test_acc[itrout]: .4f}, '
                      f' InLoss: {loss.item():.4f}')
        # Do one outer iteration, first:
        # compute predicted classes
        out = fmodel(data.x, data.edge_index, g2g)
        loss = F.cross_entropy(out[data.train_out_mask],
                               data.y[data.train_out_mask])
        print(f"Outer loss:  {loss.item():.6f}, "
              f"Best valid acc: {best_val_acc:.4f}, "
              f"Best test acc: {best_test_acc:.4f}")
        tr_out_loss[itrout] = loss.item()
        # regularizing the graph
        adj_dense = torch.zeros((len(data.y), len(data.y)), device=device)
        adj_dense[data.edge_index[0], data.edge_index[1]] = g2g
        graph_reg = - graph_reg_mag * torch.log(adj_dense.sum(dim=1)).mean()
        loss = loss + graph_reg
    g2g_optim.zero_grad()
    loss.backward()
    # saved_grad = g2g.grad.clone()
    if itrout % 3 == 0:
        adj_to_save = torch.zeros((len(data.y), len(data.y)))
        grad_to_save = torch.zeros((len(data.y), len(data.y)))
        adj_to_save[data.edge_index[0], data.edge_index[1]] = g2g
        # grad_to_save[data.edge_index[0], data.edge_index[1]] = saved_grad
        gradstr = 'GradScreenshot_Iter'+str(itrout)+".pt"
        adjstr = 'AdjScreenshot_Iter'+str(itrout)+".pt"
        torch.save(grad_to_save.detach(), os.path.join(out_dir, gradstr))
        torch.save(adj_to_save.detach(), os.path.join(out_dir, adjstr))
        # print(g2g.grad)
    #     print("number of non zero grad", (g2g.grad != 0).sum().item())
    # print('is there nan in grad', torch.isnan(g2g.grad).sum().item())
    # if itrout % 10 == 0:
    #     plt.imshow(g2g.detach())
    #     plt.colorbar()
    #     plt.show()
    #     plt.imshow(g2g.grad.detach())
    #     plt.colorbar()
    #     plt.show()
    g2g_optim.step()
    with torch.no_grad():
        g2g[g2g < 0] = 1e-10
        adj_to_save = torch.zeros((len(data.y), len(data.y)))
        adj_to_save[data.edge_index[0], data.edge_index[1]] = g2g
        adj_to_save = (adj_to_save+adj_to_save.T)/2
        g2g[:] = adj_to_save[data.edge_index[0], data.edge_index[1]]
    if save is True:
        torch.save(model, os.path.join(out_dir, "GCN_last.pth"))
        torch.save(g2g, os.path.join(out_dir, "g2g_last.pth"))
        torch.save(tr_in_loss[:itrout], os.path.join(out_dir, "tr_in_loss.pt"))
        torch.save(tr_out_loss[:itrout], os.path.join(
            out_dir, "tr_out_loss.pt"))
        torch.save(tr_in_acc[:itrout], os.path.join(out_dir, "tr_in_acc.pt"))
        torch.save(tr_out_acc[:itrout], os.path.join(out_dir, "tr_out_acc.pt"))
        torch.save(val_acc[:itrout], os.path.join(out_dir, "val_acc.pt"))
        torch.save(test_acc[:itrout], os.path.join(out_dir, "test_acc.pt"))
    # with torch.no_grad():
    #     new_params = dict(fmodel.named_parameters())
    #     for name, params in model.named_parameters():
    #         params.data.copy_(new_params[name])
print(f"Best validation accuracy: {best_val_acc:.4f}, "
      f"Best test accuracy: {best_test_acc:.4f}")
