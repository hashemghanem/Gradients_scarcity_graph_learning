import torch
import torch.nn as nn
import higher
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GraphConv, APPNP
from torch_geometric.utils import dense_to_sparse




class GCN(torch.nn.Module):
    def __init__(self, inp_dim, hidden_dim, out_dim=7):
        super().__init__()
        self.conv1 = GraphConv(inp_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, out_dim)

    def forward(self, adj_est, x):
        npoints = x.shape[1]
        x = x.view(-1, x.shape[-1])
        edge_attr = edge_index = adj_est
        # edge_index, edge_attr = dense_to_sparse(adj_est)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index, edge_attr)
        return x.view(-1, npoints, x.shape[1])


class MlpPointWise(nn.Module):
    def __init__(self, inp_dim, hid_dim, out_dim, nlayers=1):
        super().__init__()
        self.nlayers = nlayers
        self.fc1 = nn.Linear(inp_dim, hid_dim)
        self.layers = nn.ModuleList(
            [nn.Linear(hid_dim, hid_dim) for _ in range(nlayers-1)])
        self.bns = nn.ModuleList(
            [nn.BatchNorm1d(hid_dim) for _ in range(nlayers-1)])
        self.fc2 = nn.Linear(hid_dim, out_dim)

    def forward(self, x):
        x = F.elu(self.fc1(x))
        for i in range(self.nlayers-1):
            x = F.elu(self.layers[i](x))
            # x = self.bns[i](x)
        x = self.fc2(x)
        return x





    def __init__(self, inp_dim, hid_dim, out_dim=1):
        super().__init__()
        self.embed_pnt = MlpPointWise(inp_dim, hid_dim, hid_dim, 1)
        self.fc1 = torch.nn.Linear(hid_dim, hid_dim)
        self.fc2 = torch.nn.Linear(hid_dim, hid_dim)
        self.fc3 = torch.nn.Linear(hid_dim, out_dim)

    def get_g2g_input(self, x, edge_index):
        feat1 = x[edge_index[0]]
        feat2 = x[edge_index[1]]
        feats = (feat1-feat2)**2
        return feats

    def forward(self, x, edge_index):
        x = self.embed_pnt(x)
        x = self.get_g2g_input(x, edge_index)
        # x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.fc1(x))
        # x = F.dropout(x, p=0.6, training=self.training)
        # x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x)).squeeze()
        # don't forget to change GAM  BCE loss if you use sigmoid
        return x

class MlpG2g(torch.nn.Module):
    def __init__(self, inp_dim, hid_dim, out_dim=1, nlayers=1):
        super().__init__()
        self.nlayers = nlayers
        self.dp1 = nn.Dropout(p=0.5)
        self.fc1 = torch.nn.Linear(inp_dim, hid_dim)
        self.layers = nn.ModuleList([nn.Linear(hid_dim, hid_dim) for _ in range(nlayers-1)])
        self.fc2 = torch.nn.Linear(hid_dim, hid_dim)
    def forward(self, x, edge_index):
        x = x[edge_index[0]] - x[edge_index[1]]
        x = self.dp1(x**2)
        x = F.elu(self.fc1(x))
        for i in range(self.nlayers-1):
            x = F.elu(self.layers[i](x))
        x = torch.sigmoid(self.fc2(x)).squeeze()
        return x


class mlp(nn.Module):
    def __init__(self, inp_dim, hid_dim, nlayers=1, p=.5):
        super().__init__()
        self.nlayers = nlayers
        self.fc1 = nn.Linear(inp_dim, hid_dim)
        self.layers = nn.ModuleList(
            [nn.Linear(hid_dim, hid_dim) for _ in range(nlayers-1)])
        self.bns = nn.ModuleList(
            [nn.BatchNorm1d(hid_dim) for _ in range(nlayers-1)])
        self.fc2 = nn.Linear(hid_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        for i in range(self.nlayers-1):
            x = F.relu(self.layers[i](x))
            x = self.bns[i](x)
        x = torch.sigmoid(self.fc2(x))
        return x



class GNNSimple(torch.nn.Module):
    def __init__(self,inp_dim, hid_dim, out_dim):
        super().__init__()
        self.conv1 = GraphConv(inp_dim,  hid_dim)
        self.conv2 = GraphConv(hid_dim, out_dim)

    def forward(self, x, edge_index, edge_attr):
        x = F.elu(self.conv1(x, edge_index, edge_attr))
        x = self.conv2(x, edge_index, edge_attr)
        return x


class GNNAPPNP(torch.nn.Module):
    def __init__(self, inp_dim, hid_dim, out_dim, appnp_k, appnp_alpha):
        super().__init__()
        self.lin1 = torch.nn.Linear(inp_dim, hid_dim)
        self.dp1 = torch.nn.Dropout(p=0.5)
        self.lin2 = torch.nn.Linear(hid_dim, out_dim)
        self.conv1 = APPNP(K=appnp_k, alpha=appnp_alpha)
        self.conv2 = APPNP(K=appnp_k, alpha=appnp_alpha)
    def forward(self, x, edge_index, edge_attr):
        x = self.dp1(F.elu(self.lin1(x)))
        x = (self.lin2(x))
        x = self.conv1(x, edge_index, edge_attr)
        x = self.conv2(x, edge_index, edge_attr)
        return x



class LaplacianRegulaizer:
    """Identical to LaplaceDenoiser class but adapted to 1-sized datasets."""

    def __init__(self, lr, MAX_ITER, train_mask, task='regression', num_classes=None):
        self.lr = lr
        self.MAX_ITER = MAX_ITER
        self.train_mask = train_mask
        self.lossf = nn.MSELoss()
        self.task = task
        if task == 'classification':
            self.lossf = nn.CrossEntropyLoss()
            self.num_classes = num_classes

    def get_loss(self, edge_index, edge_attr, y, y_hat, lamb="inv_adj_sum", extra_lamb=1.):
        npoints = y.shape[0]
        data_fidelity = self.lossf(y_hat[self.train_mask], y[self.train_mask])
        reg_term = (edge_attr*torch.square(y_hat[edge_index[0]] - y_hat[edge_index[1]]).T).T
        reg_term = reg_term.sum()
        if lamb == "inv_npoints_sqr":
            lamb = .1/npoints**2
        elif lamb == "inv_adj_sum":
            lamb = 1./edge_index.shape[-1]
        loss = data_fidelity + extra_lamb*lamb*reg_term
        return loss, data_fidelity

    def optimize_yhat(self, data, edge_attr):
        if self.task == 'classification':
            y_hat = torch.rand((data.y.shape[0], self.num_classes), requires_grad=True, device=data.edge_index.device)
        elif self.task == 'regression':
            y_hat = torch.rand(data.y.size(), requires_grad=True, device=data.edge_index.device)

        inner_opt = torch.optim.Adam([y_hat], lr=self.lr, weight_decay=1e-5)
        inner_opt = higher.get_diff_optim(inner_opt, [y_hat], track_higher_grads=True)
        best_test_acc,  best_val_acc = 0., 0.
        for i in range(self.MAX_ITER):
            inner_loss, data_fidelity = self.get_loss(data.edge_index, edge_attr, data.y, y_hat)
            y_hat, = inner_opt.step(inner_loss, params=[y_hat])
            accs = []
            for _, mask in data('train_in_mask', 'train_out_mask', 'val_mask', 'test_mask'):
                pred = y_hat[mask].max(1)[1]
                acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
                accs.append(acc)
            if accs[2] > best_val_acc:
                best_val_acc = accs[2]
                best_test_acc = accs[3]

            if i == 0 or (i+1) % 100 == 0:
                print(f'Inner iteration: {i+1:03d}, Inner: {accs[0]:.4f}, Outer: {accs[1]:.4f}, '
                      f'Validation: {accs[2]: .4f}, Test: {accs[3]: .4f},  InLoss: {inner_loss.item():.6f}'
                      f' Data fidelity: {data_fidelity.item():.6f}')
        return y_hat, inner_loss, best_test_acc,  best_val_acc


class Alearner(nn.Module):
    def __init__(self, num_edges, init=None):
        super().__init__()
        self.edge_attr = nn.Parameter(torch.rand(num_edges))
        if init is not None:
            self.edge_attr.data = init

    def forward(self, x, edge_index):
        return torch.sigmoid(self.edge_attr)
