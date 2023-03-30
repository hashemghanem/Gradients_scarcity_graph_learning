import torch
import torch.nn as nn
import higher
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GraphConv
from torch_geometric.utils import dense_to_sparse
from .datasets import all_pairs_of_nodes_from_pointset


class embed_pointset_perm_inv(nn.Module):
    def __init__(self, ncoordinates, hidden_dim=16, output_dim=16):
        super().__init__()
        self.fc1 = nn.Conv1d(ncoordinates, hidden_dim, 1)
        self.fc2 = nn.Conv1d(hidden_dim, 2*hidden_dim, 1)
        self.fc3 = nn.Linear(2*hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """x of shape nsamples, ncoordinates, npoints."""
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = x.max(dim=2).values
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class simple_set2graph(nn.Module):
    def __init__(self, inp_dim, outer_hidden_dim,
                 pntst_embed_hiddim=16,  pntst_embed_dim=4):
        super().__init__()
        # self.embed_pointset = embed_pointset_perm_inv(inp_dim,
        #                                               pntst_embed_hiddim,
        #                                               pntst_embed_dim)
        # put inp_dim + pntst_embed_dim
        self.nodes_similarity = simple_model_learn_similarity(
            inp_dim, outer_hidden_dim)

    def forward(self, x):
        batch_size = x.shape[0]
        npoints = x.shape[1]
        npairs = int((npoints*(npoints-1))/2)+npoints

        # global_embed = self.embed_pointset(x.transpose(1, 2))[:, None, :]
        # global_embed = global_embed.expand(-1, npairs, -1)
        # global_embed = global_embed.reshape(batch_size * npairs, -1)
        print("hi before is similar")
        is_similar = all_pairs_of_nodes_from_pointset(
            x, aggregation='abs_dif')
        print("hi after is similar")
        # is_similar = torch.cat((is_similar, global_embed), dim=1)
        is_similar = self.nodes_similarity(is_similar)
        is_similar = is_similar.view(batch_size, npairs)

        triu = torch.triu_indices(npoints, npoints)
        adj_est = torch.zeros(batch_size, npoints, npoints).to(x.device)
        adj_est[:, triu[0], triu[1]] = is_similar
        adj_est = adj_est + adj_est.transpose(1, 2)
        diag = [i for i in range(npoints)]
        adj_est[:, diag, diag] = 1.
        return adj_est


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


class MlpG2g(nn.Module):
    def __init__(self, inp_dim, hid_dim, nlayers=1, pnt_inp_dim=2,
                 pnt_hid_dim=32, pnt_out_dim=2, pnt_nlayers=1):
        super().__init__()
        self.embed_pnt = MlpPointWise(
            pnt_inp_dim, pnt_hid_dim, pnt_out_dim, pnt_nlayers)
        self.nlayers = nlayers
        self.fc1 = nn.Linear(inp_dim, hid_dim)
        self.layers = nn.ModuleList(
            [nn.Linear(hid_dim, hid_dim) for _ in range(nlayers-1)])
        self.bns = nn.ModuleList(
            [nn.BatchNorm1d(hid_dim) for _ in range(nlayers-1)])
        self.fc2 = nn.Linear(hid_dim, 1)

    def forward(self, x, adj=None):
        batch_size = x.shape[0]
        npoints = x.shape[1]
        # Compute the point-wise embedding.
        x = self.embed_pnt(x.view(-1, x.shape[-1]))
        x = x.view(batch_size, npoints, -1)
        # Prepare input for the G2G model.
        dist = (x[:, None, :, :]).expand(-1, npoints, -1, -1)
        dist = ((dist - dist.transpose(1, 2))**2)
        if adj is not None:
            dist = torch.cat((dist, adj[:, :, :, None]), dim=3)
        x = dist.view(-1, dist.shape[-1])
        x = F.relu(self.fc1(x))
        for i in range(self.nlayers-1):
            x = F.relu(self.layers[i](x))
            # x = self.bns[i](x)
        x = torch.sigmoid(self.fc2(x))
        return x.view(batch_size, npoints, npoints)


class MlpG2gCoraDense(nn.Module):
    def __init__(self, inp_dim, hid_dim, out_dim=1):
        super().__init__()
        self.embed_pnt = MlpPointWise(inp_dim, hid_dim, hid_dim, 1)
        self.fc1 = torch.nn.Linear(hid_dim, hid_dim)
        self.fc2 = torch.nn.Linear(hid_dim, out_dim)

    def forward(self, x, adj=None):
        npoints = x.shape[0]
        # Compute the point-wise embedding.
        x = self.embed_pnt(x)
        # Prepare input for the G2G model.
        dist = (x[None, :, :]).expand(npoints, -1, -1)
        dist = ((dist - dist.transpose(0, 1))**2)
        if adj is not None:
            dist = torch.cat((dist, adj[:, :, None]), dim=2)
        x = dist.view(-1, dist.shape[-1])
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.fc1(x))
        x = F.dropout(x, p=0.6, training=self.training)
        x = torch.sigmoid(self.fc2(x))
        return x.view(npoints, npoints)


class MlpG2gCora_sigmoid(torch.nn.Module):
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


class MlpG2gCora(torch.nn.Module):
    def __init__(self, inp_dim, hid_dim, out_dim=1):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(inp_dim)
        self.fc1 = torch.nn.Linear(inp_dim, hid_dim)
        self.fc2 = torch.nn.Linear(hid_dim, hid_dim)
        self.fc3 = torch.nn.Linear(hid_dim, out_dim)

    def get_g2g_input(self, x, edge_index):
        feat1 = x[edge_index[0]]
        feat2 = x[edge_index[1]]
        feats = (feat1-feat2)**2
        return feats

    def forward(self, x, edge_index):
        x = self.get_g2g_input(x, edge_index)
        x = self.bn1(x)
        # print(x[:10])
        x = F.elu(self.fc1(x))
        # print(x[:10, :2])
        x = F.elu(self.fc2(x))
        # print(x[:10, :2])
        x = torch.relu(self.fc3(x)).squeeze()
        # print(x[:10])
        # don't forget to change GAM  BCE loss if you use sigmoid
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


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__(inp_dim, hid_dim, out_dim)
        self.conv1 = GraphConv(inp_dim,  hid_dim)
        self.conv2 = GraphConv(hid_dim, out_dim)

    def forward(self, x, edge_index, edge_attr):
        x = torch.relu(self.conv1(x, edge_index, edge_attr))
        x = self.conv2(x, edge_index, edge_attr)
        return x


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GraphConv(data.num_features,  GCN_hid_dim)
        self.conv2 = GraphConv(GCN_hid_dim, data.num_classes)

    def forward(self, x, edge_index, edge_attr):
        x = torch.relu(self.conv1(x, edge_index, edge_attr))
        x = self.conv2(x, edge_index, edge_attr)
        return x


class LaplaceDenoiser_one_pointset:
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

    def get_loss(self, edge_index, edge_attr, y, y_hat, lamb="inv_npoints_sqr", extra_lamb=1.):
        npoints = y.shape[0]
        data_fidelity = self.lossf(y_hat[self.train_mask], y[self.train_mask])
        # the transpose below is for broadcastability.
        reg_term = (
            edge_attr*torch.square(y_hat[edge_index[0]] - y_hat[edge_index[1]]).T).T
        reg_term = reg_term.sum()
        if lamb == "inv_npoints_sqr":
            lamb = .1/npoints**2
        elif lamb == "inv_adj_sum":
            lamb = 1./edge_index.shape[-1]
        loss = data_fidelity + extra_lamb*lamb*reg_term
        return loss, data_fidelity

    def optimize_yhat(self, edge_index, edge_attr, p, y, y_hat=None,
                      lamb="inv_npoints_sqr", unroll_optim=True,
                      extra_lamb=1.):
        if y_hat is None and self.task == 'classification':
            y_hat = torch.rand((y.shape[0], self.num_classes), requires_grad=True,
                               device=edge_index.device)
        elif y_hat is None and self.task == 'regression':
            y_hat = torch.rand(y.size(), requires_grad=True,
                               device=edge_index.device)

        inner_opt = torch.optim.Adam([y_hat], lr=self.lr)
        for i in range(self.MAX_ITER-self.MAX_ITER):
            loss, data_fidelity = self.get_loss(
                edge_index, edge_attr, y, y_hat, lamb, extra_lamb)
            inner_opt.zero_grad()
            loss.backward()
            inner_opt.step()
            if (i+1) % 100 == 0:
                print("Iter {:<30}, inner loss= {:<30}, data fidelity= {:<20}".format(
                    i+1, loss.item(), data_fidelity))
        inner_opt = higher.get_diff_optim(
            inner_opt, [y_hat], track_higher_grads=True)
        for i in range(self.MAX_ITER-self.MAX_ITER, self.MAX_ITER):
            loss, data_fidelity = self.get_loss(
                edge_index, edge_attr, y, y_hat, lamb, extra_lamb)
            y_hat, = inner_opt.step(loss, params=[y_hat])
            # if i == self.MAX_ITER-98:
            #     inner_opt = higher.get_diff_optim(
            #         torch.optim.Adam([y_hat], lr=self.lr*batch_sz),
            #         [y_hat], track_higher_grads=True)
            if i == 0 or (i+1) % 100 == 0:
                print("Iter {:<30}, inner loss= {:<30}, data fidelity= {:<20}".format(
                    i+1, loss.item(), data_fidelity))
        return y_hat, loss


class LaplaceDenoiser:
    def __init__(self, lr, MAX_ITER, train_mask):
        self.lr = lr
        self.MAX_ITER = MAX_ITER
        self.train_mask = train_mask

    def get_loss(self, adj, y, y_hat, lamb="inv_npoints_sqr", extra_lamb=1.):
        npoints = adj.shape[-1]
        batch_sz = adj.shape[0]
        mse = nn.MSELoss()
        data_fidelity = mse(y_hat[:, self.train_mask], y[:, self.train_mask])
        reg_term = y_hat[:, :, None].expand(-1, -1, npoints)
        reg_term = torch.abs(reg_term - reg_term.transpose(1, 2))
        reg_term = torch.mul(adj, torch.pow(reg_term, 2)).sum(dim=2).sum(dim=1)
        # degs = adj.sum(dim=2)
        # deg_mat = torch.zeros_like(adj)
        # for i in range(batch_sz):
        #     deg_mat[i, torch.arange(npoints), torch.arange(npoints)] = degs[i]
        # lap = deg_mat-adj
        # reg_term = torch.bmm(y_hat[:, None, :], torch.bmm(
        #     lap, y_hat[:, :, None])).flatten()
        if (reg_term).mean() < 0:
            print("here ")
        if lamb == "inv_npoints_sqr":
            lamb = .1/npoints**2 * torch.ones_like(reg_term)
        elif lamb == "inv_adj_sum":
            diag = torch.tensor(
                [i for i in range(npoints)], device=adj.device)
            diag = adj[:, diag, diag].sum(dim=1)
            lamb = 1./(adj.sum(dim=2).sum(dim=1)-diag)
        loss = data_fidelity + (extra_lamb*lamb*reg_term).mean()
        if (reg_term).mean() < 0:
            print("here ")
        return loss, data_fidelity

    def optimize_yhat(self, adj, p, y, y_hat=None,
                      lamb="inv_npoints_sqr", unroll_optim=True,
                      extra_lamb=1.):
        batch_sz = adj.shape[0]
        if y_hat is None:
            y_hat = torch.rand(
                (adj.shape[0], adj.shape[1]), requires_grad=True, device=adj.device)

        inner_opt = torch.optim.Adam([y_hat], lr=self.lr*batch_sz)
        adj_copy = adj.detach().clone()
        for i in range(self.MAX_ITER-10):
            loss, data_fidelity = self.get_loss(
                adj_copy, y, y_hat, lamb, extra_lamb)

            inner_opt.zero_grad()
            loss.backward()
            inner_opt.step()
            if (i+1) % 50 == 0:
                print("Iter {:<30}, inner loss= {:<30}, data fidelity= {:<20}".format(
                    i+1, loss.item(), data_fidelity))
        inner_opt = higher.get_diff_optim(
            inner_opt, [y_hat], track_higher_grads=True)
        for i in range(self.MAX_ITER-10, self.MAX_ITER):
            loss, data_fidelity = self.get_loss(
                adj, y, y_hat, lamb, extra_lamb)
            y_hat, = inner_opt.step(loss, params=[y_hat])
            # if i == self.MAX_ITER-98:
            #     inner_opt = higher.get_diff_optim(
            #         torch.optim.Adam([y_hat], lr=self.lr*batch_sz),
            #         [y_hat], track_higher_grads=True)
            if i == 0 or (i+1) % 4 == 0:
                print("Iter {:<30}, inner loss= {:<30}, data fidelity= {:<20}".format(
                    i+1, loss.item(), data_fidelity))
        return y_hat, loss
