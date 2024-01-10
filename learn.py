import argparse
import os
import higher
import torch
from modules.datasets import fetch_dataset, make_graph_connected
from modules.models import LaplacianRegulaizer, GNNSimple, GNNAPPNP,  Alearner, MlpG2g
from modules.myfuns import delete_files_in_directory, plot_hypergradient_GNNs_case
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def get_args():
    # dataset and models
    parser = argparse.ArgumentParser(description='Parser for the bilevel framework')
    parser.add_argument('--dataset', default="Cora", type=str, help='options: Cora, CiteSeer, PubMed, Cheaters, Synthetic1HighFrequency, Synthetic1LowFrequency')
    parser.add_argument('--inner_model', default="GNN_simple",  type=str, help='GNN_simple, APPNP or Laplacian')
    parser.add_argument('--method', default="BO", type=str, help='The version of the bilevel framework when optimizing the graph, or A_obs when using the observed one. Options: BO, BO+G2G, BO+regularization, BO+A_obs^6, A_obs')
    parser.add_argument('--plot', default = False, type = bool, help = 'True when you only want to plot the hypergradient at iteration 9')
    parser.add_argument('--save', default = True, type = bool, help = 'whether to save the different losses and the best model in the output directory or not')
    # hyperparameters
    parser.add_argument('--g2g_hid_dim', type=int, help='The hidden dimension of the G2G model if adopted')
    parser.add_argument('--g2g_num_layers', type=int, help='The number of layers of the G2G model if adopted')
    parser.add_argument('--gnn_hid_dim', type=int, help='The hidden dimension of the GNN model if adopted')
    parser.add_argument('--graph_reg_mag', type=float, help='The regularization magnitude parameter')
    parser.add_argument('--MAX_IN_ITER', type=int, help='The number of inner iterations')
    parser.add_argument('--MAX_OUT_ITER', type=int, help='The number of outer iterations')
    parser.add_argument('--lr_out', type=float, help='The outer learning rate')
    parser.add_argument('--lr_in', type=float, help='The inner learning rate')
    parser.add_argument('--output_dir', type=str, help='The output directory')
    parser.add_argument('--appnp_k', type=int, help='The number of iterations of the APPNP model if adopted')
    parser.add_argument('--appnp_alpha', type=float, help='The alpha parameter of the APPNP model if adopted')
    args = parser.parse_args()

    # Check the validity of the arguments
    if args.dataset in ['Cora', 'CiteSeer', 'PubMed'] and args.method == 'BO+A_obs^6':
        msg = 'This code cannot handle the memory demand of learning from '+ args.dataset + " with " + args.method +". To make it do so, change the method from BO+A_obs^6 to BO, BO+G2G, or BO+regularization."
        raise Exception(msg)

    if args.output_dir is None:
        args.output_dir = os.path.join(os.path.dirname(__file__), "outputs")
    delete_files_in_directory(args.output_dir)

    # Set the hyperparameters based on the dataset and the method if not specified by the user
    if args.dataset == 'Cheaters':
        args.g2g_hid_dim = 16
        args.g2g_num_layers = 2
        args.gnn_hid_dim = 8
        args.MAX_IN_ITER = 200
        args.MAX_OUT_ITER = 150
        if args.method == 'BO+G2G':
            args.lr_out = 0.001
    elif args.dataset in ['Synthetic1HighFrequency', 'Synthetic1LowFrequency']:
        args.MAX_IN_ITER = 500
        args.MAX_OUT_ITER = 150
        args.lr_in = 10.
        args.lr_out = 0.1
    elif args.dataset in ['Cora', 'CiteSeer', 'PubMed']:
        args.g2g_hid_dim = 32
        args.g2g_num_layers = 1
        args.gnn_hid_dim = 16
        args.MAX_IN_ITER = 500
        args.MAX_OUT_ITER = 300
        args.lr_in = 0.01
        args.lr_out = 0.01
        if args.inner_model == 'Laplacian':
            args.MAX_IN_ITER = 500
        else:
            args.MAX_IN_ITER = 100
    if args.graph_reg_mag is None:
        args.graph_reg_mag = 1.
    if args.appnp_k is None:
        args.appnp_k=  10
    if args.appnp_alpha is None:
        args.appnp_alpha= 0.1
    return args

def train_gnns(args, data, inner_model, edge_attr):
    # define the inner optimizer
    inner_optim = torch.optim.Adam(inner_model.parameters(), lr=args.lr_in, weight_decay=5e-4)
    inner_optim.zero_grad()
    criterion = torch.nn.CrossEntropyLoss() if args.dataset in ['Cora', 'CiteSeer', 'PubMed', 'Cheaters'] else torch.nn.MSELoss()
    best_test_acc,  best_val_acc = 0., 0.
    # start training
    with higher.innerloop_ctx(inner_model, inner_optim) as (fmodel, diffopt):
        for itrin in range(args.MAX_IN_ITER):
            # compute predicted classes
            out = fmodel(data.x, data.edge_index, edge_attr)
            inner_loss = criterion(out[data.train_in_mask], data.y[data.train_in_mask])
            diffopt.step(inner_loss)
            accs = []
            for _, mask in data('train_in_mask', 'train_out_mask', 'val_mask', 'test_mask'):
                pred = out[mask].max(1)[1]
                acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
                accs.append(acc)
            if accs[2] > best_val_acc:
                best_val_acc = accs[2]
                best_test_acc = accs[3]

            if itrin % 20 == 0:
                print(f'Inner iteration: {itrin:03d}, Inner: {accs[0]:.4f}, Outer: {accs[1]:.4f}, '
                      f'Validation: {accs[2]: .4f}, Test: {accs[3]: .4f},  InLoss: {inner_loss.item():.4f}')
        # Next, compute hypergradient to do an outer iteration.
        # compute predicted classes:
        out = fmodel(data.x, data.edge_index, edge_attr)
    return out, inner_loss, best_val_acc, best_test_acc

def train(args, data, graph_gener, inner_model):
    criterion = torch.nn.CrossEntropyLoss() if args.dataset in ['Cora', 'CiteSeer', 'PubMed', 'Cheaters'] else torch.nn.MSELoss()
    outer_optim = torch.optim.Adam(graph_gener.parameters(), lr=args.lr_out)

    best_val_acc, best_test_acc = 0., 0.
    tr_in_loss, tr_out_loss = torch.zeros(args.MAX_OUT_ITER), torch.zeros(args.MAX_OUT_ITER)
    tr_in_acc, tr_out_acc = torch.zeros(args.MAX_OUT_ITER), torch.zeros(args.MAX_OUT_ITER)
    val_acc, test_acc = torch.zeros(args.MAX_OUT_ITER), torch.zeros(args.MAX_OUT_ITER)
    for itrout in range(args.MAX_OUT_ITER):
        print ('-----------------------------------\n', 'Outer iteration', itrout, '\n-----------------------------------')
        graph_gener.train()
        # compute the output edge weights
        edge_attr = graph_gener(data.x, data.edge_index)
        # optimize the inner model
        if args.inner_model in ['GNN_simple', 'APPNP']:
            out, tr_in_loss[itrout], val_acc_iteration, test_acc_iteration = train_gnns(args, data, inner_model, edge_attr)
        elif args.inner_model == 'Laplacian':
            out, tr_in_loss[itrout] = inner_model.optimize_yhat(args, data, inner_model, graph_gener)
        # check if a better val_acc is obtained
        if val_acc_iteration > best_val_acc:
            best_val_acc = val_acc_iteration
            best_test_acc = test_acc_iteration
            if args.save is True:
                torch.save(graph_gener, os.path.join(args.output_dir, "graph_gener.pth"))
        # for saving purposes compute the accuracies of the trained inner model
        accs = []
        for _, mask in data('train_in_mask', 'train_out_mask', 'val_mask', 'test_mask'):
            pred = out[mask].max(1)[1]
            acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
            accs.append(acc)
        tr_in_acc[itrout], tr_out_acc[itrout], val_acc[itrout], test_acc[itrout] = accs
        # Compute the outer loss
        loss = criterion(out[data.train_out_mask], data.y[data.train_out_mask])
        print(f"Outer loss:  {loss.item():.6f}, Best valid acc untill now: {best_val_acc:.4f}, Test acc untill now: {best_test_acc:.4f}")
        tr_out_loss[itrout] = loss.item()
        # regularizing the graph if needed
        if args.method == 'BO+regularization':
            adj_dense = torch.zeros((len(data.y), len(data.y)), device=device)
            adj_dense[data.edge_index[0], data.edge_index[1]] = edge_attr
            graph_reg = - args.graph_reg_mag * torch.log(adj_dense.sum(dim=1)).mean()
            loss = loss + graph_reg
        outer_optim.zero_grad()
        loss.backward()
        # saving the hypergradient every 3 outer iteration for plotting for example.
        if itrout % 3 == 0 and args.method != 'BO+G2G':
            gradstr = 'GradScreenshot_Iter'+str(itrout)+".pt"
            adjstr = 'AdjScreenshot_Iter'+str(itrout)+".pt"
            torch.save(graph_gener.edge_attr.grad.clone().detach(), os.path.join(args.output_dir, gradstr))
            torch.save(edge_attr.clone().detach(), os.path.join(args.output_dir, adjstr))
            # print("non_zeros are", (grad_to_save != 0).sum())
        outer_optim.step()

        if args.save is True:
            torch.save(tr_in_loss[:itrout], os.path.join(args.output_dir, "tr_in_loss.pt"))
            torch.save(tr_out_loss[:itrout], os.path.join(args.output_dir, "tr_out_loss.pt"))
            torch.save(tr_in_acc[:itrout], os.path.join(args.output_dir, "tr_in_acc.pt"))
            torch.save(tr_out_acc[:itrout], os.path.join(args.output_dir, "tr_out_acc.pt"))
            torch.save(val_acc[:itrout], os.path.join(args.output_dir, "val_acc.pt"))
            torch.save(test_acc[:itrout], os.path.join(args.output_dir, "test_acc.pt"))
        if args.plot is True and itrout ==9:
            break
    print(f"Best validation accuracy: {best_val_acc:.4f}, Test accuracy: {best_test_acc:.4f}")

if __name__ == "__main__":
    # collect arguments
    args = get_args()
    # Fetch and save the dataset
    data = fetch_dataset(args.dataset)
    if args.inner_model =='Laplacian' and args.plot is True:
        print('#edges before making the graph connected: ', data.edge_index.shape)
        data = make_graph_connected(data)
        print('#edges after making the graph connected: ', data.edge_index.shape)
    if args.save is True:
        torch.save(data, os.path.join(args.output_dir, "dataset.pt"))

    # Migrate to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)

    # set the outer model: G2G or directly learning the adjacency matrix A.
    if args.method == 'BO+G2G':
        graph_gener = MlpG2g(data.x.shape[1], args.g2g_hid_dim, nlayers=args.g2g_num_layers).to(device)
    else:
        graph_gener = Alearner(data.edge_index.shape[1]).to(device)
    if args.inner_model =="GNN_simple":
        inner_model = GNNSimple(data.num_features, args.gnn_hid_dim, data.num_classes).to(device)
    elif args.inner_model =="APPNP":
        inner_model = GNNAPPNP(data.num_features, args.gnn_hid_dim, data.num_classes, appnp_k=args.appnp_k , appnp_alpha=args.appnp_alpha).to(device)
    elif args.inner_model =="Laplacian":
        inner_model = LaplacianRegulaizer(lr=args.lr_in, MAX_ITER=args.MAX_IN_ITER, train_mask=data.train_in_mask, task='regression' if args.dataset in ['Synthetic1HighFrequency', 'Synthetic1LowFrequency'] else 'classification', num_classes=data.num_classes)

    train(args, data, graph_gener, inner_model)
    # if plotting the hypergradient at iteration 9 is the goal of running the goal:
    if args.plot is True:
        plot_hypergradient_GNNs_case(args,9)
