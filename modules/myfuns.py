import os
import torch
import networkx as nx
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.pylab as pylab
from matplotlib.ticker import SymmetricalLogLocator


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
