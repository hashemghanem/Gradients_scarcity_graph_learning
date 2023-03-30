# %%
import torch
import os.path as osp
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.colors import SymLogNorm, LogNorm
from matplotlib.ticker import SymmetricalLogLocator
import matplotlib as mpl
import torch_geometric.transforms as T
from cycler import cycler
import matplotlib.pylab as pylab
from torch_geometric.datasets import Planetoid


fontsize = 20
plt.style.use('default')
plt.rcParams.update({'font.size': fontsize})
plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
# plt.rc('text', usetex=True)
matplotlib.rc('text', usetex=True)
matplotlib.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]
line_width, marker_size = 2, 0


# Colorblind friendly color cycles
ccycles = ['tableau-colorblind10', 'seaborn-colorblind']

# # markers in case needed
mrk = ['s', 'v', 'o', 'x', '3', 'p', '|']

# making the labels/ticks' font larger
params = {'legend.fontsize': fontsize,
          'axes.labelsize': fontsize,
          'xtick.labelsize': 22,
          'ytick.labelsize': 22,
          "axes.titlesize": 22,
          "axes.labelsize": 24}
pylab.rcParams.update(params)

# Set the color style (cycle + map) you want.
plt.style.use(ccycles[0])

min_grad = 1e10
max_grad = -1e10
min_adj = 1e10
max_adj = -1e10

################################################################
################################################################
################################################################


MAX_OUT_ITER = 100
a = torch.arange(0, MAX_OUT_ITER, 3).tolist()
for i in a:
    gradstr = 'GradScreenshot_Iter'+str(i)+".pt"
    adjstr = 'AdjScreenshot_Iter'+str(i)+".pt"
    theogradstr = 'GradTheoretical_Iter'+str(i)+".pt"
    grad = torch.load('saved_data/'+gradstr, map_location=torch.device('cpu'))
    adj = torch.load('saved_data/'+adjstr, map_location=torch.device('cpu'))
    # grad_L = torch.load('saved_data/'+theogradstr,
    # map_location=torch.device('cpu'))
    min_grad = min(min_grad, grad.min().item())
    # min_grad = min(min_grad, grad_L.min().item())
    max_grad = max(max_grad, grad.max().item())
    # max_grad = max(max_grad, grad_L.max().item())
    # print(min_grad, max_grad)
    min_adj = min(min_adj, adj.min().item())
    max_adj = max(max_adj, adj.max().item())


for i in a:
    gradstr = 'GradScreenshot_Iter'+str(i)+".pt"
    adjstr = 'AdjScreenshot_Iter'+str(i)+".pt"
    theogradstr = 'GradTheoretical_Iter'+str(i)+".pt"
    grad = torch.load('saved_data/'+gradstr, map_location=torch.device('cpu'))
    adj = torch.load('saved_data/'+adjstr, map_location=torch.device('cpu'))
    # grad_L = torch.load('saved_data/'+theogradstr,
    #                     map_location=torch.device('cpu')).detach()
    # fig, axs = plt.subplots(constrained_layout=True)
    # im_grad = axs.imshow(grad, norm=SymLogNorm(
    #     linthresh=1e-6, vmin=min_grad, vmax=max_grad))
    # cb = fig.colorbar(im_grad)
    # # tick_locator = SymmetricalLogLocator(base=1000, linthresh=1e-9)
    # # cb.locator = tick_locator
    # # cb.update_ticks()

    # plt.savefig('figs/'+'GradScreenshot_Iter' +
    #             str(i)+'.pdf', bbox_inches='tight')
    # plt.show()
    # fig, axs = plt.subplots(constrained_layout=True)
    # im_grad = axs.imshow((grad_L+grad_L)/2, norm=SymLogNorm(
    # linthresh=1e-8, vmin=min_grad, vmax=max_grad))
    # # im_grad = axs.imshow(grad)
    # axs.set_title('Gradient theoretical at outer iter'+str(i))
    # # axs.set_yscale('log')
    # fig.colorbar(im_grad)
    # plt.savefig('figs/'+'GradTheoretical' +
    #             str(i)+'.pdf', bbox_inches='tight')
    # plt.show()

    fig, axs = plt.subplots(constrained_layout=True)
    im_grad = axs.imshow(adj)
    fig.colorbar(im_grad)
    plt.savefig('figs/'+'AdjScreenshot_Iter' +
                str(i)+'.pdf', bbox_inches='tight')
    plt.show()


# norm0 = torch.zeros(len(a))
# threshold = .1
# for ind, i in enumerate(a):
#     gradstr = 'GradScreenshot_Iter'+str(i)+".pt"
#     adjstr = 'AdjScreenshot_Iter'+str(i)+".pt"
#     theogradstr = 'GradTheoretical_Iter'+str(i)+".pt"
#     grad = torch.load('saved_data/'+gradstr, map_location=torch.device('cpu'))
#     adj = torch.load('saved_data/'+adjstr, map_location=torch.device('cpu'))
#     # grad_L = torch.load('saved_data/'+theogradstr,
#     #                     map_location=torch.device('cpu')).detach()
#     # norm0[ind] = (adj.abs() > threshold).sum().item()
#     norm0[ind] = (grad.abs() > (grad.abs().max()/100)).sum().item()

# fig, axs = plt.subplots(constrained_layout=True)
# # Plotting inner losses
# axs.plot(a[1:], norm0[1:])
# # im_grad = axs.imshow(grad, vmin=min_grad, vmax=max_grad)
# axs.set_title('# of non zero elements')
# # plt.yscale('log')
# # axs.set_yscale('log')
# plt.savefig('figs/'+'nrom0' + '.pdf', bbox_inches='tight')
# plt.show()

val_acc = torch.load(os.path.join('saved_data/', "val_acc.pt")).detach()
test_acc = torch.load(os.path.join('saved_data/', "test_acc.pt")).detach()
tr_in_acc = torch.load(os.path.join('saved_data/', "tr_in_acc.pt")).detach()
tr_out_acc = torch.load(os.path.join('saved_data/', "tr_out_acc.pt")).detach()
tr_out_loss = torch.load(os.path.join('saved_data/', "tr_out_loss.pt"))
tr_in_loss = torch.load(os.path.join('saved_data/', "tr_in_loss.pt"))
plt.plot(tr_in_acc, label='in acc')
plt.plot(tr_out_acc, label='out acc')
plt.plot(val_acc, label='val acc')
plt.plot(test_acc, label='test acc')
plt.legend()
plt.savefig('figs/accuraies.pdf', bbox_inches='tight')
plt.show()
plt.plot(tr_in_loss, label='in loss')
plt.plot(tr_out_loss, label='out loss')
# plt.yscale('log')
plt.legend()
plt.savefig('figs/losses.pdf', bbox_inches='tight')
plt.show()

# %%
