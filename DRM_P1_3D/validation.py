import numpy as np
from torch.autograd import Variable
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import pickle as pkl
import json
import g_tr as gt
import tools

dtype = torch.float32


resolution = 50
val_x1=np.arange(0,1,1/resolution).reshape(-1,1)
val_x2=np.arange(0,1,1/resolution).reshape(-1,1)
val_x3=np.arange(0,1,1/resolution).reshape(-1,1)
t_vx1 = Variable(torch.from_numpy(val_x1)).type(dtype)
t_vx2 = Variable(torch.from_numpy(val_x2)).type(dtype)
t_vx3 = Variable(torch.from_numpy(val_x3)).type(dtype)



#Generate grids to output graph
val_ms_x1, val_ms_x2, val_ms_x3 = np.meshgrid(val_x1, val_x2,val_x3)
plot_val_x1 = np.ravel(val_ms_x1).reshape(-1,1)
plot_val_x2 = np.ravel(val_ms_x2).reshape(-1,1)
plot_val_x3 = np.ravel(val_ms_x3).reshape(-1,1)

t_val_vx1,t_val_vx2,t_val_vx3 = tools.from_numpy_to_tensor([plot_val_x1,plot_val_x2,plot_val_x3],[False,False,False],dtype=dtype)

y_gt,f = gt.data_gen_interior(np.concatenate([plot_val_x1,plot_val_x2,plot_val_x3],axis=1))

t_ygt = tools.from_numpy_to_tensor([y_gt],[False,False,False,False],dtype=dtype)


# def plot_2D(net,path):
#     data = net(t_val_vx1,t_val_vx2).detach().numpy().reshape([resolution,resolution])
#     fig = plt.figure()
#     ax = fig.add_subplot(projection='3d')
#     surf = ax.plot_surface(val_ms_x1,val_ms_x2,data, cmap=cm.coolwarm,linewidth=0, antialiased=False)
#     ax.zaxis.set_major_locator(LinearLocator(10))
#     ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
#     fig.colorbar(surf, shrink=0.5, aspect=5)

#     plt.savefig(path)
#     plt.close()


def compute_error(net):
    # Evaluate the network
    with torch.no_grad():
        pred = net(t_val_vx1, t_val_vx2,t_val_vx3).cpu().numpy().reshape([resolution, resolution,resolution])

    # Ground truth
    ms_ugt = y_gt.reshape([resolution, resolution,resolution])

    # Flatten
    ms_ugt_flat = ms_ugt.flatten()
    pred_flat = pred.flatten()

    # L² error and relative error
    l2_error = np.sqrt(np.mean((ms_ugt_flat - pred_flat)**2))
    l2_relative_error = l2_error / np.sqrt(np.mean(ms_ugt_flat**2))

    return l2_error, l2_relative_error    