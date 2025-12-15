import numpy as np
import torch
import torch.optim as opt
import matplotlib.pyplot as plt
import torch.utils.data.dataset as Dataset
import torch.utils.data.dataloader as Dataloader
from torch.autograd import Variable
import pickle as pkl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import model,pde,data,tools,g_tr,validation
import time
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


torch.set_default_dtype(torch.float32)

y = model.NN()
y.apply(model.init_weights)

dataname = '10000pts'
name = 'results/'

bw = 30.0

if not os.path.exists(name):
    os.makedirs(name)

if not os.path.exists(name+"y_plot/"):
    os.makedirs(name+"y_plot/")


params = list(y.parameters())



with open("dataset/"+dataname,'rb') as pfile:
    int_col = pkl.load(pfile)
    bdry_col = pkl.load(pfile)

print(int_col.shape,bdry_col.shape)

intx1,intx2 = np.split(int_col,2,axis=1)
bdx1,bdx2 = np.split(bdry_col,2,axis=1)

tintx1,tintx2,tbdx1,tbdx2 = tools.from_numpy_to_tensor([intx1,intx2,bdx1,bdx2],[True,True,False,False,],dtype=torch.float32)


with open("dataset/gt_on_{}".format(dataname),'rb') as pfile:
    y_gt = pkl.load(pfile)
    f_np = pkl.load(pfile)
    bdry_np = pkl.load(pfile)

f,bdrydat,ygt = tools.from_numpy_to_tensor([f_np,bdry_np,y_gt],[False,False,False],dtype=torch.float32)


optimizer = opt.LBFGS(params,line_search_fn='strong_wolfe',max_iter=600,tolerance_grad=1e-20,tolerance_change=1e-20)
def closure():
    optimizer.zero_grad()
    loss,pres,bres = pde.pdeloss(y,tintx1,tintx2,f,tbdx1,tbdx2,bdrydat,bw)
    loss.backward()
    nploss = loss.detach().numpy()

    return nploss


start = time.time()
optimizer.step(closure)
end = time.time()



fig = plt.figure()
ax = fig.add_subplot(1,2,1,projection='3d')


x_pts = np.linspace(0,1,200)
y_pts = np.linspace(0,1,200)

ms_x, ms_y = np.meshgrid(x_pts,y_pts)

x_pts = np.ravel(ms_x).reshape(-1,1)
t_pts = np.ravel(ms_y).reshape(-1,1)

collocations = np.concatenate([x_pts,t_pts], axis=1)

u_gt1,f = g_tr.data_gen_interior(collocations)
#u_gt1 = [np.sin(np.pi*x_col)*np.sin(np.pi*y_col) for x_col,y_col in zip(x_pts,t_pts)]
#u_gt1 = [np.exp(x_col+y_col) for x_col,y_col in zip(x_pts,t_pts)]

#u_gt = np.array(u_gt1)

ms_ugt = u_gt1.reshape(ms_x.shape)

pt_x = Variable(torch.from_numpy(x_pts).float(),requires_grad=True)
pt_t = Variable(torch.from_numpy(t_pts).float(),requires_grad=True)

pt_y = y(pt_x,pt_t)
y = pt_y.data.cpu().numpy()
ms_ysol = y.reshape(ms_x.shape)

ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

surf = ax.plot_surface(ms_x,ms_y,ms_ysol,cmap=cm.coolwarm, linewidth=0,antialiased= False)
fig.colorbar(surf, shrink=0.5, aspect=5)

ax = fig.add_subplot(1,2,2,projection='3d')
surf1 = ax.plot_surface(ms_x,ms_y,ms_ugt,cmap=cm.coolwarm, linewidth=0,antialiased= False)
fig.colorbar(surf1, shrink=0.5, aspect=5)

plt.show()        






