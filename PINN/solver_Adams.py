from time import time
from tracemalloc import start
import numpy as np
import torch
from torch.utils.data import TensorDataset
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
from time import time

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


torch.set_default_dtype(torch.float32)

y = model.NN()
y.apply(model.init_weights)

dataname = '5000pts'
name = 'results/'

bw_dir = 5000.0
bw_neu = 5000.0

if not os.path.exists(name):
    os.makedirs(name)

if not os.path.exists(name+"y_plot/"): 
    os.makedirs(name+"y_plot/")


params = list(y.parameters())



with open("dataset/"+dataname,'rb') as pfile:
    int_col = pkl.load(pfile)
    bdry_col = pkl.load(pfile)
    normal_vec = pkl.load(pfile)

print(int_col.shape,bdry_col.shape,normal_vec.shape)

intx1,intx2 = np.split(int_col,2,axis=1)
bdx1,bdx2 = np.split(bdry_col,2,axis=1)
nx1,nx2 = np.split(normal_vec,2,axis=1)

tintx1,tintx2,tbdx1,tbdx2,tnx1,tnx2 = tools.from_numpy_to_tensor([intx1,intx2,bdx1,bdx2,nx1,nx2],[True,True,False,False,True,True],dtype=torch.float32)


with open("dataset/gt_on_{}".format(dataname),'rb') as pfile:
    y_gt = pkl.load(pfile)
    f_np = pkl.load(pfile)
    dirichlet_data_np = pkl.load(pfile)
    neumann_data_np = pkl.load(pfile)

f,bdrydata_dirichlet,bdrydata_neumann,ygt = tools.from_numpy_to_tensor([f_np,dirichlet_data_np,neumann_data_np,y_gt],[False,False,True,False],dtype=torch.float32)


optimizer = opt.Adam(params,lr=1e-4)


mse_loss = torch.nn.MSELoss()

scheduler = opt.lr_scheduler.ReduceLROnPlateau(optimizer,patience=500)

loader = torch.utils.data.DataLoader([intx1,intx2],batch_size = 2000,shuffle = True)



def closure():
    tot_loss = 0
    tot_pres = 0
    tot_diri_loss = 0
    tot_neumann_loss = 0

    for i,subquad in enumerate(loader):
       optimizer.zero_grad()
       ttintx1 = Variable(subquad[0].float(),requires_grad = True)
       ttintx2 = Variable(subquad[1].float(),requires_grad = True)

       loss,pres,bres,diri_loss,neuman_loss = pde.pdeloss(y,ttintx1,ttintx2,f,tbdx1,tbdx2,tnx1,tnx2,bdrydata_dirichlet,bdrydata_neumann,bw_dir,bw_neu) #ttintx1 are from loader
       loss.backward()
       optimizer.step()
       tot_loss = tot_loss + loss

       tot_pres += pres
       tot_diri_loss += diri_loss
       tot_neumann_loss += neuman_loss


       


    np_presloss = tot_pres.detach().numpy()
    np_diri_loss = tot_diri_loss.detach().numpy()
    np_neumann_loss = tot_neumann_loss.detach().numpy()
    nploss = tot_loss.detach().numpy()
    scheduler.step(nploss)
    return nploss, np_presloss, np_diri_loss,np_neumann_loss    


# def closure():
#     optimizer.zero_grad()
#     loss,pres,bres,diri_loss,neuman_loss = pde.pdeloss(y,tintx1,tintx2,f,tbdx1,tbdx2,tnx1,tnx2,bdrydata_dirichlet,bdrydata_neumann,bw_dir,bw_neu) #ttintx1 are from loader
#     loss.backward()
#     optimizer.step()

#     nploss = loss.detach().numpy()
#     scheduler.step(nploss)
#     return nploss



  

losslist = list()

 
for epoch in range(20000):
    loss, presloss, diri_loss,neumann_loss = closure()
    losslist.append(loss)
    if epoch %100==0:
         print(f"Epoch {epoch} | Total loss: {loss:.8f} | Loss_int: {presloss:.8f} | Loss_neumann: {neumann_loss:.8f} | Loss_dirichlet: {diri_loss:.8f}")

        # validation.plot_2D(y,name+"y_plot/"+'epoch{}'.format(epoch))

with open("results/loss.pkl",'wb') as pfile:
    pkl.dump(losslist,pfile)








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



##L^2 error computation
# Flatten the arrays for easier computation
ms_ugt_flat = ms_ugt.flatten()
ms_ysol_flat = ms_ysol.flatten()

# Compute the L2 error
l2_error = np.sqrt(np.mean((ms_ugt_flat - ms_ysol_flat)**2))

l2_realtiveError= l2_error / (np.sqrt(np.mean((ms_ugt_flat)**2)))

print(f"L2 Error: {l2_error}")

print(f"L2relativeError : {l2_realtiveError}")




print("Current directory:", os.getcwd()) 

fig_1 = plt.figure(1, figsize=(6, 5))
plt.pcolor(ms_x,ms_y,ms_ysol, cmap='jet', shading='auto')
h=plt.colorbar()
h.ax.tick_params(labelsize=20)
plt.xticks([])
plt.yticks([])
plt.savefig('NNsolution.png',bbox_inches='tight')
plt.close()



fig_2 = plt.figure(2, figsize=(6, 5))
plt.pcolor(ms_x,ms_y,ms_ugt, cmap='jet', shading='auto')
h=plt.colorbar()
h.ax.tick_params(labelsize=20)
plt.xticks([])
plt.yticks([])
plt.savefig('GTsolution.jpg',bbox_inches='tight')
plt.close()


# fig_3 = plt.figure(3, figsize=(6, 5))
# plt.pcolor(ms_x,ms_y,abs(ms_ugt-ms_ysol), cmap='jet', shading='auto')
# h=plt.colorbar()
# h.ax.tick_params(labelsize=20)
# plt.xticks([])
# plt.yticks([])
# plt.savefig('Error.jpg',bbox_inches='tight')


fig = plt.figure(4,figsize=(6, 5))
ax = fig.add_subplot(111, projection='3d')

# Assuming ms_x, ms_y, ms_ysol are 2D arrays of same shape
surf = ax.plot_surface(ms_x, ms_y, ms_ysol, cmap='jet', edgecolor='none')

h = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
h.ax.tick_params(labelsize=20)

ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

plt.savefig('NNsolution1.png', bbox_inches='tight')
#plt.show()


fig = plt.figure(5,figsize=(6, 5))
ax = fig.add_subplot(111, projection='3d')

# Assuming ms_x, ms_y, ms_ysol are 2D arrays of same shape
surf = ax.plot_surface(ms_x, ms_y, ms_ugt, cmap='jet', edgecolor='none')

h = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
h.ax.tick_params(labelsize=20)

ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

plt.savefig('GTsol1.png', bbox_inches='tight')
#plt.show()









