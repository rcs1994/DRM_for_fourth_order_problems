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
import csv
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


torch.set_default_dtype(torch.float32)

y = model.NN()
# load weights if available
if os.path.exists('Codes_2D/P1/DRM/results/y.pt'):
    y = torch.load('Codes_2D/P1/DRM/results/y.pt', weights_only=False)
else:
    y.apply(model.init_weights)
dataname = '20000pts'
name = 'results/'

lambda_parameter = 8000

nepochs = 18000
bw_dir = 0.5 * lambda_parameter
bw_neu = 0.5 * lambda_parameter

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




optimizer = opt.Adam(params, lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)

# It reduces the learning rate by factor=0.1 if the loss hasn't improved for 100 epochs
scheduler = opt.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=500)

# Alternatively, you can use MultiStepLR to decay the learning rate at specific milestones

# Learning rate will decay by gamma at 10000 and 13000 iterations
#scheduler = opt.lr_scheduler.MultiStepLR(optimizer, milestones=[8000, 12000], gamma=0.1)



mse_loss = torch.nn.MSELoss()

loader = torch.utils.data.DataLoader([intx1,intx2],batch_size = 2000,shuffle = True)



def closure():
    tot_loss = 0
    tot_loss_int = 0
    tot_loss_neumann = 0
    tot_loss_diri = 0

    for i, subquad in enumerate(loader):
        optimizer.zero_grad()
        ttintx1 = Variable(subquad[0].float(), requires_grad=True)
        ttintx2 = Variable(subquad[1].float(), requires_grad=True)

        loss, loss_int_D2, loss_int, loss_neumann, loss_diri = pde.pdeloss(
            y, ttintx1, ttintx2, f, tbdx1, tbdx2, tnx1, tnx2,
            bdrydata_dirichlet, bdrydata_neumann, bw_dir, bw_neu
        )

        loss.backward()
        optimizer.step()

        tot_loss += loss
        tot_loss_int += loss_int
        tot_loss_neumann += loss_neumann
        tot_loss_diri += loss_diri


    nploss = tot_loss.detach().numpy()
    nploss_int = tot_loss_int.detach().numpy()
    nploss_neumann = tot_loss_neumann.detach().numpy()
    nploss_diri = tot_loss_diri.detach().numpy()

    scheduler.step(nploss)       #if schedular is ReduceLROnPlateau, then scheduler.step(nploss)

    #if schedular is MultistepLR, then scheduler.step()
    return nploss, nploss_int, nploss_neumann, nploss_diri
  


# def closure():
#     optimizer.zero_grad()
#     loss,pres,bres,diri_loss,neuman_loss = pde.pdeloss(y,tintx1,tintx2,f,tbdx1,tbdx2,tnx1,tnx2,bdrydata_dirichlet,bdrydata_neumann,bw_dir,bw_neu) #ttintx1 are from loader
#     loss.backward()
#     optimizer.step()

#     nploss = loss.detach().numpy()
#     scheduler.step(nploss)
#     return nploss



  

losslist = []

for epoch in range(nepochs): 
    loss, loss_int, loss_neumann, loss_diri = closure()
    losslist.append(loss)

    if epoch % 100 == 0:
        print(f"Epoch {epoch} | Total loss: {loss:.8f} | Loss_int: {loss_int:.8f} | Loss_neumann: {loss_neumann:.8f} | Loss_dirichlet: {loss_diri:.8f}")
        validation.plot_2D(y,name+"y_plot/"+'epoch{}'.format(epoch))
        # Compute and print L2 error
        l2_error, l2_relative_error = validation.compute_error(y)
        print(f"Epoch {epoch} | L2 Error: {l2_error:.8f} | Relative L2 Error: {l2_relative_error:.8f}")




torch.save(y,name+'y.pt')

with open("results/loss.pkl",'wb') as pfile:
    pkl.dump(losslist,pfile)








x_pts = np.linspace(0,1,200)
y_pts = np.linspace(0,1,200)

ms_x, ms_y = np.meshgrid(x_pts,y_pts)

x_pts = np.ravel(ms_x).reshape(-1,1)
t_pts = np.ravel(ms_y).reshape(-1,1)

collocations = np.concatenate([x_pts,t_pts], axis=1)

y_gt, y_gt_x, y_gt_y, y_gt_xx, y_gt_yy, y_gt_xy = g_tr.error_data_gen_interior(collocations)


ms_ugt = y_gt.reshape(ms_x.shape)
ms_ugt_x = y_gt_x.reshape(ms_x.shape)
ms_ugt_y = y_gt_y.reshape(ms_x.shape)   
ms_ugt_xx = y_gt_xx.reshape(ms_x.shape)
ms_ugt_yy = y_gt_yy.reshape(ms_x.shape) 
ms_ugt_xy = y_gt_xy.reshape(ms_x.shape)

pt_x = Variable(torch.from_numpy(x_pts).float(),requires_grad=True)
pt_t = Variable(torch.from_numpy(t_pts).float(),requires_grad=True)

pt_y = y(pt_x,pt_t)


y_x = torch.autograd.grad(
    pt_y, pt_x,
    grad_outputs=torch.ones_like(pt_y),
    create_graph=True,
    retain_graph=True
)[0]

y_t = torch.autograd.grad(
    pt_y, pt_t,
    grad_outputs=torch.ones_like(pt_y),
    create_graph=True,
    retain_graph=True
)[0]

y_xx = torch.autograd.grad(
    y_x, pt_x,
    grad_outputs=torch.ones_like(y_x),
    create_graph=True,
    retain_graph=True
)[0]

y_tt = torch.autograd.grad(
    y_t, pt_t,
    grad_outputs=torch.ones_like(y_t),
    create_graph=True,
    retain_graph=True
)[0]


y_xt = torch.autograd.grad(
    y_x, pt_t,
    grad_outputs=torch.ones_like(y_x),
    create_graph=True,
    retain_graph=True
)[0]
 

y_np = pt_y.detach().cpu().numpy().reshape(ms_x.shape)
y_x_np = y_x.detach().cpu().numpy().reshape(ms_x.shape)
y_t_np = y_t.detach().cpu().numpy().reshape(ms_x.shape)

y_xx_np = y_xx.detach().cpu().numpy().reshape(ms_x.shape)
y_tt_np = y_tt.detach().cpu().numpy().reshape(ms_x.shape)
y_xt_np = y_xt.detach().cpu().numpy().reshape(ms_x.shape)








#groudtruth
ms_ugt_flat = ms_ugt.flatten()

ms_ugt_x_flat = ms_ugt_x.flatten()
ms_ugt_y_flat = ms_ugt_y.flatten()

ms_ugt_xx_flat = ms_ugt_xx.flatten()
ms_ugt_yy_flat = ms_ugt_yy.flatten()
ms_ugt_xy_flat = ms_ugt_xy.flatten()


#computed solution

ms_ysol_flat = y_np.flatten()
ms_ysol_x_flat = y_x_np.flatten()
ms_ysol_y_flat = y_t_np.flatten()
ms_ysol_xx_flat = y_xx_np.flatten()
ms_ysol_yy_flat = y_tt_np.flatten()
ms_ysol_xy_flat = y_xt_np.flatten()





# Compute the L2 error
l2_error = np.sqrt(np.mean((ms_ugt_flat - ms_ysol_flat)**2))

h1_seminorm_error = np.sqrt(np.mean((ms_ugt_x_flat - ms_ysol_x_flat)**2) + np.mean((ms_ugt_y_flat - ms_ysol_y_flat)**2))

h2_seminorm_error = np.sqrt(np.mean((ms_ugt_xx_flat - ms_ysol_xx_flat)**2) + np.mean((ms_ugt_yy_flat - ms_ysol_yy_flat)**2) + 2*np.mean((ms_ugt_xy_flat - ms_ysol_xy_flat)**2))

h2_error = l2_error + h1_seminorm_error + h2_seminorm_error

h2_norm = np.sqrt(np.mean((ms_ugt_flat)**2) + np.mean((ms_ugt_x_flat)**2) + np.mean((ms_ugt_y_flat)**2) + np.mean((ms_ugt_xx_flat)**2) + np.mean((ms_ugt_yy_flat)**2) + 2*np.mean((ms_ugt_xy_flat)**2))

l2_realtiveError= l2_error / (np.sqrt(np.mean((ms_ugt_flat)**2)))

h2_relativeError = h2_error / h2_norm

print(f"L2 Error: {l2_error}")

print(f"L2relativeError : {l2_realtiveError}")

print(f"h2 Error: {h2_error}")
print(f"h2 relative Error: {h2_relativeError}")



print(f"L2 Error: {l2_error}")
print(f"L2relativeError : {l2_realtiveError}")
print(f"h2 Error: {h2_error}")
print(f"h2 relative Error: {h2_relativeError}")

# Save errors to CSV
csv_path = 'Codes_2D/P1/DRM/results/errors.csv'
os.makedirs(os.path.dirname(csv_path), exist_ok=True)

with open(csv_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Metric', 'Value'])
    writer.writerow(['L2 Error', l2_error])
    writer.writerow(['L2 Relative Error', l2_realtiveError])
    writer.writerow(['H2 Error', h2_error])
    writer.writerow(['H2 Relative Error', h2_relativeError])

print(f"Errors saved to {csv_path}")






print("Current directory:", os.getcwd()) 

fig_1 = plt.figure(1, figsize=(6, 5))
plt.pcolor(ms_x,ms_y,y_np, cmap='jet', shading='auto')
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
surf = ax.plot_surface(ms_x, ms_y, y_np, cmap='jet', edgecolor='none')

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









