import os
import pickle as pkl
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import model, pde, tools

# 1) Choose device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 2) Build and move model
y = model.NN().to(device)
y.apply(model.init_weights)

# 3) Hyper‑params
dataname    = '5000pts'
results_dir = 'results/'
bw          = 100.0
lr          = 1e-4
batch_size  = 500
max_epochs  = 50

os.makedirs(results_dir, exist_ok=True)
os.makedirs(os.path.join(results_dir, "y_plot"), exist_ok=True)

# 4) Load collocation points
with open(f"dataset/{dataname}", 'rb') as f:
    int_col    = pkl.load(f)   # (5000,2)
    bdry_col   = pkl.load(f)   # (1000,2)
    normal_vec = pkl.load(f)   # (1000,2)

intx1, intx2 = np.split(int_col,  2, axis=1)
bdx1, bdx2   = np.split(bdry_col, 2, axis=1)
nx1, nx2     = np.split(normal_vec, 2, axis=1)

# 5) Convert ONCE to torch.Tensors (matching tools API) and move to device
#    We mark int & bdry coords to require grad for PDE and boundary derivatives
tintx1, tintx2, tbdx1, tbdx2, tnx1, tnx2 = tools.from_numpy_to_tensor(
    [intx1, intx2, bdx1, bdx2, nx1, nx2],
    [True,  True,  True,  True,  False, False],
    torch.float32
)
tintx1 = tintx1.to(device).requires_grad_(True)
tintx2 = tintx2.to(device).requires_grad_(True)
tbdx1  = tbdx1.to(device).requires_grad_(True)
tbdx2  = tbdx2.to(device).requires_grad_(True)
tnx1   = tnx1.to(device)
tnx2   = tnx2.to(device)

# 6) Load ground truth & move to device
with open(f"dataset/gt_on_{dataname}", 'rb') as f:
    y_gt_np      = pkl.load(f)
    f_np         = pkl.load(f)
    dirichlet_np = pkl.load(f)
    neumann_np   = pkl.load(f)

f_t, bdry_dir_t, bdry_neu_t, ygt_t = tools.from_numpy_to_tensor(
    [f_np, dirichlet_np, neumann_np, y_gt_np],
    [False, False, False, False],
    torch.float32
)
f_t         = f_t.to(device)
bdry_dir_t  = bdry_dir_t.to(device)
bdry_neu_t  = bdry_neu_t.to(device)
ygt_t       = ygt_t.to(device)

# 7) Build DataLoader for interior
train_ds     = TensorDataset(tintx1, tintx2)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True)

# 8) Optimizer + scheduler
optimizer = optim.Adam(y.parameters(), lr=lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=500)

# 9) Training loop
loss_list = []
for epoch in range(1, max_epochs + 1):
    y.train()
    tot_loss = 0.0

    for batch_x1, batch_x2 in train_loader:
        # coords already require grad
        optimizer.zero_grad()
        loss, pres, bres = pde.pdeloss(
            y,
            batch_x1, batch_x2,     # interior coords
            f_t,                   # source term
            tbdx1, tbdx2,          # boundary coords
            tnx1, tnx2,            # boundary normals
            bdry_dir_t,            # Dirichlet targets
            bdry_neu_t,            # Neumann targets
            bw
        )
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()

    scheduler.step(tot_loss)
    loss_list.append(tot_loss)

    if epoch == 1 or epoch % 10 == 0:
        print(f"Epoch {epoch:3d} | Loss = {tot_loss:.4e}")

    # optional: validation.plot_2D(y, os.path.join(results_dir,"y_plot",f"epoch{epoch}"))

# 10) Save loss history
with open(os.path.join(results_dir, "loss.pkl"), "wb") as f:
    pkl.dump(loss_list, f)
