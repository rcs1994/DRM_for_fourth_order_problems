import torch

mse_loss = torch.nn.MSELoss()

def pde(x1, x2, x3, net):
    """
    Compute Δ²u = (u_xx + u_yy)_xx + (u_xx + u_yy)_yy
    at the points (x1, x2) via automatic differentiation.
    """
    # 1) forward pass
    u = net(x1, x2, x3)

    # 2) first derivatives
    u_x = torch.autograd.grad(u.sum(),x1, create_graph=True)[0]
    u_y = torch.autograd.grad(u.sum(),x2, create_graph=True)[0]
    u_z = torch.autograd.grad(u.sum(),x3, create_graph=True)[0]
   

    # 3) second derivatives
    u_xx = torch.autograd.grad(u_x.sum(), x1, create_graph=True)[0]
    u_xy = torch.autograd.grad(u_x.sum(), x2, create_graph=True)[0]
    u_xz = torch.autograd.grad(u_x.sum(), x3, create_graph=True)[0]
    
    u_yx = torch.autograd.grad(u_y.sum(), x1, create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y.sum(), x2, create_graph=True)[0]
    u_yz = torch.autograd.grad(u_y.sum(), x3, create_graph=True)[0]
     
    u_zx = torch.autograd.grad(u_z.sum(), x1, create_graph=True)[0]
    u_zy = torch.autograd.grad(u_z.sum(), x2, create_graph=True)[0]
    u_zz = torch.autograd.grad(u_z.sum(), x3, create_graph=True)[0]

   
   # |D^2u|^2 = u_xx^2 + 2 u_xy^2 + u_yy^2


    return u_xx, u_xy, u_xz, u_yx, u_yy, u_yz, u_zx, u_zy, u_zz



def bdry(x1,x2,x3,n1,n2,n3,net):
    out = net(x1,x2,x3)
    u_x = torch.autograd.grad(out.sum(),x1,create_graph=True)[0]
    u_y = torch.autograd.grad(out.sum(),x2,create_graph=True)[0]
    u_z = torch.autograd.grad(out.sum(),x3,create_graph=True)[0]


    return out, u_x*n1 + u_y*n2 + u_z*n3


def pdeloss(net,intx1,intx2,intx3,pdedata,bdx1,bdx2,bdx3,nx1,nx2,nx3,bdrydata_diri,bdrydat_neumann,bw_diri,bw_neumann):
    out = net(intx1,intx2,intx3)
    bdx1  = bdx1.detach().requires_grad_(True)
    bdx2  = bdx2.detach().requires_grad_(True)
    bdx3  = bdx3.detach().requires_grad_(True)

    u_xx, u_xy, u_xz, u_yx, u_yy, u_yz, u_zx, u_zy, u_zz = pde(intx1,intx2,intx3, net)
    zero_vec = torch.zeros([len(intx1),1])
    loss_int_1 = mse_loss(u_xx,zero_vec)
    loss_int_2 = mse_loss(u_xy,zero_vec)
    loss_int_3 = mse_loss(u_xz,zero_vec)
    loss_int_4 = mse_loss(u_yx,zero_vec)
    loss_int_5 = mse_loss(u_yy,zero_vec)
    loss_int_6 = mse_loss(u_yz,zero_vec)
    loss_int_7 = mse_loss(u_zx,zero_vec)    
    loss_int_8 = mse_loss(u_zy,zero_vec)
    loss_int_9 = mse_loss(u_zz,zero_vec)
    
    loss_int_f = torch.mean(out*pdedata)

    loss_int_D2 = 0.5*loss_int_1 + 0.5*loss_int_2 + 0.5*loss_int_3 + 0.5*loss_int_4 + 0.5*loss_int_5 + 0.5*loss_int_6 + 0.5*loss_int_7 + 0.5*loss_int_8 + 0.5*loss_int_9

    loss_int = loss_int_D2 - loss_int_f

    bout_diri,bout_neumann = bdry(bdx1,bdx2,bdx3,nx1,nx2,nx3,net)

    loss_bdry_neumann = mse_loss(bout_neumann,bdrydat_neumann)
    loss_bdry_diri = mse_loss(bout_diri,bdrydata_diri)

    loss_bdry = bw_diri*loss_bdry_neumann + bw_diri*loss_bdry_diri



    loss = loss_int + loss_bdry

    return loss, loss_int_D2, loss_int , loss_bdry_neumann, loss_bdry_diri
