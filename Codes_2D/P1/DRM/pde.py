import torch

mse_loss = torch.nn.MSELoss()

def pde(x1, x2, net):
    """
    Compute Δ²u = (u_xx + u_yy)_xx + (u_xx + u_yy)_yy
    at the points (x1, x2) via automatic differentiation.
    """
    # 1) forward pass
    u = net(x1, x2)

    # 2) first derivatives
    u_x = torch.autograd.grad(u.sum(),x1, create_graph=True)[0]
    u_y = torch.autograd.grad(u.sum(),x2, create_graph=True)[0]
   

    # 3) second derivatives
    u_xx = torch.autograd.grad(u_x.sum(), x1, create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y.sum(), x2, create_graph=True)[0]
    u_xy = torch.autograd.grad(u_x.sum(), x2, create_graph=True)[0] 
    u_yx = torch.autograd.grad(u_y.sum(), x1, create_graph=True)[0]

   
   # |D^2u|^2 = u_xx^2 + 2 u_xy^2 + u_yy^2


    return u_xx, u_xy, u_yx, u_yy



def bdry(x1,x2,n1,n2,net):
    out = net(x1,x2)
    u_x = torch.autograd.grad(out.sum(),x1,create_graph=True)[0]
    u_y = torch.autograd.grad(out.sum(),x2,create_graph=True)[0]


    return out, u_x*n1 + u_y*n2


def pdeloss(net,intx1,intx2,pdedata,bdx1,bdx2,nx1,nx2,bdrydata_diri,bdrydat_neumann,bw_diri,bw_neumann):
    out = net(intx1,intx2)
    bdx1  = bdx1.detach().requires_grad_(True)
    bdx2  = bdx2.detach().requires_grad_(True)

    u_xx, u_xy, u_yx, u_yy = pde(intx1,intx2,net)
    zero_vec = torch.zeros([len(intx1),1])
    loss_int_1 = mse_loss(u_xx,zero_vec)
    loss_int_2 = mse_loss(u_xy,zero_vec)
    loss_int_3 = mse_loss(u_yx,zero_vec)
    loss_int_4 = mse_loss(u_yy,zero_vec)
    loss_int_f = torch.mean(out*pdedata)

    loss_int_D2 = 0.5*loss_int_1 + 0.5*loss_int_2 + 0.5*loss_int_3 + 0.5*loss_int_4

    loss_int = 0.5*loss_int_1 + 0.5*loss_int_2 + 0.5*loss_int_3 + 0.5*loss_int_4 - loss_int_f

    bout_diri,bout_neumann = bdry(bdx1,bdx2,nx1,nx2,net)

    loss_bdry_neumann = mse_loss(bdrydat_neumann, bout_neumann)
    loss_bdry_diri = mse_loss(bout_diri,bdrydata_diri)

    loss_bdry =  bw_diri*loss_bdry_neumann + bw_diri*loss_bdry_diri



    loss = loss_int + loss_bdry

    return loss, loss_int_D2, loss_int , loss_bdry_neumann, loss_bdry_diri
