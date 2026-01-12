import torch
from .base_pde import BasePDE


class P1_PDE(BasePDE):
    """
    PDE implementation for Problem 1.
    Biharmonic equation: Δ²u = f
    """

    def pde(self, x1, x2, net):
        """
        Compute second derivatives for biharmonic operator.

        Args:
            x1: First spatial coordinate
            x2: Second spatial coordinate
            net: Neural network model

        Returns:
            Tuple of (u_xx, u_xy, u_yx, u_yy)
        """
        u = net(x1, x2)

        # First derivatives
        u_x = torch.autograd.grad(u.sum(), x1, create_graph=True)[0]
        u_y = torch.autograd.grad(u.sum(), x2, create_graph=True)[0]

        # Second derivatives
        u_xx = torch.autograd.grad(u_x.sum(), x1, create_graph=True)[0]
        u_yy = torch.autograd.grad(u_y.sum(), x2, create_graph=True)[0]
        u_xy = torch.autograd.grad(u_x.sum(), x2, create_graph=True)[0]
        u_yx = torch.autograd.grad(u_y.sum(), x1, create_graph=True)[0]

        return u_xx, u_xy, u_yx, u_yy

    def pdeloss(self, net, intx1, intx2, pdedata, bdx1, bdx2, nx1, nx2,
                bdrydata_1, bdrydata_2, bw_diri, bw_neumann, **kwargs):
        """
        Compute PDE loss for Problem 1.

        Returns:
            Tuple of (total_loss, loss_int_D2, loss_int, loss_neumann, loss_diri)
        """
        out = net(intx1, intx2)
        bdx1 = bdx1.detach().requires_grad_(True)
        bdx2 = bdx2.detach().requires_grad_(True)

        u_xx, u_xy, u_yx, u_yy = self.pde(intx1, intx2, net)
        zero_vec = torch.zeros([len(intx1), 1], device=intx1.device)

        # Interior loss
        loss_int_1 = self.mse_loss(u_xx, zero_vec)
        loss_int_2 = self.mse_loss(u_xy, zero_vec)
        loss_int_3 = self.mse_loss(u_yx, zero_vec)
        loss_int_4 = self.mse_loss(u_yy, zero_vec)
        loss_int_f = torch.mean(out * pdedata)

        loss_int_D2 = 0.5 * loss_int_1 + 0.5 * loss_int_2 + 0.5 * loss_int_3 + 0.5 * loss_int_4
        loss_int = 0.5 * loss_int_1 + 0.5 * loss_int_2 + 0.5 * loss_int_3 + 0.5 * loss_int_4 - loss_int_f

        # Boundary loss
        bout_diri, bout_neumann = self.bdry(bdx1, bdx2, nx1, nx2, net)
        loss_bdry_neumann = self.mse_loss(bdrydata_2, bout_neumann)
        loss_bdry_diri = self.mse_loss(bout_diri, bdrydata_1)
        loss_bdry = bw_diri * loss_bdry_neumann + bw_diri * loss_bdry_diri

        # Total loss
        loss = loss_int + loss_bdry

        return loss, loss_int_D2, loss_int, loss_bdry_neumann, loss_bdry_diri
