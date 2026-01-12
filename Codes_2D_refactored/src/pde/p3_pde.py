import torch
from .base_pde import BasePDE


class P3_PDE(BasePDE):
    """
    PDE implementation for Problem 3.
    Uses Laplacian formulation: Δu
    """

    def pde(self, x1, x2, net):
        """
        Compute Laplacian: Δu = u_xx + u_yy

        Args:
            x1: First spatial coordinate
            x2: Second spatial coordinate
            net: Neural network model

        Returns:
            Laplacian of u
        """
        u = net(x1, x2)

        # First derivatives
        u_x = torch.autograd.grad(u.sum(), x1, create_graph=True)[0]
        u_y = torch.autograd.grad(u.sum(), x2, create_graph=True)[0]

        # Second derivatives
        u_xx = torch.autograd.grad(u_x.sum(), x1, create_graph=True)[0]
        u_yy = torch.autograd.grad(u_y.sum(), x2, create_graph=True)[0]

        return u_xx + u_yy

    def pdeloss(self, net, intx1, intx2, pdedata, bdx1, bdx2, nx1, nx2,
                bdrydata_1, bdrydata_2, bw_diri, bw_neumann, **kwargs):
        """
        Compute PDE loss for Problem 3.

        Returns:
            Tuple of (total_loss, loss_int, loss_neumann, loss_diri)
        """
        out = net(intx1, intx2)
        bdx1 = bdx1.detach().requires_grad_(True)
        bdx2 = bdx2.detach().requires_grad_(True)

        lap_u = self.pde(intx1, intx2, net)
        zero_vec = torch.zeros([len(intx1), 1], device=intx1.device)

        # Interior loss
        loss_int_1 = self.mse_loss(lap_u, zero_vec)
        loss_int_2 = torch.mean(out * pdedata)
        loss_int = 0.5 * loss_int_1 - loss_int_2

        # Boundary loss
        bout_diri, bout_neumann = self.bdry(bdx1, bdx2, nx1, nx2, net)
        loss_bdry_neumann = torch.mean(bdrydata_2 * bout_neumann)
        loss_bdry_diri = self.mse_loss(bout_diri, bdrydata_1)
        loss_bdry = -bw_neumann * loss_bdry_neumann + bw_diri * loss_bdry_diri

        # Total loss
        loss = loss_int + loss_bdry

        return loss, loss_int, loss_bdry_neumann, loss_bdry_diri
