import torch
from .base_pde import BasePDE


class P4_PDE(BasePDE):
    """
    PDE implementation for Problem 4.
    Laplacian with reaction term: Δu + u = f
    """

    def pde(self, x1, x2, net):
        """
        Compute solution and Laplacian: (u, Δu)

        Args:
            x1: First spatial coordinate
            x2: Second spatial coordinate
            net: Neural network model

        Returns:
            Tuple of (u, Δu)
        """
        u = net(x1, x2)

        # First derivatives
        u_x = torch.autograd.grad(u.sum(), x1, create_graph=True)[0]
        u_y = torch.autograd.grad(u.sum(), x2, create_graph=True)[0]

        # Second derivatives
        u_xx = torch.autograd.grad(u_x.sum(), x1, create_graph=True)[0]
        u_yy = torch.autograd.grad(u_y.sum(), x2, create_graph=True)[0]

        return u, u_xx + u_yy

    def pdeloss(self, net, intx1, intx2, pdedata, bdx1, bdx2, nx1, nx2,
                bdrydata_1, bdrydata_2, bw_diri, bw_neumann, **kwargs):
        """
        Compute PDE loss for Problem 4.

        Args:
            balancing_wt: Balancing weight (from kwargs)

        Returns:
            Tuple of (total_loss, loss_int, loss_bdry)
        """
        balancing_wt = kwargs.get('balancing_wt', 1.0)

        out = net(intx1, intx2)
        bdx1 = bdx1.detach().requires_grad_(True)
        bdx2 = bdx2.detach().requires_grad_(True)

        u, lap_u = self.pde(intx1, intx2, net)
        zero_vec = torch.zeros([len(intx1), 1], device=intx1.device)
        ones_vec = torch.ones([len(intx1), 1], device=intx1.device)

        # Interior loss
        loss_int_1 = self.mse_loss(lap_u, zero_vec)
        loss_int_1_with_reaction = self.mse_loss(u, zero_vec)
        loss_int_2 = torch.mean(out * pdedata)
        loss_int_3 = torch.square(torch.mean(ones_vec * out))

        # Loss for PDE with reaction term
        loss_int = 0.5 * loss_int_1 + 0.5 * loss_int_1_with_reaction - loss_int_2

        # Alternative: Loss for PDE without reaction term (commented in original)
        # loss_int = 0.5 * loss_int_1 - loss_int_2 + balancing_wt * loss_int_3

        # Boundary loss
        bout_u, bout_dudn = self.bdry(bdx1, bdx2, nx1, nx2, net)
        loss_bdry_1 = torch.mean(bdrydata_2 * bout_u)
        loss_bdry_2 = self.mse_loss(bout_dudn, bdrydata_1)
        loss_bdry = loss_bdry_1 + bw_neumann * loss_bdry_2

        # Total loss
        loss = loss_int + loss_bdry

        return loss, loss_int, loss_bdry
