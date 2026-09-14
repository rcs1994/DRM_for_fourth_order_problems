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

    def bdry(self, x1, x2, n1, n2, net):
        out = net(x1, x2)
        u_x = torch.autograd.grad(out.sum(), x1, create_graph=True)[0]
        u_y = torch.autograd.grad(out.sum(), x2, create_graph=True)[0]


        return out, u_x*n1 + u_y*n2


    def h_fractional_norm_flat(self, w, s, L): #s is fractional order and L is boundary length
        """
        H^s norm of w defined on a flat/periodic boundary,
        discretized on N uniformly spaced points.

        w: tensor of shape (..., N)  -- last dim is boundary DOFs
        s: float -- Sobolev exponent (3/2 here)
        L: float -- length/period of the boundary
        """
        N = w.shape[-1]

        # Fourier coefficients
        w_hat = torch.fft.rfft(w, dim=-1)  # shape (..., N//2 + 1)

        # Frequencies: k = 0, 1, ..., N//2
        k = torch.arange(w_hat.shape[-1], device=w.device, dtype=w.real.dtype)
        freq = 2 * torch.pi * k / L  # physical frequencies

        # Multiplier: (1 + |freq|^2)^s
        multiplier = (1.0 + freq**2)**s  # shape (N//2 + 1,)

        # Weighted Parseval: ||w||^2_{H^s} = (1/N^2) sum_k multiplier_k |w_hat_k|^2
        power = w_hat.abs()**2
        norm_sq = (multiplier * power).sum(dim=-1) / N**2
        norm_sq = norm_sq.mean()
        #norm_sq.backward()

        return norm_sq  # return squared norm; take sqrt if needed



    def pdeloss(self, net, intx1, intx2, pdedata, ygt, bdx1, bdx2, nx1, nx2,
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
        #loss_int_2 = self.mse_loss(u_xy, zero_vec)  # not needed for the new formulation 
        #loss_int_3 = self.mse_loss(u_yx, zero_vec)  # not needed for the new formulation
        loss_int_4 = self.mse_loss(u_yy, zero_vec)
        loss_int_f = torch.mean(out * pdedata)

        loss_int_D2 = 0.5 * loss_int_1  + 0.5 * loss_int_4
        loss_int = 0.5 * loss_int_1 + 0.5 * loss_int_4 - loss_int_f

        # Boundary loss
        bout_diri, bout_neumann = self.bdry(bdx1, bdx2, nx1, nx2, net)
        #loss_bdry_neumann = self.mse_loss(bdrydata_2, bout_neumann)
        #loss_bdry_diri = self.mse_loss(bout_diri, bdrydata_1)
        loss_bdry_diri = self.h_fractional_norm_flat(bout_diri-bdrydata_1,1.5,1)
        loss_bdry_neumann = self.h_fractional_norm_flat(bout_neumann-bdrydata_2,0.5,1)

        loss_bdry = bw_neumann * loss_bdry_neumann + bw_diri * loss_bdry_diri

        # Total loss
        loss = loss_int + loss_bdry

        return loss, loss_int_D2, loss_int, loss_bdry_neumann, loss_bdry_diri
