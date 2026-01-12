import torch


class BasePDE:
    """
    Base class for PDE loss computation.
    Each problem (P1, P2, P3, P4) should inherit from this class.
    """

    def __init__(self):
        self.mse_loss = torch.nn.MSELoss()

    def pde(self, x1, x2, net):
        """
        Compute PDE operator on the interior.
        Should be overridden by subclasses.

        Args:
            x1: First spatial coordinate
            x2: Second spatial coordinate
            net: Neural network model

        Returns:
            PDE residual(s)
        """
        raise NotImplementedError("Subclasses must implement pde() method")

    def bdry(self, x1, x2, n1, n2, net):
        """
        Compute boundary values.

        Args:
            x1: First spatial coordinate on boundary
            x2: Second spatial coordinate on boundary
            n1: First component of outward normal vector
            n2: Second component of outward normal vector
            net: Neural network model

        Returns:
            Tuple of (boundary_value, normal_derivative)
        """
        out = net(x1, x2)
        u_x = torch.autograd.grad(out.sum(), x1, create_graph=True)[0]
        u_y = torch.autograd.grad(out.sum(), x2, create_graph=True)[0]

        return out, u_x * n1 + u_y * n2

    def pdeloss(self, net, intx1, intx2, pdedata, bdx1, bdx2, nx1, nx2,
                bdrydata_1, bdrydata_2, bw_diri, bw_neumann, **kwargs):
        """
        Compute total PDE loss (interior + boundary).
        Should be overridden by subclasses if needed.

        Args:
            net: Neural network model
            intx1, intx2: Interior collocation points
            pdedata: Source term data
            bdx1, bdx2: Boundary collocation points
            nx1, nx2: Normal vectors at boundary points
            bdrydata_1: First boundary condition data (e.g., Dirichlet)
            bdrydata_2: Second boundary condition data (e.g., Neumann)
            bw_diri: Boundary weight for Dirichlet condition
            bw_neumann: Boundary weight for Neumann condition
            **kwargs: Additional problem-specific parameters

        Returns:
            Tuple of losses (total_loss, interior_loss, boundary_loss)
        """
        raise NotImplementedError("Subclasses must implement pdeloss() method")
