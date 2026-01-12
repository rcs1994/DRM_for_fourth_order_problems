import numpy as np
import torch
from torch.autograd import Variable


class ValidationMetrics:
    """
    Compute validation metrics (L2, H1, H2 errors) for neural network solutions.
    """

    def __init__(self, ground_truth_func, resolution=50, device=None):
        """
        Initialize validation metrics.

        Args:
            ground_truth_func: Function that returns ground truth data
                               Should accept collocation points and return
                               (y_gt, y_gt_x, y_gt_y, y_gt_xx, y_gt_yy, y_gt_xy)
            resolution: Grid resolution for computing errors
            device: Device to place tensors on (default: cpu)
        """
        self.ground_truth_func = ground_truth_func
        self.resolution = resolution
        self.device = device if device is not None else torch.device('cpu')

        # Create validation grid
        val_x1 = np.arange(0, 1, 1/resolution).reshape(-1, 1)
        val_x2 = np.arange(0, 1, 1/resolution).reshape(-1, 1)
        val_ms_x1, val_ms_x2 = np.meshgrid(val_x1, val_x2)
        plot_val_x1 = np.ravel(val_ms_x1).reshape(-1, 1)
        plot_val_x2 = np.ravel(val_ms_x2).reshape(-1, 1)

        self.val_ms_x1 = val_ms_x1
        self.val_ms_x2 = val_ms_x2
        self.plot_val_x1 = plot_val_x1
        self.plot_val_x2 = plot_val_x2

        # Compute ground truth
        collocations = np.concatenate([plot_val_x1, plot_val_x2], axis=1)
        y_gt, y_gt_x, y_gt_y, y_gt_xx, y_gt_yy, y_gt_xy = ground_truth_func(collocations)

        # Store ground truth values
        self.y_gt = y_gt
        self.y_gt_x = y_gt_x
        self.y_gt_y = y_gt_y
        self.y_gt_xx = y_gt_xx
        self.y_gt_yy = y_gt_yy
        self.y_gt_xy = y_gt_xy

        # Create tensors for network evaluation (on the correct device)
        self.t_val_vx1 = Variable(torch.from_numpy(plot_val_x1).float().to(self.device), requires_grad=True)
        self.t_val_vx2 = Variable(torch.from_numpy(plot_val_x2).float().to(self.device), requires_grad=True)

    def compute_errors(self, net):
        """
        Compute L2, H1, and H2 errors for the network solution.

        Args:
            net: Neural network model

        Returns:
            Dictionary with keys:
                - l2_error: L2 error
                - l2_relative_error: Relative L2 error
                - h1_error: H1 seminorm error
                - h1_relative_error: Relative H1 error
                - h2_error: H2 seminorm error
                - h2_relative_error: Relative H2 error
        """
        # Evaluate network
        pt_y = net(self.t_val_vx1, self.t_val_vx2)

        # Compute first derivatives
        y_x = torch.autograd.grad(
            pt_y, self.t_val_vx1,
            grad_outputs=torch.ones_like(pt_y),
            create_graph=True,
            retain_graph=True
        )[0]

        y_y = torch.autograd.grad(
            pt_y, self.t_val_vx2,
            grad_outputs=torch.ones_like(pt_y),
            create_graph=True,
            retain_graph=True
        )[0]

        # Compute second derivatives
        y_xx = torch.autograd.grad(
            y_x, self.t_val_vx1,
            grad_outputs=torch.ones_like(y_x),
            create_graph=True,
            retain_graph=True
        )[0]

        y_yy = torch.autograd.grad(
            y_y, self.t_val_vx2,
            grad_outputs=torch.ones_like(y_y),
            create_graph=True,
            retain_graph=True
        )[0]

        y_xy = torch.autograd.grad(
            y_x, self.t_val_vx2,
            grad_outputs=torch.ones_like(y_x),
            create_graph=True,
            retain_graph=True
        )[0]

        # Convert to numpy
        y_np = pt_y.detach().cpu().numpy().flatten()
        y_x_np = y_x.detach().cpu().numpy().flatten()
        y_y_np = y_y.detach().cpu().numpy().flatten()
        y_xx_np = y_xx.detach().cpu().numpy().flatten()
        y_yy_np = y_yy.detach().cpu().numpy().flatten()
        y_xy_np = y_xy.detach().cpu().numpy().flatten()

        # Ground truth (flatten)
        y_gt_flat = self.y_gt.flatten()
        y_gt_x_flat = self.y_gt_x.flatten()
        y_gt_y_flat = self.y_gt_y.flatten()
        y_gt_xx_flat = self.y_gt_xx.flatten()
        y_gt_yy_flat = self.y_gt_yy.flatten()
        y_gt_xy_flat = self.y_gt_xy.flatten()

        # Compute L2 error
        l2_error = np.sqrt(np.mean((y_gt_flat - y_np)**2))
        l2_norm = np.sqrt(np.mean(y_gt_flat**2))
        l2_relative_error = l2_error / l2_norm if l2_norm > 0 else 0.0

        # Compute H1 seminorm error
        h1_seminorm_error = np.sqrt(
            np.mean((y_gt_x_flat - y_x_np)**2) +
            np.mean((y_gt_y_flat - y_y_np)**2)
        )
        h1_seminorm_norm = np.sqrt(
            np.mean(y_gt_x_flat**2) +
            np.mean(y_gt_y_flat**2)
        )
        h1_relative_error = h1_seminorm_error / h1_seminorm_norm if h1_seminorm_norm > 0 else 0.0

        # Compute H2 seminorm error
        h2_seminorm_error = np.sqrt(
            np.mean((y_gt_xx_flat - y_xx_np)**2) +
            np.mean((y_gt_yy_flat - y_yy_np)**2) +
            2 * np.mean((y_gt_xy_flat - y_xy_np)**2)
        )
        h2_seminorm_norm = np.sqrt(
            np.mean(y_gt_xx_flat**2) +
            np.mean(y_gt_yy_flat**2) +
            2 * np.mean(y_gt_xy_flat**2)
        )
        h2_relative_error = h2_seminorm_error / h2_seminorm_norm if h2_seminorm_norm > 0 else 0.0

        return {
            'l2_error': l2_error,
            'l2_relative_error': l2_relative_error,
            'h1_error': h1_seminorm_error,
            'h1_relative_error': h1_relative_error,
            'h2_error': h2_seminorm_error,
            'h2_relative_error': h2_relative_error
        }

    def compute_l2_error(self, net):
        """
        Compute only L2 error (faster than computing all errors).

        Args:
            net: Neural network model

        Returns:
            Tuple of (l2_error, l2_relative_error)
        """
        with torch.no_grad():
            pred = net(self.t_val_vx1, self.t_val_vx2).cpu().numpy().flatten()

        y_gt_flat = self.y_gt.flatten()
        l2_error = np.sqrt(np.mean((y_gt_flat - pred)**2))
        l2_norm = np.sqrt(np.mean(y_gt_flat**2))
        l2_relative_error = l2_error / l2_norm if l2_norm > 0 else 0.0

        return l2_error, l2_relative_error
