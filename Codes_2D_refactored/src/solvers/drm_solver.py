import os
import torch
import torch.optim as opt
from torch.autograd import Variable
import pickle as pkl
import numpy as np

from ..models import NN
from ..pde import get_pde
from ..data import DataLoader
from ..validation import ValidationMetrics
from ..utils import CheckpointManager
from ..utils.plotting import plot_2D_solution, plot_3D_solution


class DRMSolver:
    """
    Unified Deep Ritz Method solver for 2D fourth-order PDE problems.
    Supports all four problems (P1-P4) through configuration.
    """

    def __init__(self, config, ground_truth_func):
        """
        Initialize DRM solver.

        Args:
            config: Configuration dictionary containing all parameters
            ground_truth_func: Function to compute ground truth for validation
                              Should accept collocation points and return
                              (y_gt, y_gt_x, y_gt_y, y_gt_xx, y_gt_yy, y_gt_xy)
        """
        self.config = config
        self.ground_truth_func = ground_truth_func

        # Set random seed for reproducibility
        if 'seed' in config:
            torch.manual_seed(config['seed'])
            np.random.seed(config['seed'])

        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() and config.get('use_cuda', False) else 'cpu')
        print(f"Using device: {self.device}")

        # Initialize model
        self._initialize_model()

        # Load data
        self._load_data()

        # Initialize PDE
        self.pde = get_pde(config['problem'])

        # Initialize optimizers
        self._initialize_optimizers()

        # Create results directory (needed before checkpoint manager)
        self.results_dir = config.get('results_dir', 'results')
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(os.path.join(self.results_dir, 'y_plot'), exist_ok=True)

        # Initialize validation
        self._initialize_validation()

        # Initialize checkpoint manager
        self._initialize_checkpoint_manager()

        # Loss tracking
        self.losslist = []

    def _initialize_model(self):
        """Initialize neural network model."""
        model_config = self.config.get('model', {})
        self.model = NN(
            input_dim=model_config.get('input_dim', 2),
            hidden_dim=model_config.get('hidden_dim', 50),
            num_layers=model_config.get('num_layers', 4),
            output_dim=model_config.get('output_dim', 1)
        )

        # Load pretrained weights if available
        checkpoint_path = os.path.join(
            self.config.get('results_dir', 'results'),
            'y.pt'
        )
        if os.path.exists(checkpoint_path) and self.config.get('resume_training', False):
            print(f"Loading pretrained model from {checkpoint_path}")
            self.model = torch.load(checkpoint_path, weights_only=False)
        else:
            from ..models import init_weights
            self.model.apply(init_weights)

        self.model.to(self.device)

    def _load_data(self):
        """Load training data."""
        data_config = self.config['data']
        self.data_loader = DataLoader(
            dataset_dir=data_config['dataset_dir'],
            dataname=data_config['dataname'],
            dtype=torch.float32
        )

        # Get training data
        self.train_data = self.data_loader.get_training_data()

        # Move training data to device
        for key in self.train_data:
            self.train_data[key] = self.train_data[key].to(self.device)

        # Create batch loader
        self.batch_loader = self.data_loader.create_data_loader(
            batch_size=data_config.get('batch_size', 2000),
            shuffle=True
        )

    def _initialize_optimizers(self):
        """Initialize Adam and LBFGS optimizers."""
        adam_config = self.config['optimizer']['adam']
        lbfgs_config = self.config['optimizer']['lbfgs']

        # Adam optimizer
        self.optimizer_adam = opt.Adam(
            self.model.parameters(),
            lr=adam_config.get('lr', 1e-4),
            betas=adam_config.get('betas', (0.9, 0.999)),
            eps=adam_config.get('eps', 1e-8),
            weight_decay=adam_config.get('weight_decay', 0.0)
        )

        # Learning rate scheduler
        scheduler_type = adam_config.get('scheduler', 'plateau')
        if scheduler_type == 'plateau':
            self.scheduler = opt.lr_scheduler.ReduceLROnPlateau(
                self.optimizer_adam,
                mode='min',
                factor=adam_config.get('lr_factor', 0.5),
                patience=adam_config.get('lr_patience', 500)
            )
        elif scheduler_type == 'multistep':
            self.scheduler = opt.lr_scheduler.MultiStepLR(
                self.optimizer_adam,
                milestones=adam_config.get('lr_milestones', [8000, 12000]),
                gamma=adam_config.get('lr_gamma', 0.1)
            )
        else:
            self.scheduler = None

        # LBFGS optimizer
        self.optimizer_lbfgs = opt.LBFGS(
            self.model.parameters(),
            line_search_fn=lbfgs_config.get('line_search_fn', 'strong_wolfe'),
            max_iter=lbfgs_config.get('max_iter', 20),
            tolerance_grad=lbfgs_config.get('tolerance_grad', 1e-10),
            tolerance_change=lbfgs_config.get('tolerance_change', 1e-10)
        )

    def _initialize_validation(self):
        """Initialize validation metrics."""
        validation_config = self.config.get('validation', {})
        self.validation_metrics = ValidationMetrics(
            ground_truth_func=self.ground_truth_func,
            resolution=validation_config.get('resolution', 50),
            device=self.device
        )

    def _initialize_checkpoint_manager(self):
        """Initialize checkpoint manager."""
        checkpoint_config = self.config.get('checkpoint', {})
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=os.path.join(self.results_dir, 'checkpoints'),
            save_frequency=checkpoint_config.get('save_frequency', 100)
        )

    def _closure_adam(self):
        """Closure function for Adam optimizer."""
        tot_loss = 0
        tot_loss_int = 0
        tot_loss_bdry = 0

        for i, subquad in enumerate(self.batch_loader):
            self.optimizer_adam.zero_grad()
            ttintx1 = Variable(subquad[0].float().to(self.device), requires_grad=True)
            ttintx2 = Variable(subquad[1].float().to(self.device), requires_grad=True)

            # Compute PDE loss
            loss_output = self.pde.pdeloss(
                self.model, ttintx1, ttintx2,
                self.train_data['f'],
                self.train_data['bdx1'],
                self.train_data['bdx2'],
                self.train_data['nx1'],
                self.train_data['nx2'],
                self.train_data['bdrydata_dirichlet'],
                self.train_data['bdrydata_neumann'],
                self.config['training']['bw_dir'],
                self.config['training']['bw_neu'],
                balancing_wt=self.config['training'].get('balancing_wt', 1.0)
            )

            loss = loss_output[0]
            loss_int = loss_output[1] if len(loss_output) > 1 else loss
            loss_bdry = loss_output[-1] if len(loss_output) > 2 else torch.tensor(0.0, device=self.device)

            loss.backward()
            self.optimizer_adam.step()

            tot_loss += loss
            tot_loss_int += loss_int
            tot_loss_bdry += loss_bdry

        nploss = tot_loss.detach().cpu().numpy()
        nploss_int = tot_loss_int.detach().cpu().numpy()
        nploss_bdry = tot_loss_bdry.detach().cpu().numpy()

        # Update scheduler
        if self.scheduler is not None:
            if isinstance(self.scheduler, opt.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(nploss)
            else:
                self.scheduler.step()

        return nploss, nploss_int, nploss_bdry

    def _closure_lbfgs(self):
        """Closure function for LBFGS optimizer."""
        self.optimizer_lbfgs.zero_grad()

        tot_loss = 0
        for i, subquad in enumerate(self.batch_loader):
            ttintx1 = Variable(subquad[0].float().to(self.device), requires_grad=True)
            ttintx2 = Variable(subquad[1].float().to(self.device), requires_grad=True)

            loss_output = self.pde.pdeloss(
                self.model, ttintx1, ttintx2,
                self.train_data['f'],
                self.train_data['bdx1'],
                self.train_data['bdx2'],
                self.train_data['nx1'],
                self.train_data['nx2'],
                self.train_data['bdrydata_dirichlet'],
                self.train_data['bdrydata_neumann'],
                self.config['training']['bw_dir'],
                self.config['training']['bw_neu'],
                balancing_wt=self.config['training'].get('balancing_wt', 1.0)
            )

            tot_loss += loss_output[0]

        tot_loss.backward()
        return tot_loss

    def train(self):
        """
        Main training loop with two-phase optimization (Adam + LBFGS).
        """
        print("=" * 80)
        print("PHASE 1: Training with Adam Optimizer")
        print("=" * 80)

        # Phase 1: Adam optimizer
        adam_epochs = self.config['training']['adam_epochs']
        validation_freq = self.config['validation'].get('frequency', 100)

        for epoch in range(adam_epochs):
            loss, loss_int, loss_bdry = self._closure_adam()
            self.losslist.append(loss)

            if epoch % validation_freq == 0:
                print(f"Epoch {epoch} | Total loss: {loss:.8f} | Loss_int: {loss_int:.8f} | Loss_bdry: {loss_bdry:.8f}")

                # Compute validation errors
                errors = self.validation_metrics.compute_errors(self.model)
                print(f"  L2 Error: {errors['l2_error']:.8f} | H1 Error: {errors['h1_error']:.8f}")

                # Save checkpoint
                metrics = {
                    'total_loss': loss,
                    'loss_int': loss_int,
                    'loss_bdry': loss_bdry,
                    **errors
                }
                self.checkpoint_manager.save_checkpoint(self.model, epoch, 'adam', metrics)

                # Plot solution
                if self.config.get('save_plots', True):
                    plot_path = os.path.join(self.results_dir, 'y_plot', f'epoch{epoch}.png')
                    plot_2D_solution(self.model, 50, plot_path, title=f"Epoch {epoch}")

        print("\n" + "=" * 80)
        print("PHASE 2: Fine-tuning with LBFGS Optimizer")
        print("=" * 80)

        # Phase 2: LBFGS optimizer
        lbfgs_iterations = self.config['training']['lbfgs_iterations']

        for iteration in range(lbfgs_iterations):
            self.optimizer_lbfgs.step(self._closure_lbfgs)

            if iteration % validation_freq == 0 or iteration == lbfgs_iterations - 1:
                # Evaluate loss
                tot_loss = 0
                tot_loss_int = 0
                tot_loss_bdry = 0

                for i, subquad in enumerate(self.batch_loader):
                    ttintx1 = Variable(subquad[0].float().to(self.device), requires_grad=True)
                    ttintx2 = Variable(subquad[1].float().to(self.device), requires_grad=True)

                    loss_output = self.pde.pdeloss(
                        self.model, ttintx1, ttintx2,
                        self.train_data['f'],
                        self.train_data['bdx1'],
                        self.train_data['bdx2'],
                        self.train_data['nx1'],
                        self.train_data['nx2'],
                        self.train_data['bdrydata_dirichlet'],
                        self.train_data['bdrydata_neumann'],
                        self.config['training']['bw_dir'],
                        self.config['training']['bw_neu'],
                        balancing_wt=self.config['training'].get('balancing_wt', 1.0)
                    )

                    tot_loss += loss_output[0]
                    tot_loss_int += loss_output[1] if len(loss_output) > 1 else loss_output[0]
                    tot_loss_bdry += loss_output[-1] if len(loss_output) > 2 else torch.tensor(0.0, device=self.device)

                loss_val = tot_loss.detach().cpu().numpy()
                loss_int_val = tot_loss_int.detach().cpu().numpy()
                loss_bdry_val = tot_loss_bdry.detach().cpu().numpy()

                self.losslist.append(loss_val)

                print(f"LBFGS Iteration {iteration} | Total loss: {loss_val:.8f} | Loss_int: {loss_int_val:.8f} | Loss_bdry: {loss_bdry_val:.8f}")

                # Compute validation errors
                errors = self.validation_metrics.compute_errors(self.model)
                print(f"  L2 Error: {errors['l2_error']:.8f} | H1 Error: {errors['h1_error']:.8f}")

                # Save checkpoint
                metrics = {
                    'total_loss': loss_val,
                    'loss_int': loss_int_val,
                    'loss_bdry': loss_bdry_val,
                    **errors
                }
                self.checkpoint_manager.save_checkpoint(self.model, iteration, 'lbfgs', metrics, force_save=(iteration == lbfgs_iterations - 1))

                # Plot solution
                if self.config.get('save_plots', True):
                    plot_path = os.path.join(self.results_dir, 'y_plot', f'lbfgs_iter{iteration}.png')
                    plot_2D_solution(self.model, 50, plot_path, title=f"LBFGS Iteration {iteration}")

        print("\n" + "=" * 80)
        print("Training Complete!")
        print("=" * 80)

        # Save final model
        torch.save(self.model, os.path.join(self.results_dir, 'y.pt'))

        # Save loss history
        with open(os.path.join(self.results_dir, 'loss.pkl'), 'wb') as pfile:
            pkl.dump(self.losslist, pfile)

        # Print checkpoint summary
        self.checkpoint_manager.print_summary()
