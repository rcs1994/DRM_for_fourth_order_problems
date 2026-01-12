import os
import csv
import torch
import numpy as np
from datetime import datetime


class CheckpointManager:
    """
    Manages model checkpoints based on validation metrics.
    Saves the best models according to L2 error and H1 seminorm error.
    """

    def __init__(self, checkpoint_dir, save_frequency=100):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to save checkpoints
            save_frequency: Save checkpoint every N epochs
        """
        self.checkpoint_dir = checkpoint_dir
        self.save_frequency = save_frequency

        # Create checkpoint directory if it doesn't exist
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Initialize tracking variables
        self.best_l2_error = float('inf')
        self.best_h1_error = float('inf')
        self.best_l2_relative_error = float('inf')

        # Paths for different checkpoints
        self.best_l2_path = os.path.join(checkpoint_dir, 'best_l2_model.pt')
        self.best_h1_path = os.path.join(checkpoint_dir, 'best_h1_model.pt')
        self.last_path = os.path.join(checkpoint_dir, 'last_model.pt')

        # CSV file for tracking history
        self.history_path = os.path.join(checkpoint_dir, 'checkpoint_history.csv')
        self._initialize_history_file()

    def _initialize_history_file(self):
        """Initialize CSV file for tracking checkpoint history."""
        if not os.path.exists(self.history_path):
            with open(self.history_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'epoch', 'phase', 'total_loss', 'loss_int', 'loss_bdry',
                    'l2_error', 'l2_relative_error', 'h1_error',
                    'best_l2_saved', 'best_h1_saved', 'timestamp'
                ])

    def save_checkpoint(self, model, epoch, phase, metrics, force_save=False):
        """
        Save checkpoint based on validation metrics.

        Args:
            model: PyTorch model to save
            epoch: Current epoch/iteration number
            phase: Training phase ('adam' or 'lbfgs')
            metrics: Dictionary containing validation metrics
                     {'total_loss', 'loss_int', 'loss_bdry', 'l2_error', 'l2_relative_error', 'h1_error'}
            force_save: Force save regardless of frequency

        Returns:
            Dictionary with keys 'best_l2_saved' and 'best_h1_saved' (bool)
        """
        l2_error = metrics.get('l2_error', float('inf'))
        h1_error = metrics.get('h1_error', float('inf'))
        l2_relative_error = metrics.get('l2_relative_error', float('inf'))

        saved_info = {
            'best_l2_saved': False,
            'best_h1_saved': False
        }

        # Check if this is the best L2 error
        if l2_error < self.best_l2_error:
            self.best_l2_error = l2_error
            self.best_l2_relative_error = l2_relative_error
            torch.save(model, self.best_l2_path)
            saved_info['best_l2_saved'] = True
            print(f"  → Saved new best L2 model (L2 error: {l2_error:.8f})")

        # Check if this is the best H1 error
        if h1_error < self.best_h1_error:
            self.best_h1_error = h1_error
            torch.save(model, self.best_h1_path)
            saved_info['best_h1_saved'] = True
            print(f"  → Saved new best H1 model (H1 error: {h1_error:.8f})")

        # Save last model checkpoint at specified frequency or if forced
        if force_save or (epoch % self.save_frequency == 0):
            torch.save(model, self.last_path)

        # Log to history
        self._log_to_history(epoch, phase, metrics, saved_info)

        return saved_info

    def _log_to_history(self, epoch, phase, metrics, saved_info):
        """Log checkpoint event to CSV history file."""
        with open(self.history_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                phase,
                metrics.get('total_loss', ''),
                metrics.get('loss_int', ''),
                metrics.get('loss_bdry', ''),
                metrics.get('l2_error', ''),
                metrics.get('l2_relative_error', ''),
                metrics.get('h1_error', ''),
                saved_info['best_l2_saved'],
                saved_info['best_h1_saved'],
                datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            ])

    def load_best_l2_model(self):
        """Load the model with the best L2 error."""
        if os.path.exists(self.best_l2_path):
            return torch.load(self.best_l2_path, weights_only=False)
        else:
            raise FileNotFoundError(f"Best L2 model not found at {self.best_l2_path}")

    def load_best_h1_model(self):
        """Load the model with the best H1 error."""
        if os.path.exists(self.best_h1_path):
            return torch.load(self.best_h1_path, weights_only=False)
        else:
            raise FileNotFoundError(f"Best H1 model not found at {self.best_h1_path}")

    def load_last_model(self):
        """Load the most recent model checkpoint."""
        if os.path.exists(self.last_path):
            return torch.load(self.last_path, weights_only=False)
        else:
            raise FileNotFoundError(f"Last model not found at {self.last_path}")

    def get_summary(self):
        """Get summary of best checkpoints."""
        return {
            'best_l2_error': self.best_l2_error,
            'best_l2_relative_error': self.best_l2_relative_error,
            'best_h1_error': self.best_h1_error,
            'checkpoint_dir': self.checkpoint_dir
        }

    def print_summary(self):
        """Print summary of best checkpoints."""
        print("\n" + "=" * 80)
        print("CHECKPOINT SUMMARY")
        print("=" * 80)
        print(f"Best L2 Error:          {self.best_l2_error:.8f}")
        print(f"Best L2 Relative Error: {self.best_l2_relative_error:.8f}")
        print(f"Best H1 Error:          {self.best_h1_error:.8f}")
        print(f"Checkpoint Directory:   {self.checkpoint_dir}")
        print("=" * 80)
