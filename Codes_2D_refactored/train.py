#!/usr/bin/env python
"""
Main training script for DRM solver.

Usage:
    python train.py --config config/p1_drm_config.yaml
    python train.py --config config/p2_drm_config.yaml --gpu
"""

import os
import sys
import argparse
import yaml
import torch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.solvers import DRMSolver


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_ground_truth_function(problem_name):
    """
    Get ground truth generation function for the specified problem.

    Args:
        problem_name: Name of the problem ('P1', 'P2', 'P3', or 'P4')

    Returns:
        Ground truth function
    """
    # Import the appropriate ground truth module
    # For now, we'll need to copy the g_tr.py files from the original code
    # or import them dynamically

    if problem_name == 'P1':
        # You'll need to copy Codes_2D/P1/DRM/g_tr.py to a common location
        # or import it directly
        sys.path.insert(0, '../Codes_2D/P1/DRM')
        import g_tr
        return g_tr.error_data_gen_interior
    elif problem_name == 'P2':
        sys.path.insert(0, '../Codes_2D/P2/DRM')
        import g_tr
        return g_tr.error_data_gen_interior
    elif problem_name == 'P3':
        sys.path.insert(0, '../Codes_2D/P3/DRM')
        import g_tr
        return g_tr.error_data_gen_interior
    elif problem_name == 'P4':
        sys.path.insert(0, '../Codes_2D/P4/DRM')
        import g_tr
        return g_tr.error_data_gen_interior
    else:
        raise ValueError(f"Unknown problem: {problem_name}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train DRM solver for 2D fourth-order PDEs')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to configuration YAML file')
    parser.add_argument('--gpu', action='store_true',
                        help='Use GPU if available')
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from checkpoint')

    args = parser.parse_args()

    # Load configuration
    print(f"Loading configuration from {args.config}")
    config = load_config(args.config)

    # Override config with command line arguments
    if args.gpu:
        config['use_cuda'] = True
    if args.resume:
        config['resume_training'] = True

    # Print configuration
    print("\n" + "=" * 80)
    print("CONFIGURATION")
    print("=" * 80)
    print(f"Problem: {config['problem']}")
    print(f"Method: {config['method']}")
    print(f"Dataset: {config['data']['dataname']}")
    print(f"Adam epochs: {config['training']['adam_epochs']}")
    print(f"LBFGS iterations: {config['training']['lbfgs_iterations']}")
    print(f"Results directory: {config['results_dir']}")
    print("=" * 80 + "\n")

    # Get ground truth function
    ground_truth_func = get_ground_truth_function(config['problem'])

    # Initialize solver
    print("Initializing solver...")
    solver = DRMSolver(config, ground_truth_func)

    # Train
    print("\nStarting training...")
    solver.train()

    print("\n" + "=" * 80)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"\nResults saved to: {config['results_dir']}")
    print(f"Best models saved in: {os.path.join(config['results_dir'], 'checkpoints')}")
    print("\nCheckpoint files:")
    print("  - best_l2_model.pt     : Model with best L2 error")
    print("  - best_h1_model.pt     : Model with best H1 error")
    print("  - last_model.pt        : Most recent model")
    print("  - checkpoint_history.csv : Training history")
    print("\n")


if __name__ == '__main__':
    # Set environment variables
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    # Set default tensor type
    torch.set_default_dtype(torch.float32)

    # Run main
    main()
