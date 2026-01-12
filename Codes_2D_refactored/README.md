# Refactored DRM Solver for 2D Fourth-Order PDEs

This is a refactored and improved version of the Deep Ritz Method (DRM) solver for solving 2D fourth-order PDE problems.

## Key Improvements

### 1. **Best Model Checkpointing**
- **Previous**: Only saved model at the end of training
- **Now**: Automatically saves best models based on validation metrics:
  - `best_l2_model.pt` - Model with lowest L2 error
  - `best_h1_model.pt` - Model with lowest H1 seminorm error
  - `last_model.pt` - Most recent checkpoint
  - `checkpoint_history.csv` - Complete training history

### 2. **Enhanced Validation Metrics**
- Computes L2, H1, and H2 errors during training
- Tracks both absolute and relative errors
- Validation performed at configurable frequency

### 3. **Configuration-Driven**
- All hyperparameters in YAML config files
- Easy to experiment with different settings
- No need to modify code for different problems

### 4. **Clean Code Organization**
```
Codes_2D_refactored/
├── config/                    # Configuration files
│   ├── p1_drm_config.yaml
│   ├── p2_drm_config.yaml
│   ├── p3_drm_config.yaml
│   └── p4_drm_config.yaml
├── src/                       # Source code
│   ├── models/                # Neural network models
│   ├── solvers/               # Unified solver
│   ├── pde/                   # PDE formulations (P1-P4)
│   ├── data/                  # Data loading
│   ├── validation/            # Validation metrics
│   └── utils/                 # Utilities (checkpoint, plotting, etc.)
├── experiments/               # Experiment results
│   ├── P1/DRM/results/
│   ├── P2/DRM/results/
│   └── ...
└── train.py                   # Main entry point
```

### 5. **Unified Solver**
- Single solver implementation for all problems
- Two-phase training (Adam → LBFGS)
- Automatic learning rate scheduling
- Progress tracking and logging

## Quick Start

### Installation

```bash
# Install dependencies
pip install torch numpy scipy matplotlib pyyaml sympy
```

### Training a Model

```bash
# Train Problem 1
python train.py --config config/p1_drm_config.yaml

# Train Problem 2
python train.py --config config/p2_drm_config.yaml

# Train Problem 3
python train.py --config config/p3_drm_config.yaml

# Train Problem 4
python train.py --config config/p4_drm_config.yaml

# Use GPU (if available)
python train.py --config config/p1_drm_config.yaml --gpu

# Resume from checkpoint
python train.py --config config/p1_drm_config.yaml --resume
```

## Configuration File Structure

```yaml
# Example: config/p1_drm_config.yaml
problem: P1
method: DRM

# Model architecture
model:
  input_dim: 2
  hidden_dim: 50
  num_layers: 4
  output_dim: 1

# Data configuration
data:
  dataset_dir: ../Codes_2D/P1/DRM/dataset
  dataname: 20000pts
  batch_size: 2000

# Training configuration
training:
  adam_epochs: 10000
  lbfgs_iterations: 10000
  bw_dir: 4000.0
  bw_neu: 4000.0

# Optimizer configuration
optimizer:
  adam:
    lr: 0.0001
    scheduler: plateau
    lr_patience: 500
  lbfgs:
    line_search_fn: strong_wolfe
    max_iter: 20

# Validation configuration
validation:
  frequency: 100        # Validate every 100 epochs
  resolution: 50        # Grid resolution

# Checkpoint configuration
checkpoint:
  save_frequency: 100   # Save every 100 epochs

# Results
results_dir: Codes_2D_refactored/experiments/P1/DRM/results
save_plots: true
```

## Output Structure

After training, you'll find:

```
experiments/P1/DRM/results/
├── checkpoints/
│   ├── best_l2_model.pt          # Best model by L2 error
│   ├── best_h1_model.pt          # Best model by H1 error
│   ├── last_model.pt             # Latest checkpoint
│   └── checkpoint_history.csv    # Complete training history
├── y_plot/                       # Solution plots
│   ├── epoch0.png
│   ├── epoch100.png
│   └── ...
├── y.pt                          # Final model
└── loss.pkl                      # Loss history
```

## Loading Best Models

```python
import torch

# Load the best L2 model
model = torch.load('experiments/P1/DRM/results/checkpoints/best_l2_model.pt')

# Load the best H1 model
model = torch.load('experiments/P1/DRM/results/checkpoints/best_h1_model.pt')

# Use the model
import numpy as np
from torch.autograd import Variable

x = Variable(torch.tensor([[0.5]]).float(), requires_grad=False)
y = Variable(torch.tensor([[0.5]]).float(), requires_grad=False)
output = model(x, y)
```

## Checkpoint History CSV

The `checkpoint_history.csv` file tracks all validation checkpoints:

| epoch | phase | total_loss | loss_int | loss_bdry | l2_error | l2_relative_error | h1_error | best_l2_saved | best_h1_saved | timestamp |
|-------|-------|------------|----------|-----------|----------|-------------------|----------|---------------|---------------|-----------|
| 0     | adam  | 0.125      | 0.100    | 0.025     | 0.0534   | 0.0123            | 0.234    | True          | True          | ...       |
| 100   | adam  | 0.098      | 0.075    | 0.023     | 0.0421   | 0.0097            | 0.189    | True          | True          | ...       |
| ...   | ...   | ...        | ...      | ...       | ...      | ...               | ...      | ...           | ...           | ...       |

## Customization

### Adding a New Problem (P5)

1. Create PDE class in `src/pde/p5_pde.py`
2. Update `src/pde/__init__.py` to include P5
3. Create ground truth function
4. Create config file `config/p5_drm_config.yaml`
5. Train: `python train.py --config config/p5_drm_config.yaml`

### Modifying Network Architecture

Edit the config file:
```yaml
model:
  hidden_dim: 100    # Change from 50 to 100
  num_layers: 6      # Change from 4 to 6
```

### Changing Training Parameters

Edit the config file:
```yaml
training:
  adam_epochs: 15000       # Increase epochs
  lbfgs_iterations: 5000   # Decrease LBFGS iterations

optimizer:
  adam:
    lr: 0.0005            # Increase learning rate
```

## Comparison: Old vs New

| Feature | Old Code | New Code |
|---------|----------|----------|
| Model Saving | Only last epoch | Best L2, Best H1, Last |
| Validation | L2 error only | L2, H1, H2 errors |
| Code Organization | Duplicated across P1-P4 | Single unified codebase |
| Configuration | Hardcoded in files | YAML config files |
| Checkpointing | Manual | Automatic with history |
| Training History | Loss only | Full metrics + CSV log |

## Dependencies

- Python 3.7+
- PyTorch 1.8+
- NumPy
- SciPy
- Matplotlib
- PyYAML
- SymPy

## License

Same as the original project.

## Author

Refactored version created for improved research workflow and reproducibility.
