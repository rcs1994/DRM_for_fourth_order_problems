import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


def plot_2D_solution(net, resolution, path, title="Solution"):
    """
    Plot 2D solution as a heatmap.

    Args:
        net: Neural network model
        resolution: Grid resolution
        path: Path to save the figure
        title: Plot title
    """
    import torch
    from torch.autograd import Variable

    # Get device from model
    device = next(net.parameters()).device

    # Create grid
    val_x1 = np.arange(0, 1, 1/resolution).reshape(-1, 1)
    val_x2 = np.arange(0, 1, 1/resolution).reshape(-1, 1)
    val_ms_x1, val_ms_x2 = np.meshgrid(val_x1, val_x2)

    plot_val_x1 = np.ravel(val_ms_x1).reshape(-1, 1)
    plot_val_x2 = np.ravel(val_ms_x2).reshape(-1, 1)

    t_val_vx1 = Variable(torch.from_numpy(plot_val_x1).float().to(device), requires_grad=False)
    t_val_vx2 = Variable(torch.from_numpy(plot_val_x2).float().to(device), requires_grad=False)

    # Evaluate network
    with torch.no_grad():
        data = net(t_val_vx1, t_val_vx2).cpu().numpy().reshape([resolution, resolution])

    # Plot
    fig = plt.figure(figsize=(6, 5))
    plt.pcolor(val_ms_x1, val_ms_x2, data, cmap='jet', shading='auto')
    h = plt.colorbar()
    h.ax.tick_params(labelsize=20)
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    plt.savefig(path, bbox_inches='tight')
    plt.close()


def plot_3D_solution(net, resolution, path, title="Solution"):
    """
    Plot 3D surface plot of solution.

    Args:
        net: Neural network model
        resolution: Grid resolution
        path: Path to save the figure
        title: Plot title
    """
    import torch
    from torch.autograd import Variable

    # Get device from model
    device = next(net.parameters()).device

    # Create grid
    val_x1 = np.arange(0, 1, 1/resolution).reshape(-1, 1)
    val_x2 = np.arange(0, 1, 1/resolution).reshape(-1, 1)
    val_ms_x1, val_ms_x2 = np.meshgrid(val_x1, val_x2)

    plot_val_x1 = np.ravel(val_ms_x1).reshape(-1, 1)
    plot_val_x2 = np.ravel(val_ms_x2).reshape(-1, 1)

    t_val_vx1 = Variable(torch.from_numpy(plot_val_x1).float().to(device), requires_grad=False)
    t_val_vx2 = Variable(torch.from_numpy(plot_val_x2).float().to(device), requires_grad=False)

    # Evaluate network
    with torch.no_grad():
        data = net(t_val_vx1, t_val_vx2).cpu().numpy().reshape([resolution, resolution])

    # Plot
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(val_ms_x1, val_ms_x2, data, cmap='jet', edgecolor='none')

    h = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
    h.ax.tick_params(labelsize=20)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_title(title)

    plt.savefig(path, bbox_inches='tight')
    plt.close()
