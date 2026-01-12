import torch
import numpy as np

class NN(torch.nn.Module):
    """
    Neural network for solving 2D PDEs.
    Default architecture: [2, 50, 50, 50, 50, 1] with tanh activation.
    """
    def __init__(self, input_dim=2, hidden_dim=50, num_layers=4, output_dim=1):
        super(NN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim

        # Input layer
        self.L1 = torch.nn.Linear(input_dim, hidden_dim)

        # Hidden layers
        self.hidden_layers = torch.nn.ModuleList([
            torch.nn.Linear(hidden_dim, hidden_dim)
            for _ in range(num_layers - 1)
        ])

        # Output layer
        self.output_layer = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x, y):
        """
        Forward pass through the network.

        Args:
            x: First spatial coordinate (tensor)
            y: Second spatial coordinate (tensor)

        Returns:
            Network output (tensor)
        """
        inputs = torch.cat([x, y], axis=1)

        # Input layer with activation
        out = torch.tanh(self.L1(inputs))

        # Hidden layers with activation
        for layer in self.hidden_layers:
            out = torch.tanh(layer(out))

        # Output layer (no activation)
        out = self.output_layer(out)

        return out


def init_weights(m):
    """
    Initialize network weights using Xavier uniform initialization.

    Args:
        m: PyTorch module
    """
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.0)
