import torch
from torch.autograd import Variable


def from_numpy_to_tensor(numpys, require_grads, dtype=torch.float32):
    """
    Convert numpy arrays to PyTorch tensors with gradient tracking.

    Args:
        numpys: List of numpy arrays
        require_grads: List of booleans indicating whether to track gradients
        dtype: PyTorch data type (default: torch.float32)

    Returns:
        List of PyTorch tensors
    """
    outputs = []
    for ind in range(len(numpys)):
        outputs.append(
            Variable(torch.from_numpy(numpys[ind]), requires_grad=require_grads[ind]).type(dtype)
        )
    return outputs
