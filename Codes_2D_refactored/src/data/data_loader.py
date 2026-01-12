import os
import pickle as pkl
import numpy as np
from ..utils.tools import from_numpy_to_tensor
import torch


class DataLoader:
    """
    Load training data for DRM solver.
    """

    def __init__(self, dataset_dir, dataname, dtype=torch.float32):
        """
        Initialize data loader.

        Args:
            dataset_dir: Directory containing dataset files
            dataname: Name of the dataset (e.g., '20000pts')
            dtype: PyTorch data type
        """
        self.dataset_dir = dataset_dir
        self.dataname = dataname
        self.dtype = dtype

        # Load collocation points
        self._load_collocation_data()

        # Load ground truth and boundary data
        self._load_ground_truth_data()

    def _load_collocation_data(self):
        """Load interior and boundary collocation points."""
        collocation_path = os.path.join(self.dataset_dir, self.dataname)

        with open(collocation_path, 'rb') as pfile:
            int_col = pkl.load(pfile)
            bdry_col = pkl.load(pfile)
            normal_vec = pkl.load(pfile)

        print(f"Loaded collocation data: interior={int_col.shape}, boundary={bdry_col.shape}, normals={normal_vec.shape}")

        # Split into x1 and x2 components
        self.intx1, self.intx2 = np.split(int_col, 2, axis=1)
        self.bdx1, self.bdx2 = np.split(bdry_col, 2, axis=1)
        self.nx1, self.nx2 = np.split(normal_vec, 2, axis=1)

        # Convert to tensors
        self.tintx1, self.tintx2, self.tbdx1, self.tbdx2, self.tnx1, self.tnx2 = from_numpy_to_tensor(
            [self.intx1, self.intx2, self.bdx1, self.bdx2, self.nx1, self.nx2],
            [True, True, False, False, True, True],
            dtype=self.dtype
        )

    def _load_ground_truth_data(self):
        """Load ground truth solution and boundary data."""
        gt_path = os.path.join(self.dataset_dir, f"gt_on_{self.dataname}")

        with open(gt_path, 'rb') as pfile:
            y_gt = pkl.load(pfile)
            f_np = pkl.load(pfile)
            dirichlet_data_np = pkl.load(pfile)
            neumann_data_np = pkl.load(pfile)

        print(f"Loaded ground truth data: y_gt={y_gt.shape}, f={f_np.shape}")

        # Convert to tensors
        self.f, self.bdrydata_dirichlet, self.bdrydata_neumann, self.ygt = from_numpy_to_tensor(
            [f_np, dirichlet_data_np, neumann_data_np, y_gt],
            [False, False, True, False],
            dtype=self.dtype
        )

    def get_training_data(self):
        """
        Get all training data.

        Returns:
            Dictionary containing all training data tensors
        """
        return {
            'intx1': self.tintx1,
            'intx2': self.tintx2,
            'bdx1': self.tbdx1,
            'bdx2': self.tbdx2,
            'nx1': self.tnx1,
            'nx2': self.tnx2,
            'f': self.f,
            'bdrydata_dirichlet': self.bdrydata_dirichlet,
            'bdrydata_neumann': self.bdrydata_neumann,
            'ygt': self.ygt
        }

    def create_data_loader(self, batch_size=2000, shuffle=True):
        """
        Create PyTorch DataLoader for batched training.

        Args:
            batch_size: Batch size
            shuffle: Whether to shuffle data

        Returns:
            PyTorch DataLoader
        """
        return torch.utils.data.DataLoader(
            [self.intx1, self.intx2],
            batch_size=batch_size,
            shuffle=shuffle
        )
