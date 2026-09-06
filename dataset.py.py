import torch
from torch.utils.data import Dataset
import numpy as np


class RockFractureDataset(Dataset):
    """
    Dataset for rock fracture propagation data
    
Dataset Availability:

Due to the large size of the datasets and the limited space in this GitHub repository. The datasets have been uploaded to Google Drive and can be accessed through the links provided below.

Link 1:
https://drive.google.com/file/d/1lJKwwnG-EYU3qx5adI1A-4MVk-7lNQMS/view?usp=sharing
Link 2:
https://drive.google.com/file/d/1e7IPJIIRosHwfQM_AO5cQiYZs71ToSLN/view?usp=sharing
    """
    
    def __init__(self, ct_images, displacement, phase_field, transform=None):
        self.ct_images = ct_images
        self.displacement = displacement
        self.phase_field = phase_field
        self.transform = transform
        
        # Validate shapes
        assert len(self.ct_images) == len(self.displacement) == len(self.phase_field), \
            "All inputs must have the same length"
    
    def __len__(self):
        return len(self.ct_images)
    
    def __getitem__(self, index):
        ct = self.ct_images[index]
        u = self.displacement[index]
        phi = self.phase_field[index]
        
        # Convert to torch tensors
        ct_tensor = torch.tensor(ct, dtype=torch.float32)
        u_tensor = torch.tensor(u, dtype=torch.float32)
        phi_tensor = torch.tensor(phi, dtype=torch.float32)
        
        # Ensure correct dimensions
        if len(ct_tensor.shape) == 2:
            ct_tensor = ct_tensor.unsqueeze(0)
        if len(phi_tensor.shape) == 2:
            phi_tensor = phi_tensor.unsqueeze(0)
        if len(u_tensor.shape) == 2:
            u_tensor = u_tensor.unsqueeze(0)
        
        # Apply transform
        if self.transform:
            ct_tensor, u_tensor, phi_tensor = self.transform(
                ct_tensor, u_tensor, phi_tensor
            )
        
        return {
            "input": ct_tensor,
            "u": u_tensor,
            "phi": phi_tensor
        }


def create_synthetic_dataset(n_samples=100, img_size=256):
    """
    Create synthetic dataset for testing
    
    Returns:
        ct_images: Random noise images
        displacements: Random displacement fields
        phase_fields: Random phase fields
    """
    ct_images = np.random.randn(n_samples, 1, img_size, img_size).astype(np.float32)
    displacements = np.random.randn(n_samples, 2, img_size, img_size).astype(np.float32)
    phase_fields = np.random.rand(n_samples, 1, img_size, img_size).astype(np.float32)
    
    return ct_images, displacements, phase_fields
