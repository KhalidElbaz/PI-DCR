import torch
import numpy as np
from scipy.ndimage import gaussian_filter


def normalize_data(data, min_val=None, max_val=None):
    """
    Normalize data to [-1, 1] range (Eq. 15)
    """
    if min_val is None:
        min_val = data.min()
    if max_val is None:
        max_val = data.max()
    
    # Avoid division by zero
    if max_val - min_val == 0:
        return data, min_val, max_val
    
    normalized = 2 * (data - min_val) / (max_val - min_val) - 1
    return normalized, min_val, max_val


def denormalize_data(normalized, min_val, max_val):
    """
    Denormalize data from [-1, 1] range
    """
    return (normalized + 1) / 2 * (max_val - min_val) + min_val


def create_phase_field_from_crack(crack_image, sigma=2.0):
    """
    Convert binary crack image to phase field (smooth diffuse crack)
    """
    if isinstance(crack_image, torch.Tensor):
        crack_image = crack_image.cpu().numpy()
    
    phase_field = gaussian_filter(crack_image.astype(np.float32), sigma=sigma)
    phase_field = np.clip(phase_field, 0, 1)
    return phase_field


def compute_iou(pred, target, threshold=0.5):
    """
    Compute Intersection over Union
    """
    pred_binary = (pred > threshold).float()
    target_binary = (target > threshold).float()
    
    intersection = (pred_binary * target_binary).sum()
    union = pred_binary.sum() + target_binary.sum() - intersection
    
    if union == 0:
        return torch.tensor(1.0)
    
    return (intersection / (union + 1e-8)).item()


def compute_pixel_accuracy(pred, target, threshold=0.5):
    """
    Compute Pixel Accuracy
    """
    pred_binary = (pred > threshold).float()
    target_binary = (target > threshold).float()
    
    correct = (pred_binary == target_binary).float().sum()
    total = pred.numel()
    
    return (correct / total).item()


def gradient_2d(x):
    """
    Compute 2D gradients using central difference
    """
    # Forward difference for boundaries
    grad_x = torch.zeros_like(x)
    grad_y = torch.zeros_like(x)
    
    # Central difference
    grad_x[:, :, 1:-1, :] = (x[:, :, 2:, :] - x[:, :, :-2, :]) / 2
    grad_x[:, :, 0, :] = x[:, :, 1, :] - x[:, :, 0, :]
    grad_x[:, :, -1, :] = x[:, :, -1, :] - x[:, :, -2, :]
    
    grad_y[:, :, :, 1:-1] = (x[:, :, :, 2:] - x[:, :, :, :-2]) / 2
    grad_y[:, :, :, 0] = x[:, :, :, 1] - x[:, :, :, 0]
    grad_y[:, :, :, -1] = x[:, :, :, -1] - x[:, :, :, -2]
    
    return grad_x, grad_y