import numpy as np
import torch
from sklearn.metrics import mean_squared_error, r2_score
from utils import compute_iou, compute_pixel_accuracy

def evaluate_model(model, test_loader, device):
    """
    Comprehensive model evaluation
    """
    model.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            inputs = batch['input'].to(device)
            phi_true = batch['phi'].to(device)
            
            phi_pred = model(inputs)
            
            all_preds.append(phi_pred.cpu().numpy())
            all_targets.append(phi_true.cpu().numpy())
    
    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    
    # Flatten for metrics
    pred_flat = preds.flatten()
    target_flat = targets.flatten()
    
    # Compute metrics
    metrics = {
        'rmse': RMSE(target_flat, pred_flat),
        'r2': R2(target_flat, pred_flat),
        'iou': compute_iou(torch.from_numpy(preds), torch.from_numpy(targets)),
        'pixel_accuracy': compute_pixel_accuracy(torch.from_numpy(preds), torch.from_numpy(targets))
    }
    
    return metrics


def evaluate_crack_geometry(pred, target):
    """
    Evaluate crack geometry metrics (Eq. 20-23)
    
    Returns:
        E_i: Initiation location error (mm)
        E_L: Crack length error (%)
        E_theta: Propagation angle error (°)
        E_P: Mean crack-path deviation (mm)
    """
    # Convert to binary
    pred_binary = (pred > 0.5).float()
    target_binary = (target > 0.5).float()
    
    # Find crack pixels
    pred_points = torch.nonzero(pred_binary)
    target_points = torch.nonzero(target_binary)
    
    if len(pred_points) == 0 or len(target_points) == 0:
        return {'E_i': float('inf'), 'E_L': float('inf'), 
                'E_theta': float('inf'), 'E_P': float('inf')}
    
    # Crack initiation location (first crack pixel from top)
    pred_init = pred_points[pred_points[:, 2].argmin()]  # Min y coordinate
    target_init = target_points[target_points[:, 2].argmin()]
    
    # Initiation error (E_i)
    E_i = torch.sqrt(torch.sum((pred_init - target_init) ** 2)).item()
    
    # Crack length error (E_L)
    pred_length = len(pred_points)
    target_length = len(target_points)
    E_L = abs(pred_length - target_length) / (target_length + 1e-8) * 100
    
    # Crack propagation angle (approximate)
    # Use principal component analysis of crack points
    if len(pred_points) > 1:
        pred_centroid = pred_points.mean(dim=0)
        pred_centered = pred_points - pred_centroid
        pred_cov = pred_centered.T @ pred_centered / (len(pred_points) - 1)
        pred_eigvals, pred_eigvecs = torch.linalg.eigh(pred_cov)
        pred_angle = torch.atan2(pred_eigvecs[0, 1], pred_eigvecs[0, 0]).item()
    else:
        pred_angle = 0
    
    if len(target_points) > 1:
        target_centroid = target_points.mean(dim=0)
        target_centered = target_points - target_centroid
        target_cov = target_centered.T @ target_centered / (len(target_points) - 1)
        target_eigvals, target_eigvecs = torch.linalg.eigh(target_cov)
        target_angle = torch.atan2(target_eigvecs[0, 1], target_eigvecs[0, 0]).item()
    else:
        target_angle = 0
    
    # Angle error (E_theta) in degrees
    E_theta = abs(pred_angle - target_angle) * 180 / np.pi
    
    # Mean crack-path deviation (E_P)
    # Minimum distance from predicted points to target path
    if len(pred_points) > 0 and len(target_points) > 0:
        distances = torch.cdist(pred_points.float(), target_points.float())
        min_distances = distances.min(dim=1)[0]
        E_P = min_distances.mean().item()
    else:
        E_P = float('inf')
    
    return {
        'E_i': E_i,
        'E_L': E_L,
        'E_theta': E_theta,
        'E_P': E_P
    }
