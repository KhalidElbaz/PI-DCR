import torch
from utils import gradient_2d


def degradation(phi):
    """
    Degradation function: g(phi) = (1 - phi)^2
    """
    return (1 - phi) ** 2


def crack_surface_energy(phi, Gc, lc):
    """
    Compute crack surface energy density (Eq. 2)
    
    Args:
        phi: Phase field variable [batch, 1, H, W]
        Gc: Fracture toughness
        lc: Length scale parameter
    """
    # Compute gradients
    grad_x, grad_y = gradient_2d(phi)
    grad_squared = grad_x**2 + grad_y**2
    
    # Crack surface energy density
    energy_density = (Gc / (2 * lc)) * (phi**2 + lc**2 * grad_squared)
    
    return energy_density.mean()


def total_energy(phi, elastic_energy, Gc, lc):
    """
    Compute total potential energy (Eq. 3)
    
    Pi(u, phi) = ∫[g(phi) * Psi_e + Gc/(2*lc)*(phi^2 + lc^2*|∇phi|^2)] dΩ
    """
    # Degraded elastic energy
    degraded_elastic = degradation(phi) * elastic_energy
    
    # Crack surface energy
    fracture_energy = crack_surface_energy(phi, Gc, lc) * phi.numel()  # Scale by domain size
    
    return (degraded_elastic.mean() + fracture_energy)


def phase_field_residual(phi, Gc, lc, H):
    """
    Phase field equilibrium residual (Eq. 3b)
    
    Gc/lc * phi - Gc * lc * ∇^2 phi - 2 * (1 - phi) * H = 0
    """
    grad_x, grad_y = gradient_2d(phi)
    
    # Compute Laplacian (second derivative approximation)
    laplacian_x = torch.zeros_like(phi)
    laplacian_y = torch.zeros_like(phi)
    
    laplacian_x[:, :, 1:-1, :] = phi[:, :, 2:, :] - 2 * phi[:, :, 1:-1, :] + phi[:, :, :-2, :]
    laplacian_y[:, :, :, 1:-1] = phi[:, :, :, 2:] - 2 * phi[:, :, :, 1:-1] + phi[:, :, :, :-2]
    
    laplacian = laplacian_x + laplacian_y
    
    residual = (Gc / lc) * phi - Gc * lc * laplacian - 2 * (1 - phi) * H
    
    return residual


def compute_history_field(phi, elastic_energy, previous_H=None):
    """
    Compute history field for crack irreversibility
    """
    # H = max(previous_H, elastic_energy)
    if previous_H is not None:
        H = torch.max(previous_H, elastic_energy)
    else:
        H = elastic_energy
    return H