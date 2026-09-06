import torch
from phase_field import total_energy, crack_surface_energy
from elasticity import strain_energy, stress_tensor, equilibrium_residual


def deep_ritz_loss(phi, u, E, nu, Gc, lc, loss_weights):
    """
    Deep Ritz Method energy minimization
    
    Minimizes total potential energy: Pi(u, phi) = ∫[g(phi)*Psi_e + fracture_surface] dΩ
    
    Args:
        phi: Phase field [batch, 1, H, W]
        u: Displacement field [batch, 2, H, W]
        E: Young's modulus
        nu: Poisson's ratio
        Gc: Fracture toughness
        lc: Length scale
        loss_weights: Tuple of (w1, w2, w3, w4)
    """
    w1, w2, w3, w4 = loss_weights
    
    # Compute elastic strain energy
    elastic_energy = strain_energy(u, E, nu)
    
    # Compute total potential energy (Eq. 3)
    potential_energy = total_energy(phi, elastic_energy, Gc, lc)
    
    # Compute phase field residual (Eq. 3b)
    H = elastic_energy.detach()  # History field for irreversibility
    phase_residual = phase_field_residual(phi, Gc, lc, H)
    
    # Compute equilibrium residual (Eq. 5)
    stress = stress_tensor(u, E, nu)
    eq_residual = equilibrium_residual(stress)
    
    # Energy-based loss (Ritz method)
    loss_energy = w1 * potential_energy
    
    # Physics residual losses
    loss_phase = w2 * torch.mean(phase_residual**2)
    loss_equilibrium = w3 * torch.mean(eq_residual**2)
    
    # Total Ritz loss
    total_loss = loss_energy + loss_phase + loss_equilibrium
    
    return total_loss, {
        'energy_loss': loss_energy.item(),
        'phase_residual_loss': loss_phase.item(),
        'equilibrium_loss': loss_equilibrium.item()
    }


def phase_field_residual(phi, Gc, lc, H):
    """
    Phase field equilibrium residual (Eq. 3b)
    """
    from utils import gradient_2d
    
    grad_x, grad_y = gradient_2d(phi)
    
    # Laplacian approximation
    laplacian_x = torch.zeros_like(phi)
    laplacian_y = torch.zeros_like(phi)
    
    laplacian_x[:, :, 1:-1, :] = phi[:, :, 2:, :] - 2 * phi[:, :, 1:-1, :] + phi[:, :, :-2, :]
    laplacian_y[:, :, :, 1:-1] = phi[:, :, :, 2:] - 2 * phi[:, :, :, 1:-1] + phi[:, :, :, :-2]
    
    laplacian = laplacian_x + laplacian_y
    
    residual = (Gc / lc) * phi - Gc * lc * laplacian - 2 * (1 - phi) * H
    
    return residual