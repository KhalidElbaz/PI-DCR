import torch
from utils import gradient_2d


def strain_energy(u, E, nu):
    """
    Compute elastic strain energy density for 2D plane strain
    
    Args:
        u: Displacement field [batch, 2, H, W]
        E: Young's modulus
        nu: Poisson's ratio
    
    Returns:
        strain_energy_density: [batch, 1, H, W]
    """
    # Get displacement components
    ux = u[:, 0:1, :, :]
    uy = u[:, 1:2, :, :]
    
    # Compute gradients
    ux_x, ux_y = gradient_2d(ux)
    uy_x, uy_y = gradient_2d(uy)
    
    # Strain components
    exx = ux_x
    eyy = uy_y
    exy = 0.5 * (ux_y + uy_x)
    
    # Lame parameters (plane strain)
    mu = E / (2 * (1 + nu))
    lamb = E * nu / ((1 + nu) * (1 - 2 * nu))
    
    # Volumetric and deviatoric strain energy
    trace = exx + eyy
    strain_energy_density = 0.5 * lamb * trace**2 + mu * (exx**2 + eyy**2 + 2 * exy**2)
    
    return strain_energy_density


def stress_tensor(u, E, nu):
    """
    Compute Cauchy stress tensor for 2D plane strain
    
    Args:
        u: Displacement field [batch, 2, H, W]
        E: Young's modulus
        nu: Poisson's ratio
    
    Returns:
        stress: [batch, 3, H, W] where 3 components: sigma_xx, sigma_yy, sigma_xy
    """
    ux = u[:, 0:1, :, :]
    uy = u[:, 1:2, :, :]
    
    ux_x, ux_y = gradient_2d(ux)
    uy_x, uy_y = gradient_2d(uy)
    
    exx = ux_x
    eyy = uy_y
    exy = 0.5 * (ux_y + uy_x)
    
    mu = E / (2 * (1 + nu))
    lamb = E * nu / ((1 + nu) * (1 - 2 * nu))
    
    trace = exx + eyy
    
    sigma_xx = 2 * mu * exx + lamb * trace
    sigma_yy = 2 * mu * eyy + lamb * trace
    sigma_xy = 2 * mu * exy
    
    return torch.stack([sigma_xx, sigma_yy, sigma_xy], dim=1)


def equilibrium_residual(stress):
    """
    Quasi-static equilibrium residual: ∇·σ + f = 0
    
    Args:
        stress: Stress tensor [batch, 3, H, W] (sigma_xx, sigma_yy, sigma_xy)
    """
    sigma_xx = stress[:, 0:1, :, :]
    sigma_yy = stress[:, 1:2, :, :]
    sigma_xy = stress[:, 2:3, :, :]
    
    # Divergence of stress tensor
    d_sigma_xx_x, _ = gradient_2d(sigma_xx)
    _, d_sigma_xy_y = gradient_2d(sigma_xy)
    _, d_sigma_yy_y = gradient_2d(sigma_yy)
    d_sigma_xy_x, _ = gradient_2d(sigma_xy)
    
    # Divergence components
    div_x = d_sigma_xx_x + d_sigma_xy_y
    div_y = d_sigma_xy_x + d_sigma_yy_y
    
    return torch.stack([div_x, div_y], dim=1)