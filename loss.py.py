import torch
import torch.nn.functional as F
from phase_field import crack_surface_energy, degradation
from elasticity import strain_energy, equilibrium_residual, stress_tensor


class PIDCRLoss(torch.nn.Module):
    """
    Complete Physics-Informed Loss Function (Eq. 9-13)
    
    L_total = w1*L_elasticity + w2*L_phase + w3*L_BC + w4*L_data
    """
    
    def __init__(self, E, nu, Gc, lc, weights):
        super(PIDCRLoss, self).__init__()
        self.E = E
        self.nu = nu
        self.Gc = Gc
        self.lc = lc
        self.w1, self.w2, self.w3, self.w4 = weights
        
    def forward(self, pred_phi, true_phi, pred_u, true_u, boundary_mask=None):
        """
        Compute total loss with physics constraints
        
        Args:
            pred_phi: Predicted phase field [batch, 1, H, W]
            true_phi: Ground truth phase field [batch, 1, H, W]
            pred_u: Predicted displacement [batch, 2, H, W]
            true_u: Ground truth displacement [batch, 2, H, W]
            boundary_mask: Mask for boundary conditions [batch, 1, H, W]
        """
        # ===== L_data: Data loss (Eq. 13) =====
        loss_phase_data = F.mse_loss(pred_phi, true_phi)
        loss_displacement_data = F.mse_loss(pred_u, true_u)
        loss_data = loss_phase_data + loss_displacement_data
        
        # ===== L_phase: Phase field physics loss (Eq. 11) =====
        # Compute gradient for phase field
        from utils import gradient_2d
        
        # Elastic strain energy from displacement
        elastic_energy = strain_energy(pred_u, self.E, self.nu)
        
        # History field for crack irreversibility
        H = elastic_energy.detach()
        
        # Phase field residual (Eq. 3b)
        grad_x, grad_y = gradient_2d(pred_phi)
        laplacian_x = torch.zeros_like(pred_phi)
        laplacian_y = torch.zeros_like(pred_phi)
        
        laplacian_x[:, :, 1:-1, :] = pred_phi[:, :, 2:, :] - 2 * pred_phi[:, :, 1:-1, :] + pred_phi[:, :, :-2, :]
        laplacian_y[:, :, :, 1:-1] = pred_phi[:, :, :, 2:] - 2 * pred_phi[:, :, :, 1:-1] + pred_phi[:, :, :, :-2]
        
        laplacian = laplacian_x + laplacian_y
        
        residual_phase = (self.Gc / self.lc) * pred_phi - self.Gc * self.lc * laplacian - 2 * (1 - pred_phi) * H
        
        loss_phase = F.mse_loss(residual_phase, torch.zeros_like(residual_phase))
        
        # ===== L_elasticity: Elasticity loss (Eq. 10) =====
        stress = stress_tensor(pred_u, self.E, self.nu)
        eq_residual = equilibrium_residual(stress)
        loss_elasticity = F.mse_loss(eq_residual, torch.zeros_like(eq_residual))
        
        # ===== L_BC: Boundary condition loss (Eq. 12) =====
        if boundary_mask is not None:
            loss_bc = F.mse_loss(pred_phi * boundary_mask, true_phi * boundary_mask)
        else:
            loss_bc = torch.tensor(0.0, device=pred_phi.device)
        
        # ===== Total loss (Eq. 9) =====
        total_loss = (self.w1 * loss_elasticity + 
                      self.w2 * loss_phase +
                      self.w3 * loss_bc +
                      self.w4 * loss_data)
        
        return total_loss, {
            'loss_elasticity': loss_elasticity.item(),
            'loss_phase': loss_phase.item(),
            'loss_bc': loss_bc.item(),
            'loss_data': loss_data.item(),
            'loss_phase_data': loss_phase_data.item(),
            'loss_displacement_data': loss_displacement_data.item()
        }