import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

from config import *
from unet import PIDCR_UNet
from loss import PIDCRLoss
from dataset import RockFractureDataset
from uncertainty import MC_dropout


def train_epoch(model, loader, optimizer, loss_fn, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    loss_dict_sum = {}
    
    pbar = tqdm(loader, desc='Training')
    for batch in pbar:
        inputs = batch['input'].to(device)
        u_true = batch['u'].to(device)
        phi_true = batch['phi'].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        phi_pred = model(inputs)
        
        # For displacement, we need a separate prediction
        # This is a simplified version - in practice, model should predict both
        u_pred = torch.zeros_like(u_true)  # Placeholder - to be implemented
        
        # Compute loss
        loss, loss_dict = loss_fn(phi_pred, phi_true, u_pred, u_true)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        for k, v in loss_dict.items():
            loss_dict_sum[k] = loss_dict_sum.get(k, 0) + v
        
        pbar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / len(loader)
    avg_loss_dict = {k: v / len(loader) for k, v in loss_dict_sum.items()}
    
    return avg_loss, avg_loss_dict


def validate(model, loader, loss_fn, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in loader:
            inputs = batch['input'].to(device)
            u_true = batch['u'].to(device)
            phi_true = batch['phi'].to(device)
            
            phi_pred = model(inputs)
            u_pred = torch.zeros_like(u_true)
            
            loss, _ = loss_fn(phi_pred, phi_true, u_pred, u_true)
            total_loss += loss.item()
    
    return total_loss / len(loader)


def train_model(model, train_loader, val_loader, loss_fn, epochs, lr, device):
    """Full training loop with early stopping"""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=20, verbose=True
    )
    
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Training
        train_loss, loss_dict = train_epoch(
            model, train_loader, optimizer, loss_fn, device
        )
        train_losses.append(train_loss)
        
        # Validation
        val_loss = validate(model, val_loader, loss_fn, device)
        val_losses.append(val_loss)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Print progress
        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'  Train Loss: {train_loss:.4f}')
        print(f'  Val Loss: {val_loss:.4f}')
        print(f'  Loss Components: {loss_dict}')
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f'Early stopping at epoch {epoch+1}')
                break
    
    return train_losses, val_losses


def cross_validate(dataset, n_folds=NUM_FOLDS, epochs=EPOCHS, lr=LEARNING_RATE):
    """Perform k-fold cross-validation"""
    device = DEVICE
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        print(f'\n{"="*50}')
        print(f'FOLD {fold+1}/{n_folds}')
        print(f'{"="*50}')
        
        # Split dataset
        train_dataset = torch.utils.data.Subset(dataset, train_idx)
        val_dataset = torch.utils.data.Subset(dataset, val_idx)
        
        train_loader = DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=BATCH_SIZE, shuffle=False
        )
        
        # Initialize model
        model = PIDCR_UNet(
            in_channels=1,
            out_channels=1,
            features=[64, 128, 256, 512],
            dropout_rate=dropout_probability
        ).to(device)
        
        # Initialize loss function with optimal weights from Fig. 22
        loss_fn = PIDCRLoss(E, nu, Gc, lc, (w1, w2, w3, w4)).to(device)
        
        # Train
        train_losses, val_losses = train_model(
            model, train_loader, val_loader, loss_fn,
            epochs, lr, device
        )
        
        # Store results
        fold_results.append({
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': min(val_losses),
            'model': model
        })
        
        print(f'Fold {fold+1} Best Val Loss: {min(val_losses):.4f}')
    
    # Summary
    print('\n' + '='*50)
    print('CROSS-VALIDATION RESULTS')
    print('='*50)
    best_losses = [r['best_val_loss'] for r in fold_results]
    print(f'Mean Best Val Loss: {np.mean(best_losses):.4f} ± {np.std(best_losses):.4f}')
    
    return fold_results


def run_training():
    """Main training function"""
    print('='*60)
    print('PI-DCR Model Training')
    print('='*60)
    print(f'Device: {DEVICE}')
    print(f'Batch Size: {BATCH_SIZE}')
    print(f'Epochs: {EPOCHS}')
    print(f'Learning Rate: {LEARNING_RATE}')
    print(f'Loss Weights: w1={w1}, w2={w2}, w3={w3}, w4={w4}')
    
    # Create synthetic dataset for testing
    # In practice, replace with actual data loading
    n_samples = 100
    img_size = 256
    
    print(f'\nCreating synthetic dataset ({n_samples} samples)...')
    
    # Generate synthetic data
    ct_images = np.random.randn(n_samples, 1, img_size, img_size).astype(np.float32)
    phase_fields = np.random.rand(n_samples, 1, img_size, img_size).astype(np.float32)
    displacements = np.random.randn(n_samples, 2, img_size, img_size).astype(np.float32)
    
    # Convert to tensors
    dataset = RockFractureDataset(ct_images, displacements, phase_fields)
    
    print(f'Dataset size: {len(dataset)}')
    
    # Run cross-validation
    fold_results = cross_validate(dataset)
    
    # Train final model on full dataset
    print('\n' + '='*50)
    print('Training final model on full dataset')
    print('='*50)
    
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    final_model = PIDCR_UNet(
        in_channels=1,
        out_channels=1,
        features=[64, 128, 256, 512],
        dropout_rate=dropout_probability
    ).to(DEVICE)
    
    loss_fn = PIDCRLoss(E, nu, Gc, lc, (w1, w2, w3, w4)).to(DEVICE)
    
    train_losses, _ = train_model(
        final_model, loader, None, loss_fn,
        epochs=100, lr=LEARNING_RATE, device=DEVICE
    )
    
    # Save final model
    torch.save({
        'model_state_dict': final_model.state_dict(),
        'config': {
            'E': E, 'nu': nu, 'Gc': Gc, 'lc': lc,
            'w1': w1, 'w2': w2, 'w3': w3, 'w4': w4
        }
    }, 'final_model.pth')
    
    print('\nTraining complete!')
    print('Models saved as: best_model.pth, final_model.pth')
    
    return final_model, fold_results


if __name__ == '__main__':
    model, results = run_training()