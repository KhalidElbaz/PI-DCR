import torch


def MC_dropout(model, x, T=50):
    """
    Monte Carlo Dropout for uncertainty quantification (Eq. 16-17)
    
    Args:
        model: U-Net model with dropout
        x: Input tensor
        T: Number of stochastic forward passes
    
    Returns:
        mean: Mean prediction
        variance: Prediction variance
        confidence_interval: 95% confidence interval (mean ± 1.96*std)
    """
    model.train()  # Enable dropout
    results = []
    
    with torch.no_grad():
        for _ in range(T):
            pred = model(x, mc_dropout=True)
            results.append(pred)
    
    results = torch.stack(results)
    
    mean = torch.mean(results, dim=0)
    variance = torch.var(results, dim=0)
    std = torch.sqrt(variance)
    
    # 95% confidence interval
    lower_bound = mean - 1.96 * std
    upper_bound = mean + 1.96 * std
    
    return {
        'mean': mean,
        'variance': variance,
        'std': std,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound
    }


def compute_uncertainty_metrics(predictions, ground_truth):
    """
    Compute uncertainty metrics
    """
    uncertainty = predictions['variance']
    mean_pred = predictions['mean']
    
    # Predictive uncertainty
    mean_uncertainty = uncertainty.mean().item()
    max_uncertainty = uncertainty.max().item()
    
    # Confidence coverage (95% confidence interval)
    lower = predictions['lower_bound']
    upper = predictions['upper_bound']
    coverage = ((ground_truth >= lower) & (ground_truth <= upper)).float().mean().item()
    
    return {
        'mean_uncertainty': mean_uncertainty,
        'max_uncertainty': max_uncertainty,
        'confidence_coverage': coverage
    }