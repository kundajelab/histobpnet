from datetime import datetime
import torch
import os
import lightning as L
import numpy as np
import torch.nn.functional as F

def get_curr_datetime_str():
    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime('%Y%m%d_%H%M%S')
    return formatted_datetime

def get_timestamped_str(base_str: str):
    return f"instance-{get_curr_datetime_str()}_{base_str}"

def get_instance_id():
    return f"instance-{get_curr_datetime_str()}"

def set_random_seed(seed: int, skip_scvi: bool = True):
    L.seed_everything(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if using multi-GPU

    if not skip_scvi:
        import scvi
        scvi.settings.seed = seed
    else:
        # scvi does this so I'm just calling it here if we're not using scvi
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # ensure deterministic CUDA operations for Jax (see https://github.com/google/jax/issues/13672)
        if "XLA_FLAGS" not in os.environ:
            os.environ["XLA_FLAGS"] = "--xla_gpu_deterministic_ops=true"
        else:
            os.environ["XLA_FLAGS"] += " --xla_gpu_deterministic_ops=true"

def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()

def pearson_corr(x: torch.Tensor, y: torch.Tensor, dim: int = -1, eps = 1e-8) -> torch.Tensor:
    """
    Compute the Pearson correlation coefficient along a given dimension for multi-dimensional tensors.

    Args:
        x (torch.Tensor): Input tensor of shape (..., N).
        y (torch.Tensor): Input tensor of shape (..., N).
        dim (int): The dimension along which to compute the Pearson correlation. Default is -1 (last dimension).
        eps: A small constant to prevent division by zero. Default is 1e-8.

    Returns:
        torch.Tensor: Pearson correlation coefficients along the specified dimension.
    """
    # Ensure x and y have the same shape
    assert x.shape == y.shape, "Input tensors must have the same shape"

    # Step 1: Center the data (subtract the mean along the given dimension)
    x_centered = x - torch.mean(x, dim=dim, keepdim=True)
    y_centered = y - torch.mean(y, dim=dim, keepdim=True)

    # Step 2: Compute covariance (sum of element-wise products of centered tensors)
    cov = torch.sum(x_centered * y_centered, dim=dim)

    # Step 3: Compute standard deviations for each tensor along the specified dimension
    std_x = torch.sqrt(torch.sum(x_centered ** 2, dim=dim))
    std_y = torch.sqrt(torch.sum(y_centered ** 2, dim=dim))
    
    # Step 4: Compute Pearson correlation (with numerical stability)
    corr = cov / (std_x * std_y + eps)

    return corr

def multinomial_nll(logits, true_counts):
    """Compute the multinomial negative log-likelihood in PyTorch.
    
    Args:
      true_counts: Tensor of observed counts (batch_size, num_classes) (integer counts)
      logits: Tensor of predicted logits (batch_size, num_classes)
    
    Returns:
      Mean negative log-likelihood across the batch.
    """
    # Ensure true_counts is an integer tensor
    true_counts = true_counts.to(torch.float)  # Keep as float to prevent conversion issues
    
    # Compute total counts per example (should already be integer-like)
    counts_per_example = true_counts.sum(dim=-1, keepdim=True)
    
    # Convert logits to log probabilities (Softmax + Log)
    log_probs = F.log_softmax(logits, dim=-1)
    
    # Compute log-probability of the observed counts
    log_likelihood = (true_counts * log_probs).sum(dim=-1)
    
    # Compute multinomial coefficient (log factorial term)
    log_factorial_counts = torch.lgamma(counts_per_example + 1) - torch.lgamma(true_counts + 1).sum(dim=-1)

    # Compute final NLL
    nll = -(log_factorial_counts + log_likelihood).mean()

    return nll

def softmax(x, temp=1):
    norm_x = x - np.mean(x,axis=1, keepdims=True)
    return np.exp(temp*norm_x)/np.sum(np.exp(temp*norm_x), axis=1, keepdims=True)

# I think these are needed to do interpretability with DeepSHAP
class _Exp(torch.nn.Module):
    def __init__(self):
        super(_Exp, self).__init__()

    def forward(self, X):
        return torch.exp(X)

class _Log(torch.nn.Module):
    def __init__(self):
        super(_Log, self).__init__()

    def forward(self, X):
        return torch.log(X)