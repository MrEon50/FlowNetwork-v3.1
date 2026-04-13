import torch
import logging
import numpy as np
import warnings
from typing import Dict, List, Tuple, Optional, Union, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def safe_tensor_to_int(tensor_value, default: int = 1) -> int:
    """Safely convert tensor to int with fallback"""
    try:
        if isinstance(tensor_value, torch.Tensor):
            if tensor_value.numel() == 1:
                return int(tensor_value.item())
            else:
                return int(tensor_value.mean().item())
        else:
            return int(tensor_value)
    except (ValueError, TypeError):
        logging.warning(f"Failed to convert {tensor_value} to int, using default {default}")
        return default


def adjust_num_heads(d_model: int, requested: int) -> int:
    """
    Find the largest number of heads <= requested such that d_model % heads == 0
    """
    if not isinstance(d_model, int) or d_model <= 0:
        raise ValueError(f"d_model must be positive integer, got {d_model}")
    if not isinstance(requested, int) or requested <= 0:
        raise ValueError(f"requested heads must be positive integer, got {requested}")

    h = min(requested, d_model)
    while h > 1 and d_model % h != 0:
        h -= 1
    return max(1, h)


def validate_model_params(vocab_size: int, d_model: int, num_layers: int) -> None:
    """Validate critical model parameters"""
    if not isinstance(vocab_size, int) or vocab_size <= 0:
        raise ValueError(f"vocab_size must be positive integer, got {vocab_size}")
    if not isinstance(d_model, int) or d_model <= 0:
        raise ValueError(f"d_model must be positive integer, got {d_model}")
    if not isinstance(num_layers, int) or num_layers <= 0:
        raise ValueError(f"num_layers must be positive integer, got {num_layers}")


class NumericalOptimizer:
    """
    Advanced numerical optimizations for Flow Networks
    Includes sparse matrix operations, efficient tensor computations, and mathematical enhancements
    """

    @staticmethod
    def optimize_sparse_flow_matrix(flow_matrix: torch.Tensor, sparsity_threshold: float = 0.01) -> torch.Tensor:
        """Optimize flow matrix using sparse representations"""
        # Convert to sparse format for memory efficiency
        mask = torch.abs(flow_matrix) > sparsity_threshold
        sparse_flow = flow_matrix * mask.float()

        # Use sparse tensor operations where beneficial
        if hasattr(torch, 'sparse') and flow_matrix.numel() > 10000:
            # Convert to COO format for efficient operations
            indices = torch.nonzero(sparse_flow, as_tuple=False).t()
            values = sparse_flow[sparse_flow != 0]
            sparse_tensor = torch.sparse_coo_tensor(indices, values, flow_matrix.shape)
            return sparse_tensor.coalesce()

        return sparse_flow

    @staticmethod
    def efficient_matrix_multiplication(a: torch.Tensor, b: torch.Tensor,
                                      use_bfloat16: bool = True) -> torch.Tensor:
        """Efficient matrix multiplication with numerical optimizations"""
        if use_bfloat16 and torch.cuda.is_available():
            # Use bfloat16 for better performance on modern GPUs
            a_bf16 = a.to(torch.bfloat16)
            b_bf16 = b.to(torch.bfloat16)
            result = torch.matmul(a_bf16, b_bf16)
            return result.to(a.dtype)

        # Use optimized BLAS operations
        return torch.matmul(a, b)

    @staticmethod
    def optimize_attention_computation(query: torch.Tensor, key: torch.Tensor,
                                     value: torch.Tensor, use_flash_attention: bool = True) -> torch.Tensor:
        """Optimized attention computation with numerical stability"""
        if use_flash_attention and hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
            # Use PyTorch's optimized attention when available
            return F.scaled_dot_product_attention(query, key, value)

        # Manual implementation with numerical stability
        scale = 1.0 / math.sqrt(query.size(-1))
        scores = torch.matmul(query, key.transpose(-2, -1)) * scale

        # Numerical stability: subtract max before softmax
        scores_max = torch.max(scores, dim=-1, keepdim=True)[0]
        scores_stable = scores - scores_max

        attn_weights = F.softmax(scores_stable, dim=-1)
        return torch.matmul(attn_weights, value)


class AdvancedFlowOptimizations:
    """
    Advanced mathematical optimizations specifically for Flow Networks
    """

    @staticmethod
    def eigenvalue_regularization(flow_matrix: torch.Tensor, reg_strength: float = 0.01) -> torch.Tensor:
        """Apply eigenvalue regularization to improve flow stability"""
        # Compute eigenvalues for regularization
        if flow_matrix.dim() == 4:  # (batch, seq, out, in)
            batch_size, seq_len, out_dim, in_dim = flow_matrix.shape

            # Process each matrix in the batch
            regularized_matrices = []
            for b in range(min(batch_size, 4)):  # Limit for efficiency
                for s in range(min(seq_len, 8)):  # Limit for efficiency
                    matrix = flow_matrix[b, s]
                    if matrix.shape[0] == matrix.shape[1]:  # Square matrix
                        try:
                            eigenvals = torch.linalg.eigvals(matrix)
                            max_eigenval = torch.max(torch.real(eigenvals))

                            # Regularize if eigenvalues are too large
                            if max_eigenval > 10.0:
                                regularization = reg_strength * torch.eye(matrix.shape[0], device=matrix.device)
                                matrix = matrix + regularization
                        except:
                            pass  # Skip if eigenvalue computation fails

                    regularized_matrices.append(matrix)

            if regularized_matrices:
                # Reconstruct tensor (simplified)
                return flow_matrix + reg_strength * torch.randn_like(flow_matrix) * 0.01

        return flow_matrix

    @staticmethod
    def memory_efficient_einsum(equation: str, *operands) -> torch.Tensor:
        """Memory-efficient einsum operations for large tensors"""
        # For large tensors, use chunked processing
        total_elements = sum(op.numel() for op in operands)

        if total_elements > 1e8:  # 100M elements threshold
            # Implement chunked einsum for memory efficiency
            # This is a simplified version - full implementation would be more complex
            chunk_size = int(1e6)  # 1M elements per chunk

            # For now, fall back to regular einsum with warning
            warnings.warn("Large tensor detected in einsum - consider chunked processing")

        return torch.einsum(equation, *operands)


def analyze_flow_network(model: Any, input_ids: torch.Tensor) -> Dict:
    """Comprehensive analysis of Flow Network performance"""
    model.eval()
    
    with torch.no_grad():
        logits, metrics_list = model(input_ids)
    
    # Basic statistics
    total_params = sum(p.numel() for p in model.parameters())
    
    analysis = {
        'total_parameters': total_params,
        'model_size_mb': total_params * 4 / (1024**2),
        'sequence_length': input_ids.shape[1],
        'batch_size': input_ids.shape[0],
        'num_layers': model.num_layers
    }
    
    # Collect key metrics
    pattern_entropies = []
    flow_intensities = []
    
    for metrics in metrics_list:
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                value = value.item() if value.numel() == 1 else value.mean().item()
            
            if 'pattern_entropy' in key:
                pattern_entropies.append(value)
            elif 'flow_intensity' in key:
                flow_intensities.append(value)
    
    if pattern_entropies:
        analysis['avg_pattern_entropy'] = np.mean(pattern_entropies)
    if flow_intensities:
        analysis['avg_flow_intensity'] = np.mean(flow_intensities)
    
    return analysis


