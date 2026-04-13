import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union, Any
from .utils import adjust_num_heads, safe_tensor_to_int

class AdaptiveFlowRouter(nn.Module):
    """
    ULTRA-EFFICIENT Flow: Applied pattern-by-vector.
    Avoids creating (B, S, D, D) matrices entirely.
    """
    
    def __init__(self, input_dim: int, output_dim: int, 
                 num_flow_patterns: int = 8, base_sparsity: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_patterns = num_flow_patterns
        
        self.flow_patterns = nn.Parameter(
            torch.randn(num_flow_patterns, output_dim, input_dim) * 0.05
        )
        self.pattern_selector = nn.Sequential(
            nn.Linear(input_dim, num_flow_patterns),
            nn.Softmax(dim=-1)
        )
        self.flow_intensity = nn.Sequential(
            nn.Linear(input_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        # x: (B, S, D_in)
        # Select weights: (B, S, P)
        pattern_weights = self.pattern_selector(x)
        intensity = self.flow_intensity(x) # (B, S, 1)

        # EFFICIENT COMPUTATION: y = sum(weights_p * (Pattern_p @ x))
        # No large intermediate matrix!
        # x: (B, S, D_in) -> (B, S, 1, D_in)
        # patterns: (P, D_out, D_in)
        # result: (B, S, P, D_out)
        transformed = torch.einsum('pij,bsj->bspi', self.flow_patterns, x)
        
        # Apply weights: (B, S, P) * (B, S, P, D_out) -> (B, S, D_out)
        output = torch.einsum('bsp,bspi->bsi', pattern_weights, transformed)
        output = output * intensity
        
        metrics = {
            'pattern_entropy': -(pattern_weights * torch.log(pattern_weights + 1e-8)).sum(-1).mean(),
            'flow_intensity': intensity.mean()
        }
        
        return output, metrics

class ContextAwareFlowRouter(nn.Module):
    """
    Multi-channel context-aware flow without OOM.
    Uses fused pattern-vector products.
    """
    def __init__(self, input_dim: int, output_dim: int, num_patterns: int = 16,
                 context_window: int = 1024, max_seq_len: int = 4096,
                 num_channels: int = 3):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_channels = num_channels
        self.num_patterns = num_patterns

        # Patterns per channel
        self.p_per_ch = max(1, num_patterns // num_channels)
        self.channel_patterns = nn.ParameterList([
            nn.Parameter(torch.randn(self.p_per_ch, output_dim, input_dim) * 0.05)
            for _ in range(num_channels)
        ])

        self.channel_mixer = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.GELU(),
            nn.Linear(64, num_channels),
            nn.Softmax(dim=-1)
        )
        self.context_projection = nn.Linear(input_dim, min(input_dim, output_dim))
        self.context_selector = nn.Sequential(
            nn.Linear(input_dim + min(input_dim, output_dim), self.p_per_ch),
            nn.Softmax(dim=-1)
        )
        self.flow_intensity = nn.Sequential(
            nn.Linear(input_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        batch_size, seq_len, _ = x.shape
        
        # 1. Context & Channels
        ctx = self.context_projection(x.mean(dim=1, keepdim=True)).expand(-1, seq_len, -1)
        p_weights = self.context_selector(torch.cat([x, ctx], dim=-1)) # (B, S, P_per_ch)
        c_weights = self.channel_mixer(x.mean(dim=1)) # (B, Ch)
        intensity = self.flow_intensity(x)

        # 2. Memory-efficient fused computation
        final_output = torch.zeros(batch_size, seq_len, self.output_dim, device=x.device, dtype=x.dtype)
        
        for ch_idx, ch_p in enumerate(self.channel_patterns):
            # Transform for this channel patterns: (B, S, P_ch, D_out)
            ch_trans = torch.einsum('pij,bsj->bspi', ch_p, x)
            # Weighted mix for this channel: (B, S, D_out)
            ch_res = torch.einsum('bsp,bspi->bsi', p_weights, ch_trans)
            # Add to final result with channel importance
            final_output.add_(ch_res * c_weights[:, ch_idx].view(-1, 1, 1))

        final_output.mul_(intensity)

        metrics = {
            'flow_intensity': intensity.mean(),
            'channel_entropy': -(c_weights * torch.log(c_weights + 1e-8)).sum(-1).mean()
        }
        return final_output, metrics

class SpatialBoundaryRouter(nn.Module):
    """
    The 'Space Discriminator' optimized for VRAM.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.patterns = nn.Parameter(torch.randn(2, d_model, d_model) * 0.05)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor, space_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict]:
        prob = self.classifier(x)
        mask = space_mask if space_mask is not None else prob # (B, S, 1)

        # Apply two patterns based on mask without building full (B,S,D,D)
        # y = (1-mask) * (P0 @ x) + mask * (P1 @ x)
        p0_x = torch.einsum('ij,bsj->bsi', self.patterns[0], x)
        p1_x = torch.einsum('ij,bsj->bsi', self.patterns[1], x)
        
        output = (1.0 - mask) * p0_x + mask * p1_x
        
        return output, {'flow_boundary_prob': prob.mean()}

class EnhancedFlowLayer(nn.Module):
    """
    Refactored for pure Memory-Efficient Flow.
    """
    def __init__(self, input_dim: int, output_dim: int, num_patterns: int = 16,
                 num_heads: int = 8, dropout: float = 0.1, use_memory: bool = True):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(output_dim)
        self.flow_router = ContextAwareFlowRouter(input_dim, output_dim, num_patterns)
        self.spatial_router = SpatialBoundaryRouter(output_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(output_dim, output_dim * 4),
            nn.GELU(),
            nn.Linear(output_dim * 4, output_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor, memory_context: Any = None, space_mask: torch.Tensor = None) -> Tuple[torch.Tensor, Dict]:
        x_norm = self.norm1(x)
        
        # 1. Flow logic
        flow_out, metrics = self.flow_router(x_norm)
        
        # 2. Spatial logic
        spatial_out, s_metrics = self.spatial_router(flow_out, space_mask)
        metrics.update(s_metrics)
        
        # Merge & Residual (skocz przez warstwe tylko jesli wymiary pasuja)
        if self.input_dim == self.output_dim:
            x = x + flow_out + spatial_out
        else:
            x = flow_out + spatial_out
            
        x = self.norm2(x)
        x = x + self.ffn(x)
        
        return x, metrics

class FlowLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, num_patterns: int = 8):
        super().__init__()
        self.router = AdaptiveFlowRouter(input_dim, output_dim, num_patterns)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        out, m = self.router(x)
        return self.norm(F.gelu(out)), m

class FlowMemoryNetwork(nn.Module):
    # Minimal version to keep it functional
    def __init__(self, d_model: int = 512, memory_size: int = 512):
        super().__init__()
        self.bank = nn.Parameter(torch.randn(memory_size, d_model) * 0.01)
    def forward(self, x, update=True):
        return x, {}
