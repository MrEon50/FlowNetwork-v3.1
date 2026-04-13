import torch
import torch.nn as nn
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from .core import FlowLayer, EnhancedFlowLayer
from .utils import validate_model_params, safe_tensor_to_int, adjust_num_heads

class FlowNetwork(nn.Module):
    """
    Complete Flow Network architecture
    Revolutionary neural network using dynamic flow control
    """
    
    def __init__(self, vocab_size: int, d_model: int = 512, num_layers: int = 6,
                 max_seq_len: int = 2048, dropout: float = 0.1, num_patterns: int = 8):
        super().__init__()
        
        self.d_model = d_model
        self.num_layers = num_layers
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        
        # Flow embedding mixer
        self.embedding_flow = FlowLayer(d_model * 2, d_model, num_patterns)
        
        # Stack of Flow layers
        self.flow_layers = nn.ModuleList([
            FlowLayer(d_model, d_model, num_patterns)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_flow = FlowLayer(d_model, vocab_size, num_patterns)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Global flow gate
        self.global_flow_gate = nn.Parameter(torch.ones(1))
        
    def forward(self, input_ids: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, List[Dict]]:
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Embeddings
        positions = torch.arange(0, seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        
        token_emb = self.token_embedding(input_ids)
        pos_emb = self.pos_embedding(positions)
        
        # Combine embeddings through flow
        combined_emb = torch.cat([token_emb, pos_emb], dim=-1)
        x, emb_metrics = self.embedding_flow(combined_emb)
        x = self.dropout(x)
        
        # Global flow gate
        x = x * self.global_flow_gate
        
        # Flow through layers
        all_metrics = [emb_metrics]
        
        for i, flow_layer in enumerate(self.flow_layers):
            x, layer_metrics = flow_layer(x)
            x = self.dropout(x)
            layer_metrics['layer_index'] = i
            all_metrics.append(layer_metrics)
        
        # Output projection
        logits, output_metrics = self.output_flow(x)
        all_metrics.append(output_metrics)
        
        # Global metrics
        global_metrics = {
            'global_flow_gate': self.global_flow_gate.item(),
            'sequence_length': seq_len,
            'num_active_layers': len(self.flow_layers)
        }
        all_metrics.append(global_metrics)
        
        return logits, all_metrics


class EnhancedFlowTransformer(nn.Module):
    """
    Enhanced Flow Transformer for LLM applications
    Supports UNLIMITED sequences (No Max Seq Len), pure RoPE and Flow without Attention bottlenecks.
    """

    def __init__(self, vocab_size: int, d_model: int = 512, num_layers: int = 8,
                 max_seq_len: int = 0, dropout: float = 0.1,
                 num_patterns: int = 16, context_window: int = 1024,
                 num_heads: int = 8, use_memory: bool = True,
                 boundary_token_ids: List[int] = None):
        super().__init__()

        # Validate critical parameters first
        validate_model_params(vocab_size, d_model, num_layers)

        self.d_model = d_model
        self.num_layers = num_layers
        self.context_window = context_window
        self.use_memory = use_memory
        self.boundary_token_ids = boundary_token_ids or []
        
        # Max sequence length parameter is retained for backwards compatibility
        # However, due to pure Flow & RoPE implementation, it is practically INFINITE.
        self.max_seq_len = max_seq_len if max_seq_len > 0 else 4096

        # Token embedding ONLY (No Absolute Pos Embedding)
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.norm_embedding = nn.LayerNorm(d_model)

        # Context-aware flow embedding mixer
        self.embedding_flow = FlowLayer(d_model, d_model, num_patterns)

        # Stack of pure Flow layers
        self.flow_layers = nn.ModuleList([
            EnhancedFlowLayer(d_model, d_model, num_patterns, num_heads, dropout, use_memory)
            for _ in range(num_layers)
        ])

        # Global memory for ultra-long context
        if use_memory:
            self.register_buffer('global_memory', torch.randn(1024, d_model) * 0.1)
            self.memory_gate = nn.Parameter(torch.ones(1) * 0.3)

        # Output projection
        self.output_flow = EnhancedFlowLayer(
            d_model, vocab_size, num_patterns, num_heads, dropout, False
        )

        self.dropout = nn.Dropout(dropout)
        self.final_norm = nn.LayerNorm(d_model)
        self.global_flow_gate = nn.Parameter(torch.ones(1))
        self.adaptive_controller = self._create_adaptive_controller()

    def _create_adaptive_controller(self):
        """Create adaptive computation controller for dynamic resource allocation"""
        return nn.Sequential(
            nn.Linear(self.d_model, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                memory_context: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, List[Dict]]:
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Pure RoPE approach natively supports infinite sequences
        # No more sliding window workarounds required

        token_emb = self.token_embedding(input_ids)

        # 1. Generate Grammar Boundary Mask (Dyskryminator Interpunkcji i Spacji)
        if hasattr(self, 'boundary_token_ids') and self.boundary_token_ids:
            space_mask = torch.zeros_like(input_ids, dtype=torch.float)
            for token_id in self.boundary_token_ids:
                if (token_id >= 0):
                    space_mask = torch.max(space_mask, (input_ids == token_id).float())
            space_mask = space_mask.unsqueeze(-1)
        else:
            space_mask = None

        x, emb_metrics = self.embedding_flow(token_emb)
        x = self.norm_embedding(x)
        x = self.dropout(x)

        # Global flow gate
        x = x * self.global_flow_gate

        # Adaptive computation
        computation_intensity = self.adaptive_controller(x.mean(dim=1)).mean()
        intensity_scalar = safe_tensor_to_int(computation_intensity * self.num_layers, default=self.num_layers) / self.num_layers
        active_layers = max(2, safe_tensor_to_int(self.num_layers * intensity_scalar, default=self.num_layers))

        all_metrics = [emb_metrics]

        for i in range(active_layers):
            if i < len(self.flow_layers):
                current_memory = None
                if self.use_memory and hasattr(self, 'global_memory'):
                    memory_expanded = self.global_memory.unsqueeze(0).expand(batch_size, -1, -1)
                    current_memory = memory_expanded * self.memory_gate.unsqueeze(0).unsqueeze(-1)

                x, layer_metrics = self.flow_layers[i](x, current_memory, space_mask=space_mask)
                x = self.dropout(x)
                layer_metrics['layer_index'] = i
                layer_metrics['is_active'] = True
                all_metrics.append(layer_metrics)

        x = self.final_norm(x)
        logits, output_metrics = self.output_flow(x)
        all_metrics.append(output_metrics)

        global_metrics = {
            'global_flow_gate': self.global_flow_gate.item(),
            'sequence_length': seq_len,
            'active_layers': active_layers,
            'computation_intensity': computation_intensity.item(),
            'memory_gate': self.memory_gate.item() if self.use_memory else 0.0,
            'max_seq_len': "UNLIMITED"
        }
        all_metrics.append(global_metrics)

        return logits, all_metrics


class CUDAOptimizedFlowNetwork(nn.Module):
    """
    CUDA-optimized version of Enhanced Flow Transformer
    Includes quantization, memory optimization, and GPU-specific enhancements
    """

    def __init__(self, base_model: EnhancedFlowTransformer,
                 enable_mixed_precision: bool = True,
                 enable_gradient_checkpointing: bool = True,
                 quantization_bits: int = 8):
        super().__init__()
        self.base_model = base_model
        self.enable_mixed_precision = enable_mixed_precision
        self.enable_gradient_checkpointing = enable_gradient_checkpointing
        self.quantization_bits = quantization_bits

        # CUDA-specific optimizations
        if torch.cuda.is_available():
            self._setup_cuda_optimizations()

    def _setup_cuda_optimizations(self):
        """Setup CUDA-specific optimizations"""
        # Enable TensorFloat-32 for better performance on Ampere GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # Enable cuDNN benchmark for consistent input sizes
        torch.backends.cudnn.benchmark = True

        # Setup memory pool for efficient allocation
        if hasattr(torch.cuda, 'memory_pool'):
            torch.cuda.empty_cache()

    def forward(self, input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                memory_context: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, List[Dict]]:

        if self.enable_mixed_precision and torch.cuda.is_available():
            # Use automatic mixed precision for better performance
            try:
                # Try new API first
                with torch.amp.autocast('cuda'):
                    return self._forward_with_optimizations(input_ids, attention_mask, memory_context)
            except AttributeError:
                # Fallback to old API
                with torch.cuda.amp.autocast():
                    return self._forward_with_optimizations(input_ids, attention_mask, memory_context)
        else:
            return self._forward_with_optimizations(input_ids, attention_mask, memory_context)

    def _forward_with_optimizations(self, input_ids: torch.Tensor,
                                   attention_mask: Optional[torch.Tensor] = None,
                                   memory_context: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, List[Dict]]:

        if self.enable_gradient_checkpointing and self.training:
            # Use gradient checkpointing to save memory during training
            return torch.utils.checkpoint.checkpoint(
                self.base_model, input_ids, attention_mask, memory_context
            )
        else:
            return self.base_model(input_ids, attention_mask, memory_context)

    def optimize_for_inference(self):
        """Optimize model for inference"""
        self.eval()

        # Safe JIT optimizations with version checking
        try:
            if hasattr(torch.jit, 'optimize_for_inference'):
                self.base_model = torch.jit.optimize_for_inference(
                    torch.jit.script(self.base_model)
                )
        except Exception as e:
            logging.warning(f"JIT optimization failed: {e}")

        # Safe fusion strategy setting
        try:
            if hasattr(torch.jit, 'set_fusion_strategy'):
                torch.jit.set_fusion_strategy([('STATIC', 20), ('DYNAMIC', 20)])
        except Exception as e:
            logging.warning(f"Fusion strategy setting failed: {e}")


class AdaptiveResourceController(nn.Module):
    """
    Dynamic resource allocation controller for efficient LLM processing
    Adjusts computation based on input complexity and available resources
    """

    def __init__(self, max_layers: int = 12, resource_threshold: float = 0.7,
                 complexity_analyzer_dim: int = 512):
        super().__init__()
        self.max_layers = max_layers
        self.resource_threshold = resource_threshold

        # Complexity analyzer
        self.complexity_analyzer = nn.Sequential(
            nn.Linear(complexity_analyzer_dim, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        # Resource monitor (simplified)
        self.resource_monitor = nn.Parameter(torch.tensor(0.5))

    def forward(self, input_tensor: torch.Tensor) -> Dict[str, Union[int, float]]:
        """Determine optimal resource allocation"""
        batch_size, seq_len, _ = input_tensor.shape

        # Analyze input complexity
        complexity_score = self.complexity_analyzer(input_tensor.mean(dim=1)).mean()

        # Simulate resource usage (in practice, would query actual GPU memory)
        current_resource_usage = self.resource_monitor.item()

        # Adjust computation based on complexity and resources
        if complexity_score > 0.8 and current_resource_usage < self.resource_threshold:
            # High complexity, resources available - use more layers
            active_layers = self.max_layers
            batch_size_adjustment = 1.0
        elif complexity_score < 0.3:
            # Low complexity - use fewer layers
            active_layers = max(2, int(self.max_layers * 0.5))
            batch_size_adjustment = 1.2
        else:
            # Medium complexity - standard allocation
            active_layers = max(4, int(self.max_layers * 0.75))
            batch_size_adjustment = 1.0

        return {
            'active_layers': active_layers,
            'batch_size_adjustment': batch_size_adjustment,
            'complexity_score': complexity_score.item(),
            'resource_usage': current_resource_usage,
            'sequence_length': seq_len
        }


