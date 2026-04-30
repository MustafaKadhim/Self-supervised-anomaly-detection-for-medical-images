#### code for Stage 2 of model framework #####


import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Optional, Tuple
from torch.distributions import Beta
from model_stage1 import Stage1RVQVAE

# Enable Flash/Memory-efficient SDPA for PyTorch 2.x
if hasattr(torch.backends.cuda, "enable_flash_sdp"):
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)  # fallback

 
# =============================================================================
# 3D Rotary Position Embeddings (RoPE) for Medical Imaging
# =============================================================================


class RotaryEmbedding3D(nn.Module):
    """3D Rotary Position Embeddings for volumetric/slice-aware tokens.
    
    Splits embedding dimension into thirds: row, column, and slice (z-axis).
    This enables the model to learn anatomically relevant 3D spatial relationships
    in medical imaging where context across slices is crucial.
    
    For head_dim that isn't divisible by 6, we allocate:
    - 1/3 for row encoding
    - 1/3 for column encoding  
    - 1/3 for slice (z) encoding
    """

    def __init__(
        self,
        dim: int,
        max_positions: int = 64,
        max_slices: int = 92,
        base: float = 25000.0,
    ) -> None:
        super().__init__()
        # For 3D, we need dim divisible by 6 (2 per dimension for rotate_half)
        # If not perfectly divisible, we'll handle it gracefully
        assert dim % 2 == 0, f"Dimension must be even, got {dim}"
        
        self.dim = dim
        # Split dimension into three parts (row, col, slice)
        # Each part needs to be even for rotate_half operation
        self.third_dim = (dim // 3) // 2 * 2  # Ensure even
        self.row_dim = self.third_dim
        self.col_dim = self.third_dim
        self.slice_dim = dim - 2 * self.third_dim  # Remainder goes to slice
        
        # Precompute inverse frequencies for each dimension
        # Row frequencies
        inv_freq_row = 1.0 / (base ** (torch.arange(0, self.row_dim, 2).float() / self.row_dim))
        self.register_buffer("inv_freq_row", inv_freq_row, persistent=False)
        
        # Column frequencies
        inv_freq_col = 1.0 / (base ** (torch.arange(0, self.col_dim, 2).float() / self.col_dim))
        self.register_buffer("inv_freq_col", inv_freq_col, persistent=False)
        
        # Slice frequencies (can use different base for different scale)
        inv_freq_slice = 1.0 / (base ** (torch.arange(0, self.slice_dim, 2).float() / self.slice_dim))
        self.register_buffer("inv_freq_slice", inv_freq_slice, persistent=False)
        
        # Precompute sin/cos tables for spatial positions
        positions = torch.arange(max_positions).float()
        
        freqs_row = torch.einsum("i,j->ij", positions, inv_freq_row)
        emb_row = torch.cat([freqs_row, freqs_row], dim=-1)
        self.register_buffer("cos_row", emb_row.cos(), persistent=False)
        self.register_buffer("sin_row", emb_row.sin(), persistent=False)
        
        freqs_col = torch.einsum("i,j->ij", positions, inv_freq_col)
        emb_col = torch.cat([freqs_col, freqs_col], dim=-1)
        self.register_buffer("cos_col", emb_col.cos(), persistent=False)
        self.register_buffer("sin_col", emb_col.sin(), persistent=False)
        
        # Precompute sin/cos tables for slice positions (larger range)
        slice_positions = torch.arange(max_slices).float()
        freqs_slice = torch.einsum("i,j->ij", slice_positions, inv_freq_slice)
        emb_slice = torch.cat([freqs_slice, freqs_slice], dim=-1)
        self.register_buffer("cos_slice", emb_slice.cos(), persistent=False)
        self.register_buffer("sin_slice", emb_slice.sin(), persistent=False)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        row_pos: torch.Tensor,
        col_pos: torch.Tensor,
        slice_pos: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply 3D rotary embeddings to queries and keys.
        
        Args:
            q, k: (B, heads, seq_len, head_dim)
            row_pos, col_pos: (B, heads, seq_len) or broadcastable - spatial positions
            slice_pos: (B, heads, seq_len) or (B, 1, 1) - slice/z-axis position.
                       If None, defaults to 0 (single-slice mode for backward compat).
        """
        # Split into row, column, and slice portions
        q_row = q[..., :self.row_dim]
        q_col = q[..., self.row_dim:self.row_dim + self.col_dim]
        q_slice = q[..., self.row_dim + self.col_dim:]
        
        k_row = k[..., :self.row_dim]
        k_col = k[..., self.row_dim:self.row_dim + self.col_dim]
        k_slice = k[..., self.row_dim + self.col_dim:]
        
        # Get sin/cos for row positions
        row_pos = row_pos.long().clamp(0, self.cos_row.size(0) - 1)
        cos_r = self.cos_row[row_pos]
        sin_r = self.sin_row[row_pos]
        
        # Get sin/cos for column positions
        col_pos = col_pos.long().clamp(0, self.cos_col.size(0) - 1)
        cos_c = self.cos_col[col_pos]
        sin_c = self.sin_col[col_pos]
        
        # Get sin/cos for slice positions
        if slice_pos is None:
            # Default to slice 0 for backward compatibility
            slice_pos = torch.zeros_like(row_pos)
        slice_pos = slice_pos.long().clamp(0, self.cos_slice.size(0) - 1)
        cos_s = self.cos_slice[slice_pos]
        sin_s = self.sin_slice[slice_pos]
        
        # Apply RoPE to row portion
        q_row = q_row * cos_r + self._rotate_half(q_row) * sin_r
        k_row = k_row * cos_r + self._rotate_half(k_row) * sin_r
        
        # Apply RoPE to column portion
        q_col = q_col * cos_c + self._rotate_half(q_col) * sin_c
        k_col = k_col * cos_c + self._rotate_half(k_col) * sin_c
        
        # Apply RoPE to slice portion
        q_slice = q_slice * cos_s + self._rotate_half(q_slice) * sin_s
        k_slice = k_slice * cos_s + self._rotate_half(k_slice) * sin_s
        
        # Concatenate back
        q_out = torch.cat([q_row, q_col, q_slice], dim=-1)
        k_out = torch.cat([k_row, k_col, k_slice], dim=-1)
        
        return q_out, k_out


# =============================================================================
# RMSNorm and SwiGLU (Modern Components)
# =============================================================================


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization - faster and more stable."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x * norm).type_as(x) * self.scale


class SwiGLU(nn.Module):
    """Swish-Gated Linear Unit FFN - outperforms GELU in transformers."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim)
        self.w2 = nn.Linear(dim, hidden_dim)
        self.w3 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w3(F.silu(self.w1(x)) * self.w2(x)))


# =============================================================================
# Custom Transformer Block with SDPA and 2D RoPE
# =============================================================================


class TransformerBlockSDPA(nn.Module):
    """Transformer block using explicit F.scaled_dot_product_attention with 3D RoPE.
    
    Supports volumetric medical imaging by encoding row, column, and slice positions.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        seq_hw: int,
        dropout: float = 0.0,
        max_slices: int = 128,
    ) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.seq_hw = seq_hw
        self.dropout_p = dropout
        
        # Attention
        self.norm1 = RMSNorm(embed_dim)
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(dropout)
        
        # 3D RoPE (row, column, slice)
        self.rope = RotaryEmbedding3D(
            self.head_dim,
            max_positions=seq_hw + 1,
            max_slices=max_slices,
        )
        
        # FFN with SwiGLU
        self.norm2 = RMSNorm(embed_dim)
        self.ffn = SwiGLU(embed_dim, embed_dim * 4, dropout)
        
        # Precompute 2D grid positions (slice is passed dynamically)
        rows, cols = torch.meshgrid(
            torch.arange(seq_hw), torch.arange(seq_hw), indexing="ij"
        )
        self.register_buffer("row_pos", rows.flatten().long(), persistent=False)
        self.register_buffer("col_pos", cols.flatten().long(), persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        slice_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with optional slice position for 3D context.
        
        Args:
            x: (B, S, D) input tokens
            slice_pos: (B,) slice indices for each sample in batch.
                       If None, defaults to 0 (backward compatible).
        """
        B, S, D = x.shape
        
        # Pre-norm attention
        residual = x
        x = self.norm1(x)
        
        # QKV projection
        qkv = self.qkv(x).view(B, S, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, S, head_dim)
        q, k, v = qkv.unbind(0)
        
        # Apply 3D RoPE (row, col, slice)
        row_pos = self.row_pos[:S].view(1, 1, S).expand(B, self.num_heads, S)
        col_pos = self.col_pos[:S].view(1, 1, S).expand(B, self.num_heads, S)
        
        # Expand slice position to match attention dims
        if slice_pos is not None:
            # slice_pos: (B,) -> (B, num_heads, S) broadcast
            slice_pos_expanded = slice_pos.view(B, 1, 1).expand(B, self.num_heads, S)
        else:
            slice_pos_expanded = None
        
        q, k = self.rope(q, k, row_pos, col_pos, slice_pos_expanded)
        
        # Scaled dot-product attention (uses Flash/Memory-efficient when available)
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout_p if self.training else 0.0,
        )
        
        # Output projection
        attn_out = attn_out.transpose(1, 2).reshape(B, S, D)
        x = residual + self.attn_dropout(self.out_proj(attn_out))
        
        # Pre-norm FFN
        x = x + self.ffn(self.norm2(x))
        
        return x


class TransformerSDPA(nn.Module):
    """Stack of TransformerBlockSDPA layers with 3D positional encoding support."""

    def __init__(
        self,
        embed_dim: int,
        depth: int,
        num_heads: int,
        seq_hw: int,
        dropout: float = 0.0,
        max_slices: int = 128,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlockSDPA(embed_dim, num_heads, seq_hw, dropout, max_slices)
            for _ in range(depth)
        ])
        self.final_norm = RMSNorm(embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        slice_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with optional slice position.
        
        Args:
            x: (B, S, D) input tokens
            slice_pos: (B,) slice indices for 3D context
        """
        for layer in self.layers:
            x = layer(x, slice_pos=slice_pos)
        return self.final_norm(x)


class LearnableWeights(nn.Module):
    """Learnable fusion weights for anomaly components."""

    def __init__(self, w_nll_l1: float = 1.0, w_nll_l2: float = 0.5, w_q_error: float = 0.2) -> None:
        super().__init__()
        self.weights = nn.Parameter(torch.tensor([w_nll_l1, w_nll_l2, w_q_error], dtype=torch.float32))

    def forward(self, nll_l1: torch.Tensor, nll_l2: torch.Tensor, q_error: torch.Tensor) -> torch.Tensor:
        coeffs = F.softmax(self.weights, dim=0)
        return coeffs[0] * nll_l1 + coeffs[1] * nll_l2 + coeffs[2] * q_error


class FactorizedMaskGIT(pl.LightningModule):
    def __init__(
        self,
        codebook_size_level1: int = 512,
        codebook_size_level2: int = 512,
        embed_dim: int = 256,
        depth: int = 8,
        num_heads: int = 8,
        seq_len: int = 1024,
        lr: float = 1e-4,
        stage1: Optional[Stage1RVQVAE] = None,
        mask_ratio: float = 0.20,
        l2_loss_weight: float = 0.25,
        q_error_weight: float = 0.1,
        weight_decay: float = 0.01,
        warmup_steps: int = 2000,
        label_smoothing: float = 0.05,
        beta_alpha: float = 4.0,
        beta_beta: float = 4.0,
        mask_ratio_min: float = 0.15,
        mask_ratio_max: float = 0.75,
        learnable_anomaly_weights: bool = False,
        train_slice_min: int = 30,
        train_slice_max: int = 60,
    ):
        super().__init__()
        torch.set_float32_matmul_precision("medium")
        self._codebook_validated = False  # one-time validation flag

        # Derive seq_len and codebook sizes from Stage-1 if provided
        self.stage1 = stage1
        if stage1 is not None:
            image_size = stage1.hparams.image_size
            patch_size = stage1.hparams.patch_size
            seq_len = (image_size // patch_size) ** 2
            codebook_size_level1 = stage1.hparams.codebook_size
            codebook_size_level2 = stage1.hparams.codebook_size
            self.patch_size = patch_size
            for p in stage1.parameters():
                p.requires_grad = False
            stage1.eval()
        else:
            self.patch_size = 16  # fallback for standalone usage

        self.save_hyperparameters(ignore=["stage1"])

        self.seq_len = seq_len
        self.seq_hw = int(math.sqrt(seq_len))

        # Token embeddings (no learned position embedding - using 3D RoPE for volumetric context)
        self.l1_embed = nn.Embedding(codebook_size_level1 + 1, embed_dim)  # extra for mask
        self.l2_embed = nn.Embedding(codebook_size_level2 + 1, embed_dim)  # extra for mask
        
        # Task embedding: 0 = L1 prediction, 1 = L2 prediction
        self.task_embed = nn.Embedding(2, embed_dim)
        
        # Custom transformer with SDPA and 3D RoPE (row, col, slice) for anatomical context
        self.transformer = TransformerSDPA(embed_dim, depth, num_heads, self.seq_hw, dropout=0.0)

        self.head_l1 = nn.Linear(embed_dim, codebook_size_level1)
        self.head_l2 = nn.Linear(embed_dim, codebook_size_level2)
        self.mask_token_id_l1 = codebook_size_level1
        self.mask_token_id_l2 = codebook_size_level2
        
        # Training hyperparameters
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.label_smoothing = label_smoothing
        self.mask_ratio = mask_ratio
        self.l2_loss_weight = l2_loss_weight
        self.q_error_weight = q_error_weight
        self.use_learnable_weights = learnable_anomaly_weights
        self.anomaly_weights = LearnableWeights() if learnable_anomaly_weights else None
        self.num_scales = getattr(stage1, "num_scales", 3)
        self.scale_weights = nn.Parameter(torch.ones(self.num_scales, dtype=torch.float32))
        
        # Beta distribution masking parameters
        self.beta_alpha = beta_alpha
        self.beta_beta = beta_beta
        self.mask_ratio_min = mask_ratio_min
        self.mask_ratio_max = mask_ratio_max
        
        # Token frequency tracking for majority-class baseline comparison
        # These track how often each token appears to compute "random guess" baseline
        self.register_buffer(
            "_l1_token_counts",
            torch.zeros(codebook_size_level1, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "_l2_token_counts",
            torch.zeros(codebook_size_level2, dtype=torch.long),
            persistent=False,
        )
        self._total_tokens_seen = 0
        self._frequency_log_interval = 1000  # Log frequency stats every N batches

        # Slice filtering for training
        self.train_slice_min = train_slice_min
        self.train_slice_max = train_slice_max

    def _validate_codebook_sizes(self, indices: torch.Tensor) -> None:
        """One-time assertion that indices fit within codebook sizes."""
        if self._codebook_validated:
            return
        max_l1 = indices[:, :, 0].max().item()
        max_l2 = indices[:, :, 1].max().item()
        assert max_l1 < self.mask_token_id_l1, (
            f"L1 index {max_l1} >= codebook_size {self.mask_token_id_l1}"
        )
        assert max_l2 < self.mask_token_id_l2, (
            f"L2 index {max_l2} >= codebook_size {self.mask_token_id_l2}"
        )
        self._codebook_validated = True
    
    def _update_token_frequency(self, indices: torch.Tensor) -> None:
        """Update token frequency counts for baseline computation.
        
        Args:
            indices: (B, seq_len, 2) token indices from Stage-1 RVQ
        """
        l1_idx = indices[:, :, 0].flatten()
        l2_idx = indices[:, :, 1].flatten()
        
        # Accumulate counts using bincount
        l1_counts = torch.bincount(l1_idx, minlength=self.mask_token_id_l1)
        l2_counts = torch.bincount(l2_idx, minlength=self.mask_token_id_l2)
        
        self._l1_token_counts += l1_counts
        self._l2_token_counts += l2_counts
        self._total_tokens_seen += l1_idx.numel()
    
    def get_token_frequency_stats(self) -> dict:
        """Compute token frequency statistics and majority-class baseline.
        
        Returns:
            Dictionary with frequency stats including:
            - Most frequent tokens and their frequencies
            - Majority-class baseline accuracy (what you'd get by always predicting most common)
            - Entropy of token distribution (higher = more uniform)
        """
        if self._total_tokens_seen == 0:
            return {"error": "No tokens seen yet"}
        
        # L1 statistics
        l1_probs = self._l1_token_counts.float() / self._l1_token_counts.sum().clamp(min=1)
        l1_most_freq_idx = self._l1_token_counts.argmax().item()
        l1_most_freq_count = self._l1_token_counts[l1_most_freq_idx].item()
        l1_majority_baseline = l1_most_freq_count / self._l1_token_counts.sum().clamp(min=1).item()
        l1_entropy = -(l1_probs * l1_probs.clamp(min=1e-10).log()).sum().item()
        l1_num_used = (self._l1_token_counts > 0).sum().item()
        
        # Top-5 most frequent L1 tokens
        l1_top5_counts, l1_top5_idx = self._l1_token_counts.topk(min(5, len(self._l1_token_counts)))
        
        # L2 statistics
        l2_probs = self._l2_token_counts.float() / self._l2_token_counts.sum().clamp(min=1)
        l2_most_freq_idx = self._l2_token_counts.argmax().item()
        l2_most_freq_count = self._l2_token_counts[l2_most_freq_idx].item()
        l2_majority_baseline = l2_most_freq_count / self._l2_token_counts.sum().clamp(min=1).item()
        l2_entropy = -(l2_probs * l2_probs.clamp(min=1e-10).log()).sum().item()
        l2_num_used = (self._l2_token_counts > 0).sum().item()
        
        # Top-5 most frequent L2 tokens
        l2_top5_counts, l2_top5_idx = self._l2_token_counts.topk(min(5, len(self._l2_token_counts)))
        
        return {
            "total_tokens_seen": self._total_tokens_seen,
            # L1 stats
            "l1_most_frequent_token": l1_most_freq_idx,
            "l1_most_frequent_count": l1_most_freq_count,
            "l1_majority_baseline_acc": l1_majority_baseline,
            "l1_entropy": l1_entropy,
            "l1_codebook_utilization": l1_num_used / self.mask_token_id_l1,
            "l1_num_tokens_used": l1_num_used,
            "l1_top5_tokens": l1_top5_idx.tolist(),
            "l1_top5_counts": l1_top5_counts.tolist(),
            # L2 stats
            "l2_most_frequent_token": l2_most_freq_idx,
            "l2_most_frequent_count": l2_most_freq_count,
            "l2_majority_baseline_acc": l2_majority_baseline,
            "l2_entropy": l2_entropy,
            "l2_codebook_utilization": l2_num_used / self.mask_token_id_l2,
            "l2_num_tokens_used": l2_num_used,
            "l2_top5_tokens": l2_top5_idx.tolist(),
            "l2_top5_counts": l2_top5_counts.tolist(),
        }
    
    def print_token_frequency_summary(self) -> None:
        """Print a human-readable summary of token frequency statistics.
        
        Call this after training to understand the token distribution and
        verify the model is learning beyond majority-class guessing.
        """
        stats = self.get_token_frequency_stats()
        if "error" in stats:
            print(f"⚠️  {stats['error']}")
            return
        
        print("\n" + "="*70)
        print("TOKEN FREQUENCY ANALYSIS - Majority Class Baseline Check")
        print("="*70)
        print(f"\n📊 Total tokens seen: {stats['total_tokens_seen']:,}")
        
        print("\n" + "-"*35 + " L1 (Structure) " + "-"*35)
        print(f"  Most frequent token: {stats['l1_most_frequent_token']} "
              f"(seen {stats['l1_most_frequent_count']:,} times)")
        print(f"  Majority baseline accuracy: {stats['l1_majority_baseline_acc']*100:.2f}%")
        print(f"  ↳ If model acc_l1 ≤ {stats['l1_majority_baseline_acc']*100:.2f}%, it's just guessing the mode!")
        print(f"  Codebook utilization: {stats['l1_codebook_utilization']*100:.1f}% "
              f"({stats['l1_num_tokens_used']}/{self.mask_token_id_l1} tokens used)")
        print(f"  Distribution entropy: {stats['l1_entropy']:.3f} "
              f"(max possible: {math.log(self.mask_token_id_l1):.3f})")
        print(f"  Top-5 tokens: {stats['l1_top5_tokens']}")
        print(f"  Top-5 counts: {stats['l1_top5_counts']}")
        
        print("\n" + "-"*35 + " L2 (Texture) " + "-"*37)
        print(f"  Most frequent token: {stats['l2_most_frequent_token']} "
              f"(seen {stats['l2_most_frequent_count']:,} times)")
        print(f"  Majority baseline accuracy: {stats['l2_majority_baseline_acc']*100:.2f}%")
        print(f"  ↳ If model acc_l2 ≤ {stats['l2_majority_baseline_acc']*100:.2f}%, it's just guessing the mode!")
        print(f"  Codebook utilization: {stats['l2_codebook_utilization']*100:.1f}% "
              f"({stats['l2_num_tokens_used']}/{self.mask_token_id_l2} tokens used)")
        print(f"  Distribution entropy: {stats['l2_entropy']:.3f} "
              f"(max possible: {math.log(self.mask_token_id_l2):.3f})")
        print(f"  Top-5 tokens: {stats['l2_top5_tokens']}")
        print(f"  Top-5 counts: {stats['l2_top5_counts']}")
        
        print("\n" + "-"*70)
        print("📈 INTERPRETATION:")
        print("  • 'lift' metrics show improvement over majority-class baseline")
        print("  • Positive lift = model learned meaningful patterns")
        print("  • Near-zero lift = model may be memorizing frequent tokens")
        print("  • Low entropy = imbalanced codebook (few dominant tokens)")
        print("="*70 + "\n")
    
    def reset_token_frequency_counts(self) -> None:
        """Reset token frequency counters (e.g., for a new training run)."""
        self._l1_token_counts.zero_()
        self._l2_token_counts.zero_()
        self._total_tokens_seen = 0
    
    def _compute_majority_baseline_accuracy(self, target: torch.Tensor, mask: torch.Tensor, level: str) -> torch.Tensor:
        """Compute what accuracy would be if we always predicted the most frequent token.
        
        Args:
            target: Ground truth token indices
            mask: Boolean mask of positions being predicted
            level: "l1" or "l2" to select which frequency counts to use
            
        Returns:
            Baseline accuracy as a scalar tensor
        """
        counts = self._l1_token_counts if level == "l1" else self._l2_token_counts
        if counts.sum() == 0:
            return torch.tensor(0.0, device=target.device)
        
        most_frequent_token = counts.argmax()
        masked_targets = target[mask]
        if masked_targets.numel() == 0:
            return torch.tensor(0.0, device=target.device)
        
        # How many of the masked targets match the most frequent token?
        baseline_correct = (masked_targets == most_frequent_token).float().mean()
        return baseline_correct

    def _sample_rate(self, task: str) -> float:
        """Task-specific masking rates to reduce train/test mismatch."""
        r = Beta(self.beta_alpha, self.beta_beta).sample().item()
        if task == "l1":
            # Prefer higher mask for structure tokens, occasionally moderate
            if torch.rand(()) < 0.7:
                r = 0.50 + torch.rand(()).item() * 0.25  # 0.50–0.75
            else:
                r = max(0.20, min(0.50, r))  # 0.20–0.50
        else:  # l2
            r = max(0.15, min(0.55, r))  # predominantly moderate
        return max(self.mask_ratio_min, min(self.mask_ratio_max, r))

    def _apply_mask(self, tokens: torch.Tensor, mask_token_id: int, task: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Random token masking with guaranteed ≥1 masked token per sample."""
        rate = self._sample_rate(task) if self.training else self.mask_ratio
        mask = torch.rand_like(tokens.float()) < rate
        # Ensure at least one token is masked per sample (prevents degenerate loss)
        for i in range(mask.size(0)):
            if mask[i].sum() == 0:
                rand_pos = torch.randint(0, mask.size(1), (1,), device=mask.device)
                mask[i, rand_pos] = True
        masked_tokens = tokens.masked_fill(mask, mask_token_id)
        return masked_tokens, mask

    def _apply_block_mask(self, tokens: torch.Tensor, mask_token_id: int, task: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Block-wise masking for spatially coherent inpainting."""
        B, S = tokens.shape
        H = W = self.seq_hw
        device = tokens.device

        rate = self._sample_rate(task) if self.training else self.mask_ratio
        target = int(S * rate)

        mask = torch.zeros(B, H, W, dtype=torch.bool, device=device)

        for b in range(B):
            # Keep adding rectangles until target coverage per sample
            while mask[b].sum().item() < max(1, target):
                bh = torch.randint(2, max(3, H // 2), (1,), device=device).item()
                bw = torch.randint(2, max(3, W // 2), (1,), device=device).item()
                r = torch.randint(0, H - bh + 1, (1,), device=device).item()
                c = torch.randint(0, W - bw + 1, (1,), device=device).item()
                mask[b, r:r + bh, c:c + bw] = True

        mask = mask.view(B, S)
        masked_tokens = tokens.masked_fill(mask, mask_token_id)
        return masked_tokens, mask

    def _apply_center_mask(self, tokens: torch.Tensor, mask_token_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Center-region masking for validation (domain-relevant for medical imaging).
        
        Masks a centered rectangular region, useful for evaluating the model's
        ability to inpaint anatomically relevant central structures.
        """
        B, S = tokens.shape
        H = W = self.seq_hw
        device = tokens.device
        
        # Mask center ~33% of each dimension to preserve more context
        margin_h = H // 6
        margin_w = W // 6
        
        mask = torch.zeros(B, H, W, dtype=torch.bool, device=device)
        mask[:, margin_h:H - margin_h, margin_w:W - margin_w] = True
        mask = mask.view(B, S)
        
        masked_tokens = tokens.masked_fill(mask, mask_token_id)
        return masked_tokens, mask

    def _prepare_inputs(self, indices: torch.Tensor, task: str, use_block_mask: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        l1_idx = indices[:, :, 0]
        l2_idx = indices[:, :, 1]
        l2_all_mask = torch.full_like(l2_idx, self.mask_token_id_l2)
        l2_all_mask = torch.full_like(l2_idx, self.mask_token_id_l2)
        batch_size = indices.size(0)

        if task == "l1":
            if use_block_mask:
                l1_in, mask = self._apply_block_mask(l1_idx, self.mask_token_id_l1, task)
            else:
                l1_in, mask = self._apply_mask(l1_idx, self.mask_token_id_l1, task)
            l2_in = torch.full_like(l2_idx, self.mask_token_id_l2)
            target = l1_idx
            task_id = 0
        elif task == "l2":
            l1_in = l1_idx
            if use_block_mask:
                l2_in, mask = self._apply_block_mask(l2_idx, self.mask_token_id_l2, task)
            else:
                l2_in, mask = self._apply_mask(l2_idx, self.mask_token_id_l2, task)
            target = l2_idx
            task_id = 1
        else:
            raise ValueError(f"Unknown task {task}")

        # Add task embedding for multi-task disambiguation
        task_ids = torch.full((batch_size,), task_id, dtype=torch.long, device=indices.device)
        task_emb = self.task_embed(task_ids).unsqueeze(1)  # (B, 1, D) -> broadcast to (B, seq_len, D)
        
        # No learned pos_embed - 3D RoPE handles position encoding in transformer
        tokens = self.l1_embed(l1_in) + self.l2_embed(l2_in) + task_emb
        return tokens, target, mask

    def forward(
        self,
        indices: torch.Tensor,
        task: str = "l1",
        use_block_mask: bool = False,
        slice_pos: Optional[torch.Tensor] = None,
    ):
        """Forward pass with optional slice position for 3D anatomical context.
        
        Args:
            indices: (B, seq_len, 2) token indices from Stage-1 RVQ
            task: "l1" for structure tokens, "l2" for texture tokens
            use_block_mask: Whether to use block-wise masking
            slice_pos: (B,) slice indices for 3D positional encoding.
                       If None, defaults to 0 (backward compatible).
        """
        tokens, target, mask = self._prepare_inputs(indices, task, use_block_mask)
        hidden = self.transformer(tokens, slice_pos=slice_pos)
        logits = self.head_l1(hidden) if task == "l1" else self.head_l2(hidden)
        return logits, target, mask

    @staticmethod
    def _masked_ce(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, label_smoothing: float = 0.0) -> torch.Tensor:
        # If no tokens are masked (rare when ratio is small), fall back to all tokens
        flat_mask = mask.reshape(-1)
        flat_target = target.reshape(-1)
        flat_logits = logits.reshape(-1, logits.size(-1))
        if flat_mask.sum() == 0:
            return F.cross_entropy(flat_logits, flat_target, label_smoothing=label_smoothing)
        sel_logits = flat_logits[flat_mask]
        sel_target = flat_target[flat_mask]
        return F.cross_entropy(sel_logits, sel_target, label_smoothing=label_smoothing)

    @staticmethod
    def _extract_slice_indices(paths: list, device: torch.device) -> torch.Tensor:
        """Extract slice indices from file paths (e.g., 'patient_slice_045.npy' -> 45).
        
        Args:
            paths: List of file paths containing '_slice_XXX' pattern
            device: Target device for the output tensor
            
        Returns:
            Tensor of shape (B,) with slice indices, defaults to 0 if parsing fails
        """
        import os
        import re
        
        slice_indices = []
        pattern = re.compile(r'_slice_([0-9]+)')
        
        for path in paths:
            basename = os.path.basename(path) if isinstance(path, str) else str(path)
            match = pattern.search(basename)
            if match:
                slice_indices.append(int(match.group(1)))
            else:
                # Default to 0 if pattern not found (backward compatibility)
                slice_indices.append(0)
        
        return torch.tensor(slice_indices, dtype=torch.long, device=device)

    def _filter_training_slices(
        self,
        images: torch.Tensor,
        paths: Optional[list],
        slice_pos: Optional[torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], Optional[list], Optional[torch.Tensor]]:
        """Keep only slices within the configured range during training.

        Returns None triplet if no samples remain after filtering.
        """
        if slice_pos is None:
            return images, paths, slice_pos

        keep = (slice_pos >= self.train_slice_min) & (slice_pos <= self.train_slice_max)
        if keep.all():
            return images, paths, slice_pos
        if keep.sum() == 0:
            return None, None, None

        images = images[keep]
        slice_pos = slice_pos[keep]
        if paths is not None:
            paths = [p for p, k in zip(paths, keep.tolist()) if k]
        return images, paths, slice_pos

    def training_step(self, batch, batch_idx):
        # Support dict batches from DataModule ("image" and "path" keys)
        if isinstance(batch, dict):
            images = batch["image"]
            paths = batch.get("path", None)
        elif isinstance(batch, (list, tuple)):
            images = batch[0]
            paths = batch[1] if len(batch) > 1 else None
        else:
            images = batch
            paths = None
            
        if self.stage1 is None:
            raise RuntimeError("Stage1 model must be provided for training Stage2")

        # Extract slice indices for 3D positional encoding
        if paths is not None:
            slice_pos = self._extract_slice_indices(paths, images.device)
        else:
            slice_pos = None

        # Filter out-of-range slices for training only
        images, paths, slice_pos = self._filter_training_slices(images, paths, slice_pos)
        if images is None:
            # Entire batch skipped; log and return zero loss to keep graph valid
            self.log("train/skipped_out_of_range", 1.0, prog_bar=True, on_step=True, batch_size=1)
            return torch.zeros((), device=self.device, requires_grad=True)

        with torch.no_grad():
            _, _, indices, _ = self.stage1.encode_tokens(images)

        # One-time sanity check that codebook sizes match
        self._validate_codebook_sizes(indices)

        # Alternate between random and block masking for diverse training
        use_block = torch.rand(()) < 0.5
        
        logits_l1, target_l1, mask_l1 = self(indices, task="l1", use_block_mask=use_block, slice_pos=slice_pos)
        logits_l2, target_l2, mask_l2 = self(indices, task="l2", use_block_mask=use_block, slice_pos=slice_pos)

        loss_l1 = self._masked_ce(logits_l1, target_l1, mask_l1, self.label_smoothing)
        loss_l2 = self._masked_ce(logits_l2, target_l2, mask_l2, self.label_smoothing)
        loss = loss_l1 + self.l2_loss_weight * loss_l2
        
        # Update token frequency tracking
        with torch.no_grad():
            self._update_token_frequency(indices)
        
        # Compute accuracy for monitoring
        with torch.no_grad():
            pred_l1 = logits_l1.argmax(dim=-1)
            pred_l2 = logits_l2.argmax(dim=-1)
            acc_l1 = (pred_l1[mask_l1] == target_l1[mask_l1]).float().mean()
            acc_l2 = (pred_l2[mask_l2] == target_l2[mask_l2]).float().mean()
            
            # Compute majority-class baseline (what acc would be if always predicting most frequent token)
            baseline_l1 = self._compute_majority_baseline_accuracy(target_l1, mask_l1, "l1")
            baseline_l2 = self._compute_majority_baseline_accuracy(target_l2, mask_l2, "l2")
            
            # Compute "lift" - how much better than random guessing
            lift_l1 = acc_l1 - baseline_l1
            lift_l2 = acc_l2 - baseline_l2

        log_dict = {
            "train/loss": loss,
            "train/loss_l1": loss_l1,
            "train/loss_l2": loss_l2,
            "train/acc_l1": acc_l1,
            "train/acc_l2": acc_l2,
            "train/mask_rate_l1": mask_l1.float().mean(),
            "train/mask_rate_l2": mask_l2.float().mean(),
            "train/baseline_l1": baseline_l1,
            "train/baseline_l2": baseline_l2,
            "train/lift_l1": lift_l1,
            "train/lift_l2": lift_l2,
        }
        
        # Periodically log detailed frequency stats
        if batch_idx % self._frequency_log_interval == 0 and self._total_tokens_seen > 0:
            freq_stats = self.get_token_frequency_stats()
            log_dict.update({
                "train/l1_majority_baseline": freq_stats["l1_majority_baseline_acc"],
                "train/l2_majority_baseline": freq_stats["l2_majority_baseline_acc"],
                "train/l1_codebook_utilization": freq_stats["l1_codebook_utilization"],
                "train/l2_codebook_utilization": freq_stats["l2_codebook_utilization"],
                "train/l1_entropy": freq_stats["l1_entropy"],
                "train/l2_entropy": freq_stats["l2_entropy"],
            })
        
        self.log_dict(log_dict, prog_bar=True, on_step=True, on_epoch=True, batch_size=images.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        # Support dict batches from DataModule ("image" and "path" keys)
        if isinstance(batch, dict):
            images = batch["image"]
            paths = batch.get("path", None)
        elif isinstance(batch, (list, tuple)):
            images = batch[0]
            paths = batch[1] if len(batch) > 1 else None
        else:
            images = batch
            paths = None
            
        if self.stage1 is None:
            raise RuntimeError("Stage1 model must be provided for validation Stage2")

        # Extract slice indices for 3D positional encoding
        if paths is not None:
            slice_pos = self._extract_slice_indices(paths, images.device)
        else:
            slice_pos = None

        with torch.no_grad():
            _, _, indices, _ = self.stage1.encode_tokens(images)

        # Standard random masking for validation
        logits_l1, target_l1, mask_l1 = self(indices, task="l1", use_block_mask=False, slice_pos=slice_pos)
        logits_l2, target_l2, mask_l2 = self(indices, task="l2", use_block_mask=False, slice_pos=slice_pos)

        loss_l1 = self._masked_ce(logits_l1, target_l1, mask_l1, label_smoothing=0.0)  # No smoothing for val
        loss_l2 = self._masked_ce(logits_l2, target_l2, mask_l2, label_smoothing=0.0)
        loss = loss_l1 + self.l2_loss_weight * loss_l2
        
        # Also evaluate center-mask (domain-relevant) with slice position
        l1_idx = indices[:, :, 0]
        l2_idx = indices[:, :, 1]
        l1_center, center_mask = self._apply_center_mask(l1_idx, self.mask_token_id_l1)
        l2_masked = torch.full_like(l2_idx, self.mask_token_id_l2)
        task_emb = self.task_embed(torch.zeros(indices.size(0), dtype=torch.long, device=indices.device)).unsqueeze(1)
        tokens_center = self.l1_embed(l1_center) + self.l2_embed(l2_masked) + task_emb
        hidden_center = self.transformer(tokens_center, slice_pos=slice_pos)
        logits_center = self.head_l1(hidden_center)
        loss_center = self._masked_ce(logits_center, l1_idx, center_mask, label_smoothing=0.0)
        
        # Accuracy metrics
        pred_l1 = logits_l1.argmax(dim=-1)
        pred_l2 = logits_l2.argmax(dim=-1)
        pred_center = logits_center.argmax(dim=-1)
        acc_l1 = (pred_l1[mask_l1] == target_l1[mask_l1]).float().mean()
        acc_l2 = (pred_l2[mask_l2] == target_l2[mask_l2]).float().mean()
        acc_center = (pred_center[center_mask] == l1_idx[center_mask]).float().mean()
        
        # Compute majority-class baselines for validation
        baseline_l1 = self._compute_majority_baseline_accuracy(target_l1, mask_l1, "l1")
        baseline_l2 = self._compute_majority_baseline_accuracy(target_l2, mask_l2, "l2")
        baseline_center = self._compute_majority_baseline_accuracy(l1_idx, center_mask, "l1")
        
        # Compute lift over baseline
        lift_l1 = acc_l1 - baseline_l1
        lift_l2 = acc_l2 - baseline_l2
        lift_center = acc_center - baseline_center

        self.log_dict({
            "val/loss": loss,
            "val/loss_l1": loss_l1,
            "val/loss_l2": loss_l2,
            "val/loss_center": loss_center,
            "val/acc_l1": acc_l1,
            "val/acc_l2": acc_l2,
            "val/acc_center": acc_center,
            "val/baseline_l1": baseline_l1,
            "val/baseline_l2": baseline_l2,
            "val/baseline_center": baseline_center,
            "val/lift_l1": lift_l1,
            "val/lift_l2": lift_l2,
            "val/lift_center": lift_center,
        }, prog_bar=True, on_epoch=True, batch_size=images.size(0))
        return loss

    @torch.no_grad()
    def compute_anomaly_map(
        self,
        images: torch.Tensor,
        slice_pos: Optional[torch.Tensor] = None,
    ) -> dict:
        """Compute anomaly map with optional 3D slice position encoding.
        
        Args:
            images: (B, C, H, W) input images
            slice_pos: (B,) slice indices for 3D anatomical context.
                       If None, defaults to 0 (single-slice mode).
        """
        if self.stage1 is None:
            raise RuntimeError("Stage1 model required for anomaly scoring")
        self.stage1.eval()
        tok_e, tok_q, indices, _ = self.stage1.encode_tokens(images)
        h = w = self.seq_hw
        batch_size = images.size(0)
        scale = self.patch_size  # dynamic upsampling factor

        # Quantization error map (computed from Stage-1 pre-quant vs quantized tokens)
        q_error = (tok_e - tok_q).pow(2).sum(-1)

        # Anatomy NLL (mask all L1 tokens)
        l1_in = torch.full_like(indices[:, :, 0], self.mask_token_id_l1)
        l2_in = torch.full_like(indices[:, :, 1], self.mask_token_id_l2)
        task_ids_l1 = torch.zeros(batch_size, dtype=torch.long, device=images.device)
        task_emb_l1 = self.task_embed(task_ids_l1).unsqueeze(1)
        # 3D RoPE handles position encoding in transformer
        tokens_anat = self.l1_embed(l1_in) + self.l2_embed(l2_in) + task_emb_l1
        hidden = self.transformer(tokens_anat, slice_pos=slice_pos)
        logits_l1 = self.head_l1(hidden)
        log_probs_l1 = F.log_softmax(logits_l1, dim=-1)
        nll_l1 = -log_probs_l1.gather(-1, indices[:, :, 0:1]).squeeze(-1)

        # Texture NLL (mask all L2 tokens, keep L1 visible)
        l1_vis = indices[:, :, 0]
        l2_mask = torch.full_like(indices[:, :, 1], self.mask_token_id_l2)
        task_ids_l2 = torch.ones(batch_size, dtype=torch.long, device=images.device)
        task_emb_l2 = self.task_embed(task_ids_l2).unsqueeze(1)
        tokens_tex = self.l1_embed(l1_vis) + self.l2_embed(l2_mask) + task_emb_l2
        hidden_tex = self.transformer(tokens_tex, slice_pos=slice_pos)
        logits_l2 = self.head_l2(hidden_tex)
        log_probs_l2 = F.log_softmax(logits_l2, dim=-1)
        nll_l2 = -log_probs_l2.gather(-1, indices[:, :, 1:2]).squeeze(-1)

        # Z-score normalize each component per-image to balance scales
        def zscore(x: torch.Tensor) -> torch.Tensor:
            """Z-score normalize over spatial dims (per image)."""
            mean = x.mean(dim=-1, keepdim=True)
            std = x.std(dim=-1, keepdim=True).clamp(min=1e-6)
            return (x - mean) / std

        nll_l1_z = zscore(nll_l1)
        nll_l2_z = zscore(nll_l2)
        q_error_z = zscore(q_error)

        # Combine with weighting (z-scored so scales are comparable)
        if self.use_learnable_weights and self.anomaly_weights is not None:
            anomaly_tokens = self.anomaly_weights(nll_l1_z, nll_l2_z, q_error_z)
        else:
            anomaly_tokens = nll_l1_z + self.l2_loss_weight * nll_l2_z + self.q_error_weight * q_error_z
        maps = {
            "anomaly_map": F.interpolate(anomaly_tokens.reshape(-1, 1, h, w), scale_factor=scale, mode="bilinear", align_corners=False),
            "nll_l1_map": F.interpolate(nll_l1.reshape(-1, 1, h, w), scale_factor=scale, mode="bilinear", align_corners=False),
            "nll_l2_map": F.interpolate(nll_l2.reshape(-1, 1, h, w), scale_factor=scale, mode="bilinear", align_corners=False),
            "q_error_map": F.interpolate(q_error.reshape(-1, 1, h, w), scale_factor=scale, mode="bilinear", align_corners=False),
        }
        return maps

    @torch.no_grad()
    def compute_anomaly_map_sliding(
        self,
        images: torch.Tensor,
        window_size: int = 4,
        stride: int = 2,
        num_monte_carlo: int = 8,
        slice_pos: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Sliding-window contextual scoring that preserves local evidence.

        Each window hides a local patch, predicts with surrounding context, and aggregates
        negative log-likelihoods across overlapping windows and Monte Carlo passes.
        
        Args:
            images: (B, C, H, W) input images
            window_size: Size of the sliding window in token space
            stride: Stride of the sliding window
            num_monte_carlo: Number of Monte Carlo passes for dropout masking
            slice_pos: (B,) slice indices for 3D anatomical context
        """
        if self.stage1 is None:
            raise RuntimeError("Stage1 model required for anomaly scoring")

        self.stage1.eval()
        tok_e, tok_q, indices, _ = self.stage1.encode_tokens(images)
        batch_size, seq_len = indices.shape[:2]
        h = w = self.seq_hw
        scale = self.patch_size

        l1_idx = indices[:, :, 0]
        l2_idx = indices[:, :, 1]

        l2_all_mask = torch.full_like(l2_idx, self.mask_token_id_l2)

        nll_l1_accum = torch.zeros(batch_size, seq_len, device=indices.device)
        nll_l2_accum = torch.zeros_like(nll_l1_accum)
        count_map = torch.zeros_like(nll_l1_accum)

        for row_start in range(0, h - window_size + 1, stride):
            for col_start in range(0, w - window_size + 1, stride):
                mask = torch.zeros(batch_size, h, w, dtype=torch.bool, device=indices.device)
                mask[:, row_start:row_start + window_size, col_start:col_start + window_size] = True
                mask_flat = mask.view(batch_size, seq_len)
                if num_monte_carlo > 1:
                    dropout_mask = torch.rand_like(mask_flat.float()) > 0.2
                    active_mask = mask_flat & dropout_mask
                    active_mask = torch.where(
                        active_mask.sum(dim=1, keepdim=True) > 0,
                        active_mask,
                        mask_flat,
                    )
                else:
                    active_mask = mask_flat

                for _ in range(num_monte_carlo):
                    l1_masked = l1_idx.masked_fill(active_mask, self.mask_token_id_l1)
                    task_emb_l1 = self.task_embed(torch.zeros(batch_size, dtype=torch.long, device=indices.device)).unsqueeze(1)
                    # Align with training: L1 prediction uses fully masked L2 context
                    tokens_l1 = self.l1_embed(l1_masked) + self.l2_embed(l2_all_mask) + task_emb_l1
                    hidden_l1 = self.transformer(tokens_l1, slice_pos=slice_pos)
                    logits_l1 = self.head_l1(hidden_l1)
                    log_probs_l1 = F.log_softmax(logits_l1, dim=-1)
                    nll_l1 = -log_probs_l1.gather(-1, l1_idx.unsqueeze(-1)).squeeze(-1)
                    nll_l1_accum += nll_l1 * active_mask.float()

                    l2_masked = l2_idx.masked_fill(active_mask, self.mask_token_id_l2)
                    task_emb_l2 = self.task_embed(torch.ones(batch_size, dtype=torch.long, device=indices.device)).unsqueeze(1)
                    tokens_l2 = self.l1_embed(l1_idx) + self.l2_embed(l2_masked) + task_emb_l2
                    hidden_l2 = self.transformer(tokens_l2, slice_pos=slice_pos)
                    logits_l2 = self.head_l2(hidden_l2)
                    log_probs_l2 = F.log_softmax(logits_l2, dim=-1)
                    nll_l2 = -log_probs_l2.gather(-1, l2_idx.unsqueeze(-1)).squeeze(-1)
                    nll_l2_accum += nll_l2 * active_mask.float()

                    count_map += active_mask.float()

        count_safe = count_map.clamp(min=1.0)
        nll_l1_avg = nll_l1_accum / count_safe
        nll_l2_avg = nll_l2_accum / count_safe

        q_error = (tok_e - tok_q).pow(2).sum(-1)

        def zscore(x: torch.Tensor) -> torch.Tensor:
            mean = x.mean(dim=-1, keepdim=True)
            std = x.std(dim=-1, keepdim=True).clamp(min=1e-6)
            return (x - mean) / std

        nll_l1_z = zscore(nll_l1_avg)
        nll_l2_z = zscore(nll_l2_avg)
        q_error_z = zscore(q_error)

        if self.use_learnable_weights and self.anomaly_weights is not None:
            anomaly_tokens = self.anomaly_weights(nll_l1_z, nll_l2_z, q_error_z)
        else:
            anomaly_tokens = nll_l1_z + self.l2_loss_weight * nll_l2_z + self.q_error_weight * q_error_z

        coverage = count_safe.view(batch_size, h, w)
        maps = {
            "anomaly_map": F.interpolate(anomaly_tokens.reshape(-1, 1, h, w), scale_factor=scale, mode="bilinear", align_corners=False),
            "nll_l1_map": F.interpolate(nll_l1_avg.reshape(-1, 1, h, w), scale_factor=scale, mode="bilinear", align_corners=False),
            "nll_l2_map": F.interpolate(nll_l2_avg.reshape(-1, 1, h, w), scale_factor=scale, mode="bilinear", align_corners=False),
            "q_error_map": F.interpolate(q_error.reshape(-1, 1, h, w), scale_factor=scale, mode="bilinear", align_corners=False),
            "coverage": coverage,
        }
        return maps

    @torch.no_grad()
    def compute_multiscale_anomaly(self, images: torch.Tensor) -> dict:
        """
        Multi-scale anomaly scoring inspired by feature pyramid aggregation.

        Requires Stage-1 to expose encode_multiscale, returning fused tokens and
        per-scale token grids (stride 1/2/4). Each scale uses quantization error
        as a proxy energy, upsampled to input resolution and fused with learnable weights.
        """
        if self.stage1 is None or not hasattr(self.stage1, "encode_multiscale"):
            raise RuntimeError("Stage1 with multiscale encoder required for multi-scale anomaly scoring")

        self.stage1.eval()
        tokens, quantized, _, _, scale_features = self.stage1.encode_multiscale(images)
        h = w = self.seq_hw
        patch = self.patch_size

        q_error = (tokens - quantized).pow(2).sum(-1)
        q_error_map = F.interpolate(q_error.reshape(-1, 1, h, w), scale_factor=patch, mode="bilinear", align_corners=False)

        scale_maps = []
        for scale_idx, feat in enumerate(scale_features):
            quantized_s, _, _ = self.stage1.quantizer(feat)
            q_err_s = (feat - quantized_s).pow(2).sum(-1)
            s_len = q_err_s.shape[1]
            s_hw = int(math.sqrt(s_len))
            scale_map = q_err_s.view(-1, 1, s_hw, s_hw)
            upsample_factor = patch * (2 ** scale_idx)
            upsampled = F.interpolate(scale_map, scale_factor=upsample_factor, mode="bilinear", align_corners=False)
            scale_maps.append(upsampled)

        def zscore_map(x: torch.Tensor) -> torch.Tensor:
            mean = x.mean(dim=(1, 2, 3), keepdim=True)
            std = x.std(dim=(1, 2, 3), keepdim=True).clamp(min=1e-6)
            return (x - mean) / std

        weights = F.softmax(self.scale_weights[: len(scale_maps)], dim=0)
        scale_maps_z = [zscore_map(m) for m in scale_maps]
        fused_map = sum(w * m for w, m in zip(weights, scale_maps_z))
        fused_map = fused_map + self.q_error_weight * zscore_map(q_error_map)

        return {
            "anomaly_map": fused_map,
            "scale_maps": scale_maps,
            "weights": weights,
            "q_error_map": q_error_map,
        }

    @torch.no_grad()
    def compute_anomaly_map_contextual(
        self,
        images: torch.Tensor,
        mask_ratio: float = 0.15,
        slice_pos: Optional[torch.Tensor] = None,
    ) -> dict:
        """Contextual anomaly scoring: mask a subset to preserve neighborhood context.
        
        Args:
            images: (B, C, H, W) input images
            mask_ratio: Ratio of tokens to mask for contextual scoring
            slice_pos: (B,) slice indices for 3D anatomical context
        """
        if self.stage1 is None:
            raise RuntimeError("Stage1 model required for anomaly scoring")

        self.stage1.eval()
        tok_e, tok_q, indices, _ = self.stage1.encode_tokens(images)
        h = w = self.seq_hw
        batch_size, S = indices.shape[:2]
        scale = self.patch_size

        # Shared mask for both heads (could split, but keep simple/consistent)
        mask = (torch.rand(batch_size, S, device=indices.device) < mask_ratio)

        # Quantization error map
        q_error = (tok_e - tok_q).pow(2).sum(-1)

        # L1 NLL with contextual masking
        l1_idx = indices[:, :, 0]
        l2_all_mask = torch.full_like(indices[:, :, 1], self.mask_token_id_l2)
        l1_masked = l1_idx.masked_fill(mask, self.mask_token_id_l1)

        task_ids_l1 = torch.zeros(batch_size, dtype=torch.long, device=indices.device)
        task_emb_l1 = self.task_embed(task_ids_l1).unsqueeze(1)
        # Align with training: L1 prediction never sees real L2 tokens
        tokens_anat = self.l1_embed(l1_masked) + self.l2_embed(l2_all_mask) + task_emb_l1
        hidden = self.transformer(tokens_anat, slice_pos=slice_pos)
        logits_l1 = self.head_l1(hidden)
        log_probs_l1 = F.log_softmax(logits_l1, dim=-1)
        nll_l1 = -log_probs_l1.gather(-1, l1_idx.unsqueeze(-1)).squeeze(-1)
        nll_l1_masked = nll_l1 * mask.float()

        # L2 NLL with contextual masking (L1 visible)
        l2_idx = indices[:, :, 1]
        l2_masked = l2_idx.masked_fill(mask, self.mask_token_id_l2)
        task_ids_l2 = torch.ones(batch_size, dtype=torch.long, device=indices.device)
        task_emb_l2 = self.task_embed(task_ids_l2).unsqueeze(1)
        tokens_tex = self.l1_embed(l1_idx) + self.l2_embed(l2_masked) + task_emb_l2
        hidden_tex = self.transformer(tokens_tex, slice_pos=slice_pos)
        logits_l2 = self.head_l2(hidden_tex)
        log_probs_l2 = F.log_softmax(logits_l2, dim=-1)
        nll_l2 = -log_probs_l2.gather(-1, l2_idx.unsqueeze(-1)).squeeze(-1)
        nll_l2_masked = nll_l2 * mask.float()

        # Normalize over spatial dims
        def zscore(x: torch.Tensor) -> torch.Tensor:
            mean = x.mean(dim=-1, keepdim=True)
            std = x.std(dim=-1, keepdim=True).clamp(min=1e-6)
            return (x - mean) / std

        nll_l1_z = zscore(nll_l1_masked)
        nll_l2_z = zscore(nll_l2_masked)
        q_error_z = zscore(q_error)

        if self.use_learnable_weights and self.anomaly_weights is not None:
            anomaly_tokens = self.anomaly_weights(nll_l1_z, nll_l2_z, q_error_z)
        else:
            anomaly_tokens = nll_l1_z + self.l2_loss_weight * nll_l2_z + self.q_error_weight * q_error_z

        maps = {
            "anomaly_map": F.interpolate(anomaly_tokens.reshape(-1, 1, h, w), scale_factor=scale, mode="bilinear", align_corners=False),
            "nll_l1_map": F.interpolate(nll_l1_masked.reshape(-1, 1, h, w), scale_factor=scale, mode="bilinear", align_corners=False),
            "nll_l2_map": F.interpolate(nll_l2_masked.reshape(-1, 1, h, w), scale_factor=scale, mode="bilinear", align_corners=False),
            "q_error_map": F.interpolate(q_error.reshape(-1, 1, h, w), scale_factor=scale, mode="bilinear", align_corners=False),
            "mask": mask.view(batch_size, h, w),
        }
        return maps

    @torch.no_grad()
    def compute_anomaly_map_iterative(
        self,
        images: torch.Tensor,
        num_steps: int = 6,
        initial_mask_ratio: float = 0.70,
        slice_pos: Optional[torch.Tensor] = None,
    ) -> dict:
        """Anomaly scoring that reuses iterative refinement for both heads.
        
        Args:
            images: (B, C, H, W) input images
            num_steps: Number of iterative refinement steps
            initial_mask_ratio: Initial ratio of masked tokens
            slice_pos: (B,) slice indices for 3D anatomical context
        """
        if self.stage1 is None:
            raise RuntimeError("Stage1 model required for anomaly scoring")

        self.stage1.eval()
        tok_e, tok_q, indices, _ = self.stage1.encode_tokens(images)
        h = w = self.seq_hw
        scale = self.patch_size

        q_error = (tok_e - tok_q).pow(2).sum(-1)

        _, probs_l1 = self.iterative_predict(
            indices, task="l1", num_steps=num_steps, initial_mask_ratio=initial_mask_ratio, slice_pos=slice_pos
        )
        _, probs_l2 = self.iterative_predict(
            indices, task="l2", num_steps=num_steps, initial_mask_ratio=initial_mask_ratio, slice_pos=slice_pos
        )

        log_probs_l1 = probs_l1.clamp_min(1e-8).log()
        log_probs_l2 = probs_l2.clamp_min(1e-8).log()

        nll_l1 = -log_probs_l1.gather(-1, indices[:, :, 0:1]).squeeze(-1)
        nll_l2 = -log_probs_l2.gather(-1, indices[:, :, 1:2]).squeeze(-1)

        def zscore(x: torch.Tensor) -> torch.Tensor:
            mean = x.mean(dim=-1, keepdim=True)
            std = x.std(dim=-1, keepdim=True).clamp(min=1e-6)
            return (x - mean) / std

        nll_l1_z = zscore(nll_l1)
        nll_l2_z = zscore(nll_l2)
        q_error_z = zscore(q_error)

        if self.use_learnable_weights and self.anomaly_weights is not None:
            anomaly_tokens = self.anomaly_weights(nll_l1_z, nll_l2_z, q_error_z)
        else:
            anomaly_tokens = nll_l1_z + self.l2_loss_weight * nll_l2_z + self.q_error_weight * q_error_z

        maps = {
            "anomaly_map": F.interpolate(anomaly_tokens.reshape(-1, 1, h, w), scale_factor=scale, mode="bilinear", align_corners=False),
            "nll_l1_map": F.interpolate(nll_l1.reshape(-1, 1, h, w), scale_factor=scale, mode="bilinear", align_corners=False),
            "nll_l2_map": F.interpolate(nll_l2.reshape(-1, 1, h, w), scale_factor=scale, mode="bilinear", align_corners=False),
            "q_error_map": F.interpolate(q_error.reshape(-1, 1, h, w), scale_factor=scale, mode="bilinear", align_corners=False),
        }
        return maps

    @torch.no_grad()
    def iterative_predict(
        self,
        indices: torch.Tensor,
        task: str = "l1",
        num_steps: int = 8,
        initial_mask_ratio: float = 0.75,
        schedule: str = "cosine",
        slice_pos: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Iterative refinement that starts with partial context visible.
        
        Args:
            indices: (B, seq_len, 2) token indices
            task: "l1" or "l2" for which head to predict
            num_steps: Number of refinement iterations
            initial_mask_ratio: Initial ratio of masked tokens
            schedule: "cosine" or "random" for mask initialization
            slice_pos: (B,) slice indices for 3D anatomical context
        """
        if task not in {"l1", "l2"}:
            raise ValueError("task must be 'l1' or 'l2'")

        mask_token_id = self.mask_token_id_l1 if task == "l1" else self.mask_token_id_l2
        head = self.head_l1 if task == "l1" else self.head_l2

        B, S = indices.shape[:2]
        if task == "l1":
            # Align with training: L1 prediction sees L2 fully masked
            other_tokens = torch.full_like(indices[:, :, 1], self.mask_token_id_l2)
            task_id = 0
            ground_truth = indices[:, :, 0]
        else:
            other_tokens = indices[:, :, 0]
            task_id = 1
            ground_truth = indices[:, :, 1]

        task_emb = self.task_embed(
            torch.full((B,), task_id, dtype=torch.long, device=indices.device)
        ).unsqueeze(1)

        if schedule == "cosine":
            num_keep = max(1, int(S * (1.0 - initial_mask_ratio)))
            visible = torch.zeros(B, S, dtype=torch.bool, device=indices.device)
            for b in range(B):
                keep_idx = torch.randperm(S, device=indices.device)[:num_keep]
                visible[b, keep_idx] = True
        elif schedule == "random":
            visible = torch.rand(B, S, device=indices.device) < (1.0 - initial_mask_ratio)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

        current = torch.where(
            visible,
            ground_truth,
            torch.full_like(ground_truth, mask_token_id),
        )

        max_probs_final = torch.zeros_like(ground_truth, dtype=torch.float)

        for _ in range(num_steps):
            if task == "l1":
                tokens = self.l1_embed(current) + self.l2_embed(other_tokens) + task_emb
            else:
                tokens = self.l1_embed(other_tokens) + self.l2_embed(current) + task_emb

            hidden = self.transformer(tokens, slice_pos=slice_pos)
            logits = head(hidden)
            probs = F.softmax(logits, dim=-1)

            max_probs, pred_tokens = probs.max(dim=-1)
            masked_pos = (current == mask_token_id)
            if masked_pos.sum() == 0:
                max_probs_final = max_probs
                break

            tokens_to_unmask = (masked_pos.sum(dim=-1).float() / num_steps).long().clamp(min=1)

            for b in range(B):
                mask_b = masked_pos[b].nonzero(as_tuple=False).squeeze(-1)
                if mask_b.numel() == 0:
                    continue
                probs_b = max_probs[b, mask_b]
                num_to_fill = min(tokens_to_unmask[b].item(), mask_b.numel())
                _, sorted_idx = probs_b.topk(num_to_fill, largest=True)
                fill_pos = mask_b[sorted_idx]
                current[b, fill_pos] = pred_tokens[b, fill_pos]

            max_probs_final = max_probs

            if (current != mask_token_id).all():
                break

        if task == "l1":
            final_tokens = self.l1_embed(current) + self.l2_embed(other_tokens) + task_emb
        else:
            final_tokens = self.l1_embed(other_tokens) + self.l2_embed(current) + task_emb

        hidden_final = self.transformer(final_tokens, slice_pos=slice_pos)
        logits_final = head(hidden_final)
        probs_final = F.softmax(logits_final, dim=-1)

        return current, probs_final

    def configure_optimizers(self):
        # AdamW with weight decay (excluding bias, norms, and embeddings)
        decay_params = []
        no_decay_params = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if "bias" in name or "norm" in name or "scale" in name or "embed" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        
        optimizer = torch.optim.AdamW([
            {"params": decay_params, "weight_decay": self.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ], lr=self.lr, betas=(0.9, 0.98))  # beta2=0.98 common for transformers
        
        # Warmup + Cosine annealing scheduler
        def lr_lambda(step: int) -> float:
            if step < self.warmup_steps:
                # Linear warmup
                return step / max(1, self.warmup_steps)
            else:
                # Cosine decay after warmup
                progress = (step - self.warmup_steps) / max(1, self.trainer.estimated_stepping_batches - self.warmup_steps)
                return 0.5 * (1.0 + math.cos(math.pi * progress))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
