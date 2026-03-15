# -*- coding: utf-8 -*-
"""
3DCvT Model Architecture for Lip Reading.

Description:
    Implements the lip reading network described in:
    "A Lip Reading Method Based on 3D Convolutional Vision Transformer".

Architecture Overview:
    1. Front-end: 3D-CNN (Spatio-temporal feature extraction).
    2. Backbone: CvT with SE-Blocks (Stage 1, 2, 3).
    3. Back-end: BiGRU with Word Boundary Concatenation.
    4. Classifier: FC Layer.

Author: Jiafeng Wu (Reproducing 3DCvT)
Environment: Python 3.10, PyTorch 2.x
Dependencies: torch, einops
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from typing import Tuple, List
from torch.nn.init import trunc_normal_

# --------------------------------------------------------------------------------
# 1. Basic Components (SE-Block, MLP)
# --------------------------------------------------------------------------------

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Module [29].
    Used inside the Convolutional Token Embedding as per Formula (3).
    """
    def __init__(self, channel: int, reduction: int = 4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        # Scale operation: F_scale(x, y)
        return x * y.expand_as(x)


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample.
    
    Paper Ref: "Deep Networks with Stochastic Depth" (Huang et al., 2016).
    Essential regularizer for deep Transformer stacks (Stage 3 has 20 blocks).
    """
    def __init__(self, drop_prob: float = 0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        # Create random tensor with shape (batch_size, 1, 1, ...) for broadcasting
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = torch.floor(random_tensor + keep_prob)
        output = x / keep_prob * random_tensor
        return output

    def extra_repr(self) -> str:
        return f'drop_prob={self.drop_prob:.3f}'


class Mlp(nn.Module):
    """
    Multilayer Perceptron used in Transformer Block.
    """
    def __init__(self, in_features: int, hidden_features: int = None, out_features: int = None, drop: float = 0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# --------------------------------------------------------------------------------
# 2. CvT Specific Components (SE-Conv-Embed, ConvProjection, TransformerBlock)
# --------------------------------------------------------------------------------

class SEConvEmbedding(nn.Module):
    """
    Modified Convolutional Token Embedding.
    Integrates 2D Convolution + SE Block.
    
    Paper Ref: "We add the Squeezing and Excitation (SE) module to the 
    Convolutional Token Embedding... to improve fine-grained extraction."
    """
    def __init__(self, in_chans: int, embed_dim: int, kernel_size: int, stride: int, padding: int):
        super().__init__()
        self.conv = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding)
        self.se = SEBlock(embed_dim)
        self.norm = nn.LayerNorm(embed_dim) # LayerNorm is standard for Tokens

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C_in, H, W) -> (B, Embed_Dim, H', W')
        """
        x = self.conv(x)
        x = self.se(x) # Apply SE Mechanism
        
        # Reshape for LayerNorm: (B, C, H, W) -> (B, H, W, C)
        x = rearrange(x, 'b c h w -> b h w c')
        x = self.norm(x)
        # Back to (B, C, H, W) for next convolution ops in CvT
        x = rearrange(x, 'b h w c -> b c h w')
        return x


class ConvAttention(nn.Module):
    """
    CvT-style Convolutional Attention.
    Uses convolutions to generate Q/K/V, then applies standard Scaled Dot-Product Attention.
    Avoids the double-projection issue caused by nn.MultiheadAttention's internal linear layers.
    """
    def __init__(self, dim: int, num_heads: int, kernel_size: int = 3, attn_drop: float = 0., proj_drop: float = 0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        padding = (kernel_size - 1) // 2
        
        # Depthwise Separable Convolution (proper CvT implementation)
        # Depthwise: groups=dim, extracts local spatial features
        # Pointwise: 1x1 conv, enables cross-channel mixing
        self.conv_proj_q = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size, padding=padding, groups=dim, bias=False),
            nn.BatchNorm2d(dim),
            nn.Conv2d(dim, dim, 1, bias=False),  # Pointwise
        )
        self.conv_proj_k = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size, padding=padding, groups=dim, bias=False),
            nn.BatchNorm2d(dim),
            nn.Conv2d(dim, dim, 1, bias=False),
        )
        self.conv_proj_v = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size, padding=padding, groups=dim, bias=False),
            nn.BatchNorm2d(dim),
            nn.Conv2d(dim, dim, 1, bias=False),
        )
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        x: (B, N, C) where N = H * W
        Returns: (B, N, C)
        """
        B, N, C = x.shape
        
        # Reshape to 2D for conv
        x_2d = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        
        # Conv Projection
        q = rearrange(self.conv_proj_q(x_2d), 'b c h w -> b (h w) c')
        k = rearrange(self.conv_proj_k(x_2d), 'b c h w -> b (h w) c')
        v = rearrange(self.conv_proj_v(x_2d), 'b c h w -> b (h w) c')
        
        # Reshape for multi-head attention: (B, N, C) -> (B, num_heads, N, head_dim)
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.num_heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.num_heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.num_heads)
        
        # Scaled Dot-Product Attention (PyTorch 2.x native implementation)
        # Automatically selects the most efficient backend:
        #   - FlashAttention v2 (if available): O(N) memory, fastest
        #   - Memory-Efficient Attention (xformers): O(sqrt(N)) memory
        #   - Math fallback: standard O(N^2) if neither is available
        # This replaces the manual q @ k^T -> softmax -> @ v pipeline.
        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop.p if self.training else 0.0,
        )  # (B, heads, N, head_dim)
        out = rearrange(out, 'b h n d -> b n (h d)')
        
        # Output projection
        out = self.proj(out)
        out = self.proj_drop(out)
        
        return out


class TransformerBlock(nn.Module):
    """
    CvT Transformer Block.
    Contains:
    - Convolutional Projection (for Q, K, V) - Formula (4)
    - Multi-Head Self Attention
    - MLP
    """
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4., qkv_bias: bool = False, 
                 drop: float = 0., attn_drop: float = 0., drop_path: float = 0., kernel_size: int = 3):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        
        # Custom ConvAttention to avoid nn.MultiheadAttention's double-projection issue
        self.attn = ConvAttention(
            dim=dim, 
            num_heads=num_heads, 
            kernel_size=kernel_size,
            attn_drop=attn_drop,
            proj_drop=drop
        )
        
        # DropPath (Stochastic Depth) — critical regularizer for deep transformers (20 layers in Stage 3)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input x: (B, C, H, W) - Feature Map
        """
        B, C, H, W = x.shape
        
        # 1. Attention Path
        shortcut = x
        # Reshape to Sequence: (B, H*W, C)
        x_seq = rearrange(x, 'b c h w -> b (h w) c')
        x_norm = self.norm1(x_seq)
        
        # Convolutional Attention
        attn_out = self.attn(x_norm, H, W)
        
        # Add Residual with DropPath (reshape back to feature map)
        x_attn = shortcut + self.drop_path(rearrange(attn_out, 'b (h w) c -> b c h w', h=H, w=W))
        
        # 2. MLP Path
        x_tokens = rearrange(x_attn, 'b c h w -> b (h w) c')
        x_mlp = self.mlp(self.norm2(x_tokens))
        
        # Add Residual with DropPath
        x_out = x_attn + self.drop_path(rearrange(x_mlp, 'b (h w) c -> b c h w', h=H, w=W))
        
        return x_out


# --------------------------------------------------------------------------------
# 3. Network Stages & Full Model
# --------------------------------------------------------------------------------

class FrontEnd3D(nn.Module):
    """
    3D-CNN Stem.
    Paper: "The 3D convolution kernel size is 5x7x7, stride 1x2x2, channels 64."
    """
    def __init__(self, in_channels: int = 1, out_channels: int = 64):
        super().__init__()
        # Input: (B, 1, T, 88, 88)
        # Target Output after Conv+Pool: (B, 64, T, 22, 22)
        # Kernel: (5, 7, 7), Stride: (1, 2, 2)
        # Padding calculation to keep T same (5 -> pad 2) and H/W half (7 -> pad 3)
        self.conv3d = nn.Conv3d(
            in_channels, out_channels, 
            kernel_size=(5, 7, 7), 
            stride=(1, 2, 2), 
            padding=(2, 3, 3), 
            bias=False
        )
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool3d(
            kernel_size=(1, 3, 3),
            stride=(1, 2, 2),
            padding=(0, 1, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv3d(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x


class LipReading3DCvT(nn.Module):
    """
    The Complete 3D-CvT Model.
    """
    def __init__(self, num_classes: int = 1000, frame_len: int = 29, use_checkpoint: bool = True,
                 drop_path_rate: float = 0.1, drop_rate: float = 0.1):
        super().__init__()
        
        # --- 1. Front-end ---
        self.frontend = FrontEnd3D(in_channels=1, out_channels=64)

        # --- 2. Backbone (CvT Stages) ---
        # Note: 'num_heads' adjusted to be divisors of 'dim' to ensure runnability.
        # Paper: D1=128, H1=3 (using 4); D2=256, H2=12 (using 8); D3=512, H3=16 (using 16)
        
        # Stochastic Depth: linearly increasing drop path rates across all 24 blocks
        depths = [2, 2, 20]
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        
        # Stage 1
        self.stage1_embed = SEConvEmbedding(64, 128, kernel_size=7, stride=2, padding=3)
        self.stage1_blocks = nn.Sequential(*[
            TransformerBlock(dim=128, num_heads=4, mlp_ratio=4, drop=drop_rate, drop_path=dpr[i])
            for i in range(depths[0])
        ])
        
        # Stage 2
        self.stage2_embed = SEConvEmbedding(128, 256, kernel_size=3, stride=2, padding=1)
        self.stage2_blocks = nn.Sequential(*[
            TransformerBlock(dim=256, num_heads=8, mlp_ratio=4, drop=drop_rate, drop_path=dpr[sum(depths[:1]) + i])
            for i in range(depths[1])
        ])
        
        # Stage 3
        self.stage3_embed = SEConvEmbedding(256, 512, kernel_size=3, stride=2, padding=1)
        # Stage 3: 20 Transformer blocks — use gradient checkpointing to save memory.
        # Trades ~30% extra compute for ~60% memory reduction on intermediate activations.
        self.stage3_blocks = nn.ModuleList([
            TransformerBlock(dim=512, num_heads=16, mlp_ratio=4, drop=drop_rate, drop_path=dpr[sum(depths[:2]) + i])
            for i in range(depths[2])
        ])
        # Gradient checkpointing: trades ~30% extra compute for ~60% memory reduction.
        # WARNING: Incompatible with SyncBatchNorm — DDP scripts must revert SyncBN
        # inside stage3_blocks to regular BatchNorm, or set use_checkpoint=False.
        self.use_checkpoint = use_checkpoint

        # --- 3. Back-end (BiGRU) ---
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Word Boundary Concatenation:
        # A binary mask (B, T, 1) indicating which frames contain the target word.
        # Critical for LRW-1000 where videos have variable-length context around the word.
        # For LRW (fixed 29 frames = entire word), the mask is all-ones.

        # BiGRU
        # Input: 512 + 1 = 513
        self.bigru = nn.GRU(
            input_size=513, 
            hidden_size=1024, 
            num_layers=3, 
            batch_first=True, 
            bidirectional=True
        )
        
        # --- 4. Classifier ---
        # BiGRU output is hidden_size * 2
        self.fc = nn.Linear(1024 * 2, num_classes)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, (nn.Conv2d, nn.Conv3d)):
            # Kaiming initialization for convolutional layers
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, boundary_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: (B, 1, T, 88, 88) Input video batch
            boundary_mask: (B, T, 1) Binary mask indicating target word frames.
                           If None, defaults to all-ones (backward compatible).
        Returns:
            logits: (B, num_classes)
        """
        B, C, T, H, W = x.shape
        
        # 1. Front-End 3D CNN
        # Conv3d(stride=1×2×2): 88 -> 44; MaxPool3d(stride=1×2×2): 44 -> 22
        x = self.frontend(x) # -> (B, 64, T, 22, 22)
        
        # 2. Fold Time Dimension for 2D CvT
        # Transform (B, C, T, H, W) -> (B*T, C, H, W)
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        
        # 3. CvT Backbone (spatial dims: 22 -> 11 -> 6 -> 3)
        # Stage 1
        x = self.stage1_embed(x)   # -> (BT, 128, 11, 11)
        x = self.stage1_blocks(x)
        
        # Stage 2
        x = self.stage2_embed(x)   # -> (BT, 256, 6, 6)
        x = self.stage2_blocks(x)
        
        # Stage 3
        x = self.stage3_embed(x)   # -> (BT, 512, 3, 3)
        for blk in self.stage3_blocks:
            if self.use_checkpoint and self.training:
                x = checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)
        
        # 4. Global Average Pooling & Unfold Time
        x = self.gap(x).flatten(1) # -> (BT, 512)
        x = rearrange(x, '(b t) d -> b t d', b=B, t=T) # -> (B, T, 512)
        
        # 5. Word Boundary Concatenation
        # Append binary boundary mask to indicate which frames contain the target word.
        # For LRW: all-ones (entire clip is the word). For LRW-1000: varies per sample.
        if boundary_mask is None:
            boundary_mask = torch.ones(B, T, 1, device=x.device, dtype=x.dtype)
        x = torch.cat([x, boundary_mask], dim=-1) # -> (B, T, 513)
        
        # 6. BiGRU Sequence Modeling
        self.bigru.flatten_parameters() # Optimize memory
        out, _ = self.bigru(x) # -> (B, T, 2048)
        
        # 7. Classification
        # Typically, we aggregate the time dimension. 
        # Option A: Last Hidden State.
        # Option B: Mean Pooling over Time (Standard for Lip Reading).
        # We use Mean Pooling here as it's robust to length variations.
        out = torch.mean(out, dim=1) # -> (B, 2048)
        
        logits = self.fc(out) # -> (B, num_classes)
        
        return logits

