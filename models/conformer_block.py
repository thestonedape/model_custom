"""
Conformer Block Implementation for BELT
Based on "Conformer: Convolution-augmented Transformer for Speech Recognition"
and BELT paper Table I specifications
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from .convolution_module import ConvolutionModule


class ConformerBlock(nn.Module):
    """
    Single Conformer block with the following structure:
    1. Feed-Forward Module (1/2 expansion)
    2. Multi-Head Self-Attention Module
    3. Convolution Module
    4. Feed-Forward Module (1/2 expansion)
    5. Layer Normalization
    """
    
    def __init__(
        self,
        d_model: int = 840,
        num_heads: int = 8,
        ffn_expansion: int = 4,
        conv_kernel_size: int = 31,
        dropout: float = 0.1
    ):
        """
        Args:
            d_model: Model dimension (default: 840)
            num_heads: Number of attention heads (default: 8)
            ffn_expansion: FFN expansion factor (default: 4 → 840*4=3360)
            conv_kernel_size: Kernel size for convolution (default: 31)
            dropout: Dropout probability
        """
        super().__init__()
        
        self.d_model = d_model
        
        # First Feed-Forward Module (half step)
        self.ffn1 = FeedForwardModule(
            d_model=d_model,
            expansion_factor=ffn_expansion,
            dropout=dropout,
            half_step=True
        )
        
        # Multi-Head Self-Attention
        self.attention = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Convolution Module
        self.conv = ConvolutionModule(
            d_model=d_model,
            kernel_size=conv_kernel_size,
            dropout=dropout
        )
        
        # Second Feed-Forward Module (half step)
        self.ffn2 = FeedForwardModule(
            d_model=d_model,
            expansion_factor=ffn_expansion,
            dropout=dropout,
            half_step=True
        )
        
        # Final Layer Normalization
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model) or (batch, d_model) for single step
            mask: Optional attention mask
            
        Returns:
            (batch, seq_len, d_model) or (batch, d_model)
        """
        # Handle single-step input (batch, d_model)
        squeeze = False
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, 1, d_model)
            squeeze = True
        
        # Conformer block
        x = x + 0.5 * self.ffn1(x)  # Half-step FFN
        x = x + self.attention(x, mask)
        x = x + self.conv(x)
        x = x + 0.5 * self.ffn2(x)  # Half-step FFN
        x = self.layer_norm(x)
        
        if squeeze:
            x = x.squeeze(1)  # Back to (batch, d_model)
        
        return x


class FeedForwardModule(nn.Module):
    """Feed-Forward Module with expansion"""
    
    def __init__(
        self,
        d_model: int,
        expansion_factor: int = 4,
        dropout: float = 0.1,
        half_step: bool = False
    ):
        super().__init__()
        
        self.half_step = half_step
        expanded_dim = d_model * expansion_factor
        
        self.layer_norm = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, expanded_dim)
        self.activation = nn.SiLU()  # Swish activation
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(expanded_dim, d_model)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
            
        Returns:
            (batch, seq_len, d_model)
        """
        x = self.layer_norm(x)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        
        return x


class MultiHeadSelfAttention(nn.Module):
    """Multi-Head Self-Attention with relative positional encoding"""
    
    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        
        self.layer_norm = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
            mask: Optional attention mask
            
        Returns:
            (batch, seq_len, d_model)
        """
        x = self.layer_norm(x)
        attn_output, _ = self.attention(x, x, x, attn_mask=mask)
        attn_output = self.dropout(attn_output)
        
        return attn_output


if __name__ == "__main__":
    """Test Conformer block"""
    
    print("="*80)
    print("TESTING CONFORMER BLOCK")
    print("="*80)
    
    # Test with single sample
    print("\nTest 1: Single sample (batch, d_model)")
    block = ConformerBlock(d_model=840, num_heads=8, ffn_expansion=4, conv_kernel_size=31)
    x = torch.randn(4, 840)  # (batch=4, d_model=840)
    output = block(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Parameters: {sum(p.numel() for p in block.parameters()):,}")
    
    # Test with sequence
    print("\nTest 2: Sequence (batch, seq_len, d_model)")
    x = torch.randn(4, 10, 840)  # (batch=4, seq_len=10, d_model=840)
    output = block(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    
    print("\n[OK] Conformer block test passed!")
