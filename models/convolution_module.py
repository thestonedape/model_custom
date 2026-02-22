"""
Convolution Module for Conformer
Implements depthwise separable convolution with gating mechanism
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvolutionModule(nn.Module):
    """
    Convolution Module from Conformer
    
    Structure:
    1. Layer Normalization
    2. Pointwise Conv (expansion)
    3. GLU activation
    4. Depthwise Conv
    5. Batch Normalization
    6. Swish activation
    7. Pointwise Conv (projection)
    8. Dropout
    """
    
    def __init__(
        self,
        d_model: int = 840,
        kernel_size: int = 31,
        expansion_factor: int = 2,
        dropout: float = 0.1
    ):
        """
        Args:
            d_model: Model dimension (default: 840)
            kernel_size: Depthwise convolution kernel size (default: 31)
            expansion_factor: Expansion for pointwise conv (default: 2)
            dropout: Dropout probability
        """
        super().__init__()
        
        self.d_model = d_model
        self.kernel_size = kernel_size
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Pointwise convolution 1 (expansion with GLU)
        # Double the channels for GLU
        self.pointwise_conv1 = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model * expansion_factor * 2,  # *2 for GLU
            kernel_size=1,
            stride=1,
            padding=0
        )
        
        # GLU (Gated Linear Unit) - will split channels and apply gating
        # No explicit layer needed, done in forward pass
        
        # Depthwise convolution
        padding = (kernel_size - 1) // 2
        self.depthwise_conv = nn.Conv1d(
            in_channels=d_model * expansion_factor,
            out_channels=d_model * expansion_factor,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            groups=d_model * expansion_factor  # Depthwise
        )
        
        # Batch normalization
        self.batch_norm = nn.BatchNorm1d(d_model * expansion_factor)
        
        # Swish activation
        self.activation = nn.SiLU()
        
        # Pointwise convolution 2 (projection)
        self.pointwise_conv2 = nn.Conv1d(
            in_channels=d_model * expansion_factor,
            out_channels=d_model,
            kernel_size=1,
            stride=1,
            padding=0
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
            
        Returns:
            (batch, seq_len, d_model)
        """
        # Layer norm
        x = self.layer_norm(x)
        
        # Transpose for conv1d: (batch, d_model, seq_len)
        x = x.transpose(1, 2)
        
        # Pointwise conv 1 + GLU
        x = self.pointwise_conv1(x)  # (batch, d_model*expansion*2, seq_len)
        x = F.glu(x, dim=1)  # (batch, d_model*expansion, seq_len)
        
        # Depthwise conv
        x = self.depthwise_conv(x)
        
        # Batch norm
        x = self.batch_norm(x)
        
        # Activation
        x = self.activation(x)
        
        # Pointwise conv 2 (projection)
        x = self.pointwise_conv2(x)  # (batch, d_model, seq_len)
        
        # Dropout
        x = self.dropout(x)
        
        # Transpose back: (batch, seq_len, d_model)
        x = x.transpose(1, 2)
        
        return x


if __name__ == "__main__":
    """Test Convolution Module"""
    
    print("="*80)
    print("TESTING CONVOLUTION MODULE")
    print("="*80)
    
    # Test with sequence
    print("\nTest 1: Sequence input (batch, seq_len, d_model)")
    conv_module = ConvolutionModule(d_model=840, kernel_size=31)
    x = torch.randn(4, 10, 840)  # (batch=4, seq_len=10, d_model=840)
    output = conv_module(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Parameters: {sum(p.numel() for p in conv_module.parameters()):,}")
    
    # Test with different kernel sizes
    print("\nTest 2: Different kernel sizes")
    for kernel_size in [3, 7, 15, 31]:
        conv_module = ConvolutionModule(d_model=840, kernel_size=kernel_size)
        x = torch.randn(2, 5, 840)
        output = conv_module(x)
        print(f"  Kernel size {kernel_size}: input {x.shape} -> output {output.shape}")
    
    print("\n[OK] Convolution module test passed!")
