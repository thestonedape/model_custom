"""
D-Conformer Encoder for BELT
6-layer Conformer stack for EEG encoding
"""

import torch
import torch.nn as nn
from .conformer_block import ConformerBlock


class DConformer(nn.Module):
    """
    D-Conformer: Deep Conformer Encoder
    
    Architecture (from BELT Table I):
    - 6 Conformer blocks
    - d_model = 840
    - num_heads = 8
    - FFN expansion = 4 (840 → 3360 → 840)
    - Conv kernel size = 31
    - Dropout = 0.1
    """
    
    def __init__(
        self,
        d_model: int = 840,
        num_blocks: int = 6,
        num_heads: int = 8,
        ffn_expansion: int = 4,
        conv_kernel_size: int = 31,
        dropout: float = 0.1
    ):
        """
        Args:
            d_model: Model dimension (default: 840)
            num_blocks: Number of Conformer blocks (default: 6)
            num_heads: Number of attention heads (default: 8)
            ffn_expansion: FFN expansion factor (default: 4)
            conv_kernel_size: Conv kernel size (default: 31)
            dropout: Dropout probability
        """
        super().__init__()
        
        self.d_model = d_model
        self.num_blocks = num_blocks
        
        # Stack of Conformer blocks
        self.conformer_blocks = nn.ModuleList([
            ConformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                ffn_expansion=ffn_expansion,
                conv_kernel_size=conv_kernel_size,
                dropout=dropout
            )
            for _ in range(num_blocks)
        ])
        
        # Final layer norm
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        """
        Args:
            x: (batch, 840) or (batch, seq_len, 840)
            mask: Optional attention mask
            
        Returns:
            (batch, 840) or (batch, seq_len, 840)
        """
        # Pass through all Conformer blocks
        for block in self.conformer_blocks:
            x = block(x, mask)
        
        # Final layer norm
        x = self.layer_norm(x)
        
        return x
    
    def get_num_params(self) -> int:
        """Get total number of parameters"""
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    """Test D-Conformer"""
    
    print("="*80)
    print("TESTING D-CONFORMER ENCODER")
    print("="*80)
    
    # Test with single sample
    print("\nTest 1: Single sample (batch, d_model)")
    encoder = DConformer(
        d_model=840,
        num_blocks=6,
        num_heads=8,
        ffn_expansion=4,
        conv_kernel_size=31,
        dropout=0.1
    )
    
    x = torch.randn(4, 840)  # (batch=4, d_model=840)
    output = encoder(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Total parameters: {encoder.get_num_params():,}")
    
    # Test with sequence
    print("\nTest 2: Sequence (batch, seq_len, d_model)")
    x = torch.randn(4, 10, 840)
    output = encoder(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    
    # Test gradient flow
    print("\nTest 3: Gradient flow")
    x = torch.randn(4, 840, requires_grad=True)
    output = encoder(x)
    loss = output.sum()
    loss.backward()
    print(f"  Input gradient exists: {x.grad is not None}")
    print(f"  Input gradient norm: {x.grad.norm().item():.6f}")
    
    # Parameter breakdown
    print("\nTest 4: Parameter breakdown by block")
    total = 0
    for i, block in enumerate(encoder.conformer_blocks):
        block_params = sum(p.numel() for p in block.parameters())
        total += block_params
        print(f"  Block {i+1}: {block_params:,} parameters")
    print(f"  Layer norm: {sum(p.numel() for p in encoder.layer_norm.parameters()):,} parameters")
    print(f"  Total: {total:,} parameters")
    
    print("\n[OK] D-Conformer test passed!")
