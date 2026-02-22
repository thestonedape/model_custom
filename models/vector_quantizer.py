"""
Vector Quantizer for BELT
Implements VQ-VAE style quantization (Equations 1 and 2 from BELT paper)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    """
    Vector Quantizer with learnable codebook
    
    Based on BELT paper:
    - Input: continuous representation h (d_model=840)
    - Project to: z_e(h) of dimension D=1024
    - Quantize: find nearest codebook vector
    - Output: discrete representation b (D=1024)
    - Loss: L_vq = ||sg[z_e(h)] - v||² + β*||z_e(h) - sg[v]||² (Equation 2)
    
    Where:
    - K = codebook size (1024)
    - D = codebook dimension (1024)
    - β = commitment loss weight (0.3)
    - sg[] = stop gradient
    """
    
    def __init__(
        self,
        input_dim: int = 840,
        codebook_size: int = 1024,
        codebook_dim: int = 1024,
        beta: float = 0.3
    ):
        """
        Args:
            input_dim: Input dimension (d_model, default: 840)
            codebook_size: Number of codebook vectors K (default: 1024)
            codebook_dim: Dimension of codebook vectors D (default: 1024)
            beta: Commitment loss weight (default: 0.3)
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.beta = beta
        
        # Projection: 840 → 1024
        self.projection = nn.Linear(input_dim, codebook_dim)
        
        # Codebook: (K, D)
        self.codebook = nn.Embedding(codebook_size, codebook_dim)
        # Initialize codebook with uniform distribution
        self.codebook.weight.data.uniform_(-1.0/codebook_size, 1.0/codebook_size)
        
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with quantization
        
        Args:
            x: (batch, d_model) or (batch, seq_len, d_model)
            
        Returns:
            vq_loss: scalar tensor - VQ loss (codebook + commitment)
            quantized: (batch, codebook_dim) or (batch, seq_len, codebook_dim) - quantized vectors
            perplexity: scalar tensor - codebook usage metric
            encodings: (batch, codebook_size) or (batch, seq_len, codebook_size) - one-hot encodings
        """
        # Handle different input shapes
        original_shape = x.shape
        if x.dim() == 2:
            # (batch, d_model)
            batch_size = x.size(0)
            flat = True
        else:
            # (batch, seq_len, d_model)
            batch_size, seq_len = x.size(0), x.size(1)
            x = x.reshape(-1, x.size(-1))  # (batch*seq_len, d_model)
            flat = False
        
        # Project to codebook dimension: z_e(h)
        z_e = self.projection(x)  # (N, codebook_dim)
        
        # Quantize using Equation 1: find nearest codebook vector
        # Compute distances: ||z_e - v_k||² for all k
        distances = (
            torch.sum(z_e ** 2, dim=1, keepdim=True) +
            torch.sum(self.codebook.weight ** 2, dim=1) -
            2 * torch.matmul(z_e, self.codebook.weight.t())
        )  # (N, K)
        
        # Get nearest codebook indices
        min_indices = torch.argmin(distances, dim=1)  # (N,)
        
        # Get quantized vectors
        quantized = self.codebook(min_indices)  # (N, codebook_dim)
        
        # Compute VQ loss (Equation 2)
        # L_vq = ||sg[z_e(h)] - v||² + β*||z_e(h) - sg[v]||²
        # FIX: Detach z_e (not quantized) for codebook loss so codebook gets gradients
        codebook_loss = F.mse_loss(quantized, z_e.detach())  # Codebook moves toward encoder
        commitment_loss = F.mse_loss(z_e, quantized.detach())  # Encoder commits to codebook
        vq_loss = codebook_loss + self.beta * commitment_loss
        
        # Straight-through estimator: copy gradients from quantized to z_e
        quantized = z_e + (quantized - z_e).detach()
        
        # Calculate perplexity (codebook usage metric)
        # Average probability across codes
        encodings = F.one_hot(min_indices, num_classes=self.codebook_size).float()  # (N, K)
        avg_probs = torch.mean(encodings, dim=0)  # (K,)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # Reshape back to original shape
        if not flat:
            quantized = quantized.reshape(batch_size, seq_len, self.codebook_dim)
            encodings = encodings.reshape(batch_size, seq_len, self.codebook_size)
        
        return vq_loss, quantized, perplexity, encodings
    
    def get_codebook_usage(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get codebook usage statistics (for monitoring)
        
        Args:
            x: (batch, d_model) or (batch, seq_len, d_model)
            
        Returns:
            usage: (codebook_size,) - count of how many times each code is used
        """
        # Project and quantize
        if x.dim() == 3:
            x = x.reshape(-1, x.size(-1))
        
        z_e = self.projection(x)
        
        distances = (
            torch.sum(z_e ** 2, dim=1, keepdim=True) +
            torch.sum(self.codebook.weight ** 2, dim=1) -
            2 * torch.matmul(z_e, self.codebook.weight.t())
        )
        
        min_indices = torch.argmin(distances, dim=1)
        
        # Count usage
        usage = torch.zeros(self.codebook_size, dtype=torch.long, device=x.device)
        usage.scatter_add_(0, min_indices, torch.ones_like(min_indices))
        
        return usage


if __name__ == "__main__":
    """Test Vector Quantizer"""
    
    print("="*80)
    print("TESTING VECTOR QUANTIZER")
    print("="*80)
    
    # Test with single sample
    print("\nTest 1: Single sample (batch, d_model)")
    vq = VectorQuantizer(input_dim=840, codebook_size=1024, codebook_dim=1024, beta=0.3)
    x = torch.randn(4, 840)  # (batch=4, d_model=840)
    quantized, vq_loss = vq(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Quantized shape: {quantized.shape}")
    print(f"  VQ loss: {vq_loss.item():.6f}")
    print(f"  Parameters: {sum(p.numel() for p in vq.parameters()):,}")
    
    # Test with sequence
    print("\nTest 2: Sequence (batch, seq_len, d_model)")
    x = torch.randn(4, 10, 840)
    quantized, vq_loss = vq(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Quantized shape: {quantized.shape}")
    print(f"  VQ loss: {vq_loss.item():.6f}")
    
    # Test codebook usage
    print("\nTest 3: Codebook usage statistics")
    x = torch.randn(100, 840)
    usage = vq.get_codebook_usage(x)
    print(f"  Total codes used: {(usage > 0).sum().item()} / {vq.codebook_size}")
    print(f"  Max usage: {usage.max().item()}")
    print(f"  Mean usage: {usage.float().mean().item():.2f}")
    
    # Test gradient flow
    print("\nTest 4: Gradient flow (straight-through estimator)")
    x = torch.randn(4, 840, requires_grad=True)
    quantized, vq_loss = vq(x)
    loss = quantized.sum() + vq_loss
    loss.backward()
    print(f"  Input gradient exists: {x.grad is not None}")
    print(f"  Input gradient norm: {x.grad.norm().item():.6f}")
    
    print("\n[OK] Vector Quantizer test passed!")
