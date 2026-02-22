"""
MLP Classifier for BELT Word Classification
Maps quantized representations to word labels
"""

import torch
import torch.nn as nn


class MLPClassifier(nn.Module):
    """
    MLP Classifier head
    
    Architecture:
    - Input: 1024 (from VQ codebook)
    - Hidden: 512 → 256
    - Output: 500 (vocabulary size)
    - Activation: ReLU
    - Dropout: 0.3
    """
    
    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dims: list = [512, 256],
        output_dim: int = 500,
        dropout: float = 0.3
    ):
        """
        Args:
            input_dim: Input dimension (from VQ, default: 1024)
            hidden_dims: List of hidden layer dimensions (default: [512, 256])
            output_dim: Output dimension (vocab size, default: 500)
            dropout: Dropout probability
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Build MLP layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Output layer (no activation, will use with CrossEntropyLoss)
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.classifier = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, input_dim) or (batch, seq_len, input_dim)
            
        Returns:
            logits: (batch, output_dim) or (batch, seq_len, output_dim)
        """
        return self.classifier(x)
    
    def get_num_params(self) -> int:
        """Get total number of parameters"""
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    """Test MLP Classifier"""
    
    print("="*80)
    print("TESTING MLP CLASSIFIER")
    print("="*80)
    
    # Test with batch input
    print("\nTest 1: Batch input (batch, input_dim)")
    classifier = MLPClassifier(
        input_dim=1024,
        hidden_dims=[512, 256],
        output_dim=500,
        dropout=0.3
    )
    
    x = torch.randn(4, 1024)  # (batch=4, input_dim=1024)
    logits = classifier(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {logits.shape}")
    print(f"  Total parameters: {classifier.get_num_params():,}")
    
    # Test with sequence input
    print("\nTest 2: Sequence input (batch, seq_len, input_dim)")
    x = torch.randn(4, 10, 1024)
    logits = classifier(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {logits.shape}")
    
    # Test with CrossEntropyLoss
    print("\nTest 3: CrossEntropyLoss compatibility")
    x = torch.randn(32, 1024)
    logits = classifier(x)
    labels = torch.randint(0, 500, (32,))
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(logits, labels)
    print(f"  Batch size: {x.size(0)}")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Labels shape: {labels.shape}")
    print(f"  Loss: {loss.item():.4f}")
    
    # Test gradient flow
    print("\nTest 4: Gradient flow")
    x = torch.randn(4, 1024, requires_grad=True)
    logits = classifier(x)
    loss = logits.sum()
    loss.backward()
    print(f"  Input gradient exists: {x.grad is not None}")
    print(f"  Input gradient norm: {x.grad.norm().item():.6f}")
    
    # Test predictions
    print("\nTest 5: Predictions (top-k)")
    x = torch.randn(1, 1024)
    logits = classifier(x)
    probs = torch.softmax(logits, dim=-1)
    top5_probs, top5_indices = torch.topk(probs, k=5, dim=-1)
    print(f"  Top-5 predictions:")
    for i in range(5):
        print(f"    {i+1}. Word {top5_indices[0, i].item()}: {top5_probs[0, i].item():.4f}")
    
    print("\n[OK] MLP Classifier test passed!")
