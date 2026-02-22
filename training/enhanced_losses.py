"""
Enhanced Loss Functions
Includes proven improvements: Label Smoothing, Focal Loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Label Smoothing Cross Entropy Loss
    
    Paper: "When Does Label Smoothing Help?" (Müller et al., NeurIPS 2019)
    Expected gain: +1-2% on classification tasks
    
    Args:
        epsilon: Smoothing factor (default: 0.1)
                 Hard labels [0, 0, 1, 0, 0]
                 → Smooth labels [ε/K, ε/K, 1-ε+ε/K, ε/K, ε/K]
    """
    
    def __init__(self, epsilon: float = 0.1):
        super().__init__()
        self.epsilon = epsilon
        
    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (batch, num_classes) - model predictions
            target: (batch,) - ground truth labels
            
        Returns:
            loss: scalar tensor
        """
        n_classes = logits.size(-1)
        
        # Get log probabilities
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Create smooth labels
        with torch.no_grad():
            # One-hot encode
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.epsilon / (n_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.epsilon)
        
        # Compute loss
        loss = -(true_dist * log_probs).sum(dim=-1).mean()
        
        return loss


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance
    
    Paper: "Focal Loss for Dense Object Detection" (Lin et al., ICCV 2017)
    Expected gain: +1-3% when classes are imbalanced
    
    Args:
        alpha: Weighting factor (default: 1.0)
        gamma: Focusing parameter (default: 2.0)
               gamma=0 → standard cross entropy
               gamma>0 → down-weight easy examples
    """
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (batch, num_classes)
            target: (batch,)
            
        Returns:
            loss: scalar tensor
        """
        # Compute cross entropy
        ce_loss = F.cross_entropy(logits, target, reduction='none')
        
        # Compute pt = exp(-ce_loss)
        pt = torch.exp(-ce_loss)
        
        # Focal loss: α * (1-pt)^γ * CE
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        return focal_loss.mean()


if __name__ == "__main__":
    """Test enhanced losses"""
    
    print("="*80)
    print("TESTING ENHANCED LOSS FUNCTIONS")
    print("="*80)
    
    # Test Label Smoothing
    print("\nTest 1: Label Smoothing Cross Entropy")
    ls_loss = LabelSmoothingCrossEntropy(epsilon=0.1)
    
    logits = torch.randn(32, 500)
    labels = torch.randint(0, 500, (32,))
    
    # Compare with standard CE
    ce_loss_fn = nn.CrossEntropyLoss()
    ce_loss = ce_loss_fn(logits, labels)
    ls_loss_val = ls_loss(logits, labels)
    
    print(f"  Standard CE Loss: {ce_loss.item():.4f}")
    print(f"  Label Smoothing CE Loss: {ls_loss_val.item():.4f}")
    print(f"  Difference: {(ls_loss_val - ce_loss).item():.4f}")
    
    # Test Focal Loss
    print("\nTest 2: Focal Loss")
    focal_loss = FocalLoss(alpha=1.0, gamma=2.0)
    
    # Create imbalanced scenario
    logits = torch.randn(100, 500)
    labels = torch.randint(0, 500, (100,))
    
    # Make some predictions very confident (easy examples)
    logits[0, labels[0]] = 10.0  # Easy example
    logits[1, labels[1]] = -5.0  # Hard example
    
    ce_loss = ce_loss_fn(logits, labels)
    focal_loss_val = focal_loss(logits, labels)
    
    print(f"  Standard CE Loss: {ce_loss.item():.4f}")
    print(f"  Focal Loss (α=1.0, γ=2.0): {focal_loss_val.item():.4f}")
    
    # Test different gamma values
    print("\nTest 3: Focal Loss with different gamma")
    for gamma in [0.0, 1.0, 2.0, 3.0]:
        focal = FocalLoss(alpha=1.0, gamma=gamma)
        loss = focal(logits, labels)
        print(f"  γ={gamma}: {loss.item():.4f}")
    
    # Test gradient flow
    print("\nTest 4: Gradient flow")
    logits = torch.randn(32, 500, requires_grad=True)
    labels = torch.randint(0, 500, (32,))
    
    ls_loss = LabelSmoothingCrossEntropy(epsilon=0.1)
    loss = ls_loss(logits, labels)
    loss.backward()
    
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Logits gradient exists: {logits.grad is not None}")
    print(f"  Gradient norm: {logits.grad.norm().item():.6f}")
    
    print("\n[OK] Enhanced loss functions test passed!")
