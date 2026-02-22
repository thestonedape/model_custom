"""
Evaluation Metrics for BELT Word Classification
Computes Top-K accuracy metrics
"""

import torch
from typing import Dict, List


def compute_topk_accuracy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    k_values: List[int] = [1, 5, 10]
) -> Dict[str, float]:
    """
    Compute Top-K accuracy
    
    Args:
        logits: (batch, num_classes) - model predictions
        labels: (batch,) - ground truth labels
        k_values: List of K values to compute (default: [1, 5, 10])
        
    Returns:
        Dictionary with top-k accuracies
    """
    batch_size = labels.size(0)
    
    # Get top-k predictions
    max_k = max(k_values)
    _, top_k_preds = torch.topk(logits, k=max_k, dim=1)  # (batch, max_k)
    
    # Expand labels for comparison
    labels_expanded = labels.unsqueeze(1).expand_as(top_k_preds)  # (batch, max_k)
    
    # Check if correct label is in top-k
    correct = (top_k_preds == labels_expanded)  # (batch, max_k)
    
    # Compute accuracy for each k
    accuracies = {}
    for k in k_values:
        # Check if correct label is in top-k predictions
        correct_at_k = correct[:, :k].any(dim=1).float()  # (batch,)
        accuracy = correct_at_k.sum().item() / batch_size
        accuracies[f'top{k}'] = accuracy
    
    return accuracies


class MetricsTracker:
    """Track metrics over multiple batches"""
    
    def __init__(self, k_values: List[int] = [1, 5, 10]):
        """
        Args:
            k_values: List of K values to track
        """
        self.k_values = k_values
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.total_samples = 0
        self.correct_at_k = {k: 0 for k in self.k_values}
        self.total_loss = 0.0
        self.loss_components = {
            'L_ce': 0.0,
            'L_vq': 0.0,
            'L_cl': 0.0
        }
        self.num_batches = 0
    
    def update(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        loss_dict: Dict[str, float] = None
    ):
        """
        Update metrics with a batch
        
        Args:
            logits: (batch, num_classes)
            labels: (batch,)
            loss_dict: Dictionary with loss components (optional)
        """
        batch_size = labels.size(0)
        self.total_samples += batch_size
        self.num_batches += 1
        
        # Update Top-K accuracies
        max_k = max(self.k_values)
        with torch.no_grad():
            _, top_k_preds = torch.topk(logits, k=max_k, dim=1)
            labels_expanded = labels.unsqueeze(1).expand_as(top_k_preds)
            correct = (top_k_preds == labels_expanded)
            
            for k in self.k_values:
                correct_at_k = correct[:, :k].any(dim=1).float().sum().item()
                self.correct_at_k[k] += correct_at_k
        
        # Update losses
        if loss_dict:
            self.total_loss += loss_dict.get('L_total', 0.0)
            self.loss_components['L_ce'] += loss_dict.get('L_ce', 0.0)
            self.loss_components['L_vq'] += loss_dict.get('L_vq', 0.0)
            self.loss_components['L_cl'] += loss_dict.get('L_cl', 0.0)
    
    def compute(self) -> Dict[str, float]:
        """
        Compute average metrics
        
        Returns:
            Dictionary with all metrics
        """
        if self.total_samples == 0:
            return {}
        
        metrics = {}
        
        # Top-K accuracies
        for k in self.k_values:
            metrics[f'top{k}_acc'] = self.correct_at_k[k] / self.total_samples
        
        # Average losses
        if self.num_batches > 0:
            metrics['loss'] = self.total_loss / self.num_batches
            metrics['L_ce'] = self.loss_components['L_ce'] / self.num_batches
            metrics['L_vq'] = self.loss_components['L_vq'] / self.num_batches
            metrics['L_cl'] = self.loss_components['L_cl'] / self.num_batches
        
        return metrics
    
    def print_summary(self, prefix: str = ""):
        """Print metrics summary"""
        metrics = self.compute()
        
        print(f"\n{prefix} Metrics:")
        print(f"  Top-1 Acc:  {metrics.get('top1_acc', 0.0):.4f} ({metrics.get('top1_acc', 0.0)*100:.2f}%)")
        print(f"  Top-5 Acc:  {metrics.get('top5_acc', 0.0):.4f} ({metrics.get('top5_acc', 0.0)*100:.2f}%)")
        print(f"  Top-10 Acc: {metrics.get('top10_acc', 0.0):.4f} ({metrics.get('top10_acc', 0.0)*100:.2f}%)")
        
        if 'loss' in metrics:
            print(f"\n  Total Loss: {metrics['loss']:.4f}")
            print(f"    L_ce: {metrics['L_ce']:.4f}")
            print(f"    L_vq: {metrics['L_vq']:.4f}")
            print(f"    L_cl: {metrics['L_cl']:.4f}")
        
        print(f"  Samples: {self.total_samples}")


if __name__ == "__main__":
    """Test metrics computation"""
    
    print("="*80)
    print("TESTING METRICS")
    print("="*80)
    
    # Test Top-K accuracy
    print("\nTest 1: Top-K Accuracy")
    logits = torch.randn(32, 500)
    labels = torch.randint(0, 500, (32,))
    
    # Set one prediction to be correct for testing
    logits[0, labels[0]] = 100.0
    
    accuracies = compute_topk_accuracy(logits, labels, k_values=[1, 5, 10])
    print(f"  Batch size: {labels.size(0)}")
    for k, acc in accuracies.items():
        print(f"  {k}: {acc:.4f} ({acc*100:.2f}%)")
    
    # Test MetricsTracker
    print("\nTest 2: MetricsTracker")
    tracker = MetricsTracker(k_values=[1, 5, 10])
    
    # Simulate multiple batches
    for i in range(5):
        logits = torch.randn(32, 500)
        labels = torch.randint(0, 500, (32,))
        
        # Make some predictions correct
        for j in range(5):
            logits[j, labels[j]] = 100.0
        
        loss_dict = {
            'L_total': 2.5 + i * 0.1,
            'L_ce': 2.0,
            'L_vq': 0.3,
            'L_cl': 0.2
        }
        
        tracker.update(logits, labels, loss_dict)
    
    tracker.print_summary(prefix="Training")
    
    # Test reset
    print("\nTest 3: Reset")
    tracker.reset()
    print(f"  Total samples after reset: {tracker.total_samples}")
    print(f"  Num batches after reset: {tracker.num_batches}")
    
    print("\n[OK] Metrics test passed!")


# Alias for backward compatibility
TopKAccuracyTracker = MetricsTracker
