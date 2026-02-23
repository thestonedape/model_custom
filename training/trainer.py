"""
BELT Trainer
Main training loop for EEG-to-word classification
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import time
from typing import Dict, Optional
import json

from .metrics import MetricsTracker


class BELTTrainer:
    """
    Trainer for BELT word classification
    
    Handles:
    - Training loop with joint optimization (all losses together)
    - Validation
    - Checkpointing
    - Logging
    """
    
    def __init__(
        self,
        encoder,
        vector_quantizer,
        classifier,
        belt_losses,
        train_loader: DataLoader,
        dev_loader: DataLoader,
        test_loader: DataLoader,
        config: Dict,
        save_dir: str = "model_custom/results"
    ):
        """
        Args:
            encoder: D-Conformer encoder
            vector_quantizer: Vector Quantizer
            classifier: MLP Classifier
            belt_losses: BELTLosses object
            train_loader: Training data loader
            dev_loader: Validation data loader
            test_loader: Test data loader
            config: Configuration dictionary
            save_dir: Directory to save checkpoints and logs
        """
        self.encoder = encoder
        self.vector_quantizer = vector_quantizer
        self.classifier = classifier
        self.belt_losses = belt_losses
        
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.test_loader = test_loader
        
        self.config = config
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Device
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        print(f"Using device: {self.device}")
        
        # Move models to device
        self.encoder.to(self.device)
        self.vector_quantizer.to(self.device)
        self.classifier.to(self.device)
        
        # Move contrastive loss components to device if they exist
        if hasattr(self.belt_losses, 'contrastive_loss') and self.belt_losses.contrastive_loss is not None:
            self.belt_losses.contrastive_loss.to(self.device)
        
        # Setup optimizer
        self.setup_optimizer()
        
        # Setup scheduler
        self.setup_scheduler()
        
        # Metrics trackers
        self.train_metrics = MetricsTracker(k_values=[1, 5, 10])
        self.dev_metrics = MetricsTracker(k_values=[1, 5, 10])
        
        # Training state
        self.current_epoch = 0
        self.best_dev_acc = 0.0
        self.training_history = {
            'train': [],
            'dev': []
        }
        
    def setup_optimizer(self):
        """Setup optimizer (SGD with momentum)"""
        config = self.config
        
        # Collect all parameters
        params = (
            list(self.encoder.parameters()) +
            list(self.vector_quantizer.parameters()) +
            list(self.classifier.parameters())
        )
        
        # Add contrastive loss projection parameters if used
        if self.belt_losses.use_contrastive and self.belt_losses.contrastive_loss:
            params += list(self.belt_losses.contrastive_loss.eeg_projection.parameters())
            params += list(self.belt_losses.contrastive_loss.word_projection.parameters())
        
        self.optimizer = optim.SGD(
            params,
            lr=config.get('learning_rate', 5e-6),
            momentum=config.get('momentum', 0.9),
            weight_decay=config.get('weight_decay', 1e-4)
        )
        
        print(f"Optimizer: SGD")
        print(f"  Learning rate: {config.get('learning_rate', 5e-6)}")
        print(f"  Momentum: {config.get('momentum', 0.9)}")
        print(f"  Weight decay: {config.get('weight_decay', 1e-4)}")
    
    def setup_scheduler(self):
        """Setup learning rate scheduler (CosineAnnealing)"""
        config = self.config
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.get('epochs', 60),
            eta_min=0
        )
        
        print(f"Scheduler: CosineAnnealingLR")
        print(f"  T_max: {config.get('epochs', 60)}")
    
    def train_epoch(self, epoch: int):
        """Train for one epoch"""
        self.encoder.train()
        self.vector_quantizer.train()
        self.classifier.train()
        
        self.train_metrics.reset()
        
        start_time = time.time()
        
        for batch_idx, (eeg, labels, words) in enumerate(self.train_loader):
            # Move to device
            eeg = eeg.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            # Encoder
            h = self.encoder(eeg)  # (batch, 840)
            
            # Vector Quantizer
            vq_loss, b, perplexity, encodings = self.vector_quantizer(h)  # Returns 4 values
            
            # Classifier
            logits = self.classifier(b)  # (batch, 1024) -> (batch, 500)
            
            # Compute total loss
            total_loss, loss_dict = self.belt_losses.compute_total_loss(
                logits=logits,
                labels=labels,
                vq_loss=vq_loss,
                eeg_features=b if self.belt_losses.use_contrastive else None,
                words=words if self.belt_losses.use_contrastive else None
            )
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping
            if self.config.get('grad_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.optimizer.param_groups[0]['params'],
                    self.config['grad_clip']
                )
            
            self.optimizer.step()
            
            # Update metrics
            self.train_metrics.update(logits.detach(), labels, loss_dict)
            
            # Log progress
            if batch_idx % self.config.get('log_interval', 10) == 0:
                elapsed = time.time() - start_time
                print(f"  Batch {batch_idx}/{len(self.train_loader)} | "
                      f"Loss: {loss_dict['L_total']:.4f} | "
                      f"Time: {elapsed:.1f}s")
            
            # Early stopping for quick tests
            max_batches = self.config.get('max_train_batches', None)
            if max_batches and batch_idx >= max_batches:
                print(f"\n  Early stop: reached max_train_batches={max_batches}")
                break
        
        # Print epoch summary
        self.train_metrics.print_summary(prefix=f"Epoch {epoch} Train")
        
        return self.train_metrics.compute()
    
    @torch.no_grad()
    def evaluate(self, data_loader: DataLoader, split_name: str = "Dev"):
        """Evaluate on a dataset"""
        self.encoder.eval()
        self.vector_quantizer.eval()
        self.classifier.eval()
        
        metrics = MetricsTracker(k_values=[1, 5, 10])
        
        for eeg, labels, words in data_loader:
            # Move to device
            eeg = eeg.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            h = self.encoder(eeg)
            vq_loss, b, perplexity, encodings = self.vector_quantizer(h)
            logits = self.classifier(b)
            
            # Compute loss
            _, loss_dict = self.belt_losses.compute_total_loss(
                logits=logits,
                labels=labels,
                vq_loss=vq_loss,
                eeg_features=b if self.belt_losses.use_contrastive else None,
                words=words if self.belt_losses.use_contrastive else None
            )
            
            # Update metrics
            metrics.update(logits, labels, loss_dict)
        
        # Print summary
        metrics.print_summary(prefix=split_name)
        
        return metrics.compute()
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'encoder_state_dict': self.encoder.state_dict(),
            'vector_quantizer_state_dict': self.vector_quantizer.state_dict(),
            'classifier_state_dict': self.classifier.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_dev_acc': self.best_dev_acc,
            'config': self.config,
            'training_history': self.training_history
        }
        
        # Save regular checkpoint
        checkpoint_path = self.save_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")
        
        # Save best model
        if is_best:
            best_path = self.save_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"Saved best model: {best_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.vector_quantizer.load_state_dict(checkpoint['vector_quantizer_state_dict'])
        self.classifier.load_state_dict(checkpoint['classifier_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_dev_acc = checkpoint['best_dev_acc']
        self.training_history = checkpoint.get('training_history', {'train': [], 'dev': []})
        
        print(f"Loaded checkpoint from epoch {self.current_epoch}")
        print(f"Best dev accuracy: {self.best_dev_acc:.4f}")
    
    def train(self, num_epochs: int = 60):
        """
        Main training loop
        
        Args:
            num_epochs: Number of epochs to train
        """
        print("\n" + "="*80)
        print("STARTING TRAINING")
        print("="*80)
        print(f"Epochs: {num_epochs}")
        print(f"Train batches: {len(self.train_loader)}")
        print(f"Dev batches: {len(self.dev_loader)}")
        print(f"Save directory: {self.save_dir}")
        
        # Print model parameters
        total_params = (
            sum(p.numel() for p in self.encoder.parameters()) +
            sum(p.numel() for p in self.vector_quantizer.parameters()) +
            sum(p.numel() for p in self.classifier.parameters())
        )
        print(f"Total parameters: {total_params:,}")
        
        for epoch in range(self.current_epoch, num_epochs):
            print(f"\n{'='*80}")
            print(f"EPOCH {epoch+1}/{num_epochs}")
            print(f"{'='*80}")
            print(f"Learning rate: {self.optimizer.param_groups[0]['lr']:.2e}")
            
            # Train
            train_metrics = self.train_epoch(epoch + 1)
            self.training_history['train'].append(train_metrics)
            
            # Evaluate on dev set
            print(f"\nEvaluating on dev set...")
            dev_metrics = self.evaluate(self.dev_loader, split_name="Dev")
            self.training_history['dev'].append(dev_metrics)
            
            # Check if best model
            dev_acc = dev_metrics.get('top10_acc', 0.0)
            is_best = dev_acc > self.best_dev_acc
            if is_best:
                self.best_dev_acc = dev_acc
                print(f"\n*** New best model! Top-10 Acc: {dev_acc:.4f} ({dev_acc*100:.2f}%) ***")
            
            # Save checkpoint
            if self.config.get('save_best', True) and is_best:
                self.save_checkpoint(epoch + 1, is_best=True)
            
            if (epoch + 1) % self.config.get('save_every', 5) == 0:
                self.save_checkpoint(epoch + 1, is_best=False)
            
            # Step scheduler
            self.scheduler.step()
            
            # Save training history
            history_path = self.save_dir / "training_history.json"
            with open(history_path, 'w') as f:
                json.dump(self.training_history, f, indent=2)
            
            self.current_epoch = epoch + 1
        
        # Final evaluation on test set
        print("\n" + "="*80)
        print("FINAL EVALUATION ON TEST SET")
        print("="*80)
        test_metrics = self.evaluate(self.test_loader, split_name="Test")
        
        # Save final results
        results = {
            'best_dev_acc': self.best_dev_acc,
            'test_metrics': test_metrics,
            'final_epoch': self.current_epoch
        }
        
        results_path = self.save_dir / "final_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nTraining complete!")
        print(f"Best dev Top-10 accuracy: {self.best_dev_acc:.4f} ({self.best_dev_acc*100:.2f}%)")
        print(f"Test Top-10 accuracy: {test_metrics.get('top10_acc', 0.0):.4f} ({test_metrics.get('top10_acc', 0.0)*100:.2f}%)")
        print(f"Results saved to: {results_path}")
