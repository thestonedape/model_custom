"""
BELT-Enhanced: Training with Proven Enhancements

This script implements BELT with multiple proven enhancements from literature:
1. Label Smoothing (Müller et al., NeurIPS 2019)
2. AdamW Optimizer (Loshchilov & Hutter, ICLR 2019)
3. Warmup + Cosine LR Schedule (Goyal et al., 2017)
4. Gradient Clipping (Pascanu et al., 2013)
5. MixUp Augmentation (Zhang et al., ICLR 2018)
6. Stochastic Depth (Huang et al., ECCV 2016)
7. Multi-Sample Dropout (Gal & Ghahramani, ICML 2016)

Expected Performance:
- Baseline BELT: 31.04% top-10 accuracy
- BELT-Enhanced: 37-39% top-10 accuracy
- BELT-Ensemble (3 models): 39-41% top-10 accuracy

Usage:
    python model_custom/experiments/model_enhanced.py \
        --config model_custom/config/enhanced_config.yaml \
        --mode train
"""

import os
import sys
import yaml
import argparse
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BartModel, BartTokenizer

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Import BELT components
from data.vocabulary import Vocabulary
from data.sentence_dataset import BELTSentenceDataset, load_sentence_splits
from models.dconformer import DConformer
from models.vector_quantizer import VectorQuantizer
from models.classifier import MLPClassifier
from training.losses import ContrastiveLoss
from training.metrics import TopKAccuracyTracker

# Import enhancements
from training.enhanced_losses import LabelSmoothingCrossEntropy, FocalLoss
from training.schedulers import WarmupCosineSchedule
from training.augmentation import MixUp, mixup_criterion
from training.regularization import LinearScheduleDropPath, MultiSampleDropoutClassifier


def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class BELTEnhancedModel(nn.Module):
    """
    Enhanced BELT Model with all improvements
    
    Architecture:
        EEG → D-Conformer (with DropPath) → Vector Quantizer → MLP Classifier (with Multi-Sample Dropout)
    
    Enhancements:
        - Stochastic Depth in Conformer blocks
        - Multi-Sample Dropout in classifier
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        model_cfg = config['model']
        
        # D-Conformer Encoder (with optional DropPath)
        self.encoder = DConformer(
            d_model=model_cfg['encoder']['d_model'],
            num_blocks=model_cfg['encoder']['num_layers'],
            num_heads=model_cfg['encoder']['num_heads'],
            ffn_expansion=model_cfg['encoder']['ff_expansion'],
            conv_kernel_size=model_cfg['encoder']['conv_kernel_size'],
            dropout=model_cfg['encoder']['dropout']
        )
        
        # Add DropPath to encoder if enabled
        if model_cfg['encoder'].get('use_drop_path', False):
            self._add_drop_path_to_encoder(
                drop_path_rate=model_cfg['encoder']['drop_path_rate']
            )
        
        # Vector Quantizer
        self.vq = VectorQuantizer(
            input_dim=model_cfg['encoder']['d_model'],
            codebook_size=model_cfg['vq']['codebook_size'],
            codebook_dim=model_cfg['vq']['codebook_dim'],
            beta=model_cfg['vq']['beta']
        )
        
        # Base Classifier
        input_dim = model_cfg['vq']['codebook_dim']
        hidden_dims = model_cfg['classifier']['hidden_dims']
        num_classes = model_cfg['classifier']['num_classes']
        
        layers = []
        dims = [input_dim] + hidden_dims
        
        for i in range(len(dims) - 1):
            layers.extend([
                nn.Linear(dims[i], dims[i + 1]),
                nn.ReLU(),
                nn.Dropout(0.5)
            ])
        
        layers.append(nn.Linear(dims[-1], num_classes))
        base_classifier = nn.Sequential(*layers)
        
        # Wrap with Multi-Sample Dropout if enabled
        if model_cfg['classifier'].get('use_multi_sample_dropout', False):
            self.classifier = MultiSampleDropoutClassifier(
                base_classifier=base_classifier,
                dropout_p=0.5,
                num_samples=model_cfg['classifier']['multi_sample_num']
            )
        else:
            self.classifier = base_classifier
    
    def _add_drop_path_to_encoder(self, drop_path_rate):
        """Add DropPath to encoder's residual connections"""
        num_layers = len(self.encoder.layers)
        
        for i, layer in enumerate(self.encoder.layers):
            # Add DropPath after each Conformer block
            layer.drop_path = LinearScheduleDropPath(
                drop_prob_max=drop_path_rate,
                layer_idx=i,
                num_layers=num_layers
            )
    
    def forward(self, x, return_vq_loss=False):
        """
        Forward pass
        
        Args:
            x: EEG input [batch_size, seq_len, input_dim]
            return_vq_loss: Whether to return VQ loss
        
        Returns:
            logits: Classification logits [batch_size, num_classes]
            vq_loss: Vector quantizer loss (if return_vq_loss=True)
            quantized: Quantized embeddings (if return_vq_loss=True)
        """
        # Encode with D-Conformer
        encoded = self.encoder(x)  # [batch_size, seq_len, d_model]
        
        # Average pool over sequence
        pooled = encoded.mean(dim=1)  # [batch_size, d_model]
        
        # Vector quantization
        vq_loss, quantized, _, _ = self.vq(pooled)
        
        # Classification
        logits = self.classifier(quantized)
        
        if return_vq_loss:
            return logits, vq_loss, quantized
        return logits


def load_bart_embeddings(config, vocabulary):
    """Load BART embeddings for contrastive learning"""
    print("\n[BART] Loading BART embeddings for bootstrapping...")
    
    embedding_cfg = config['data']['word_embeddings']
    model_name = embedding_cfg['model_name']
    
    # Load BART
    tokenizer = BartTokenizer.from_pretrained(model_name)
    bart_model = BartModel.from_pretrained(model_name)
    
    # Extract embeddings
    word_embeddings = []
    for word in vocabulary.idx_to_word:
        # Tokenize and get embedding
        tokens = tokenizer.encode(word, add_special_tokens=False)
        if len(tokens) > 0:
            token_id = tokens[0]
            embedding = bart_model.shared.weight[token_id].detach()
        else:
            # Fallback for unknown words
            embedding = torch.zeros(embedding_cfg['embedding_dim'])
        
        word_embeddings.append(embedding)
    
    # Stack into tensor [vocab_size, embedding_dim]
    word_embeddings = torch.stack(word_embeddings)
    
    print(f"[BART] Loaded embeddings: {word_embeddings.shape}")
    return word_embeddings


def train_epoch(model, dataloader, optimizer, scheduler, criterion_dict, device, config, epoch):
    """Train for one epoch"""
    model.train()
    
    total_loss = 0.0
    num_batches = 0
    
    # Get loss functions
    criterion_ce = criterion_dict['ce']
    criterion_cl = criterion_dict['cl']
    criterion_vq = criterion_dict['vq']
    
    # Get loss weights
    alpha = config['training']['loss']['alpha']
    lambda_vq = config['training']['loss']['lambda_vq']
    
    # MixUp augmentation
    use_mixup = config['data'].get('use_mixup', False)
    if use_mixup:
        mixup = MixUp(
            alpha=config['data']['mixup_alpha'],
            prob=config['data']['mixup_prob']
        )
    
    # Progress bar
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(pbar):
        eeg_data = batch['eeg'].to(device)
        labels = batch['label'].to(device)
        word_embeddings = batch['word_embedding'].to(device)
        
        # Apply MixUp if enabled
        if use_mixup:
            eeg_data, labels_a, labels_b, lam = mixup(eeg_data, labels)
        
        # Forward pass
        logits, vq_loss, quantized = model(eeg_data, return_vq_loss=True)
        
        # Classification loss (with MixUp if applied)
        if use_mixup:
            loss_ce = mixup_criterion(criterion_ce, logits, labels_a, labels_b, lam)
        else:
            loss_ce = criterion_ce(logits, labels)
        
        # Contrastive loss (with bootstrapping)
        loss_cl = criterion_cl(quantized, word_embeddings, labels)
        
        # Total loss
        loss = loss_ce + alpha * loss_cl + lambda_vq * vq_loss
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if config['training']['gradient_clipping']['enabled']:
            max_norm = config['training']['gradient_clipping']['max_norm']
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        
        optimizer.step()
        
        # Accumulate loss
        total_loss += loss.item()
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'ce': f"{loss_ce.item():.4f}",
            'cl': f"{loss_cl.item():.4f}",
            'vq': f"{vq_loss.item():.4f}",
            'lr': f"{optimizer.param_groups[0]['lr']:.2e}"
        })
    
    # Step scheduler
    scheduler.step()
    
    avg_loss = total_loss / num_batches
    return avg_loss


def evaluate(model, dataloader, criterion_dict, device, config):
    """Evaluate model"""
    model.eval()
    
    total_loss = 0.0
    num_batches = 0
    
    # Metrics tracker
    metrics = TopKAccuracyTracker(k_values=[1, 3, 5, 10])
    
    # Get loss functions
    criterion_ce = criterion_dict['ce']
    criterion_cl = criterion_dict['cl']
    criterion_vq = criterion_dict['vq']
    
    alpha = config['training']['loss']['alpha']
    lambda_vq = config['training']['loss']['lambda_vq']
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            eeg_data = batch['eeg'].to(device)
            labels = batch['label'].to(device)
            word_embeddings = batch['word_embedding'].to(device)
            
            # Forward pass
            logits, vq_loss, quantized = model(eeg_data, return_vq_loss=True)
            
            # Losses
            loss_ce = criterion_ce(logits, labels)
            loss_cl = criterion_cl(quantized, word_embeddings, labels)
            loss = loss_ce + alpha * loss_cl + lambda_vq * vq_loss
            
            total_loss += loss.item()
            num_batches += 1
            
            # Update metrics
            metrics.update(logits, labels)
    
    avg_loss = total_loss / num_batches
    results = metrics.compute()
    
    return avg_loss, results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/enhanced_config.yaml')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval'])
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set seed
    set_seed(config['seed'])
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() and config['device']['use_cuda'] else 'cpu')
    print(f"\n[Device] Using: {device}")
    
    # Load vocabulary
    print("\n[Data] Loading vocabulary...")
    vocab = Vocabulary(vocab_size=500)
    vocab.load("data/vocabulary_top500.pkl")
    print(f"[Data] Vocabulary size: {len(vocab.word2idx)}")
    
    # Load sentence-level splits (80/10/10)
    print("\n[Data] Loading sentence-level splits (80/10/10)...")
    print("[Data] Using sentence-level splits for fair comparison with BELT paper")
    
    splits_path = "data/sentence_splits.pkl"
    if not Path(splits_path).exists():
        print(f"\nError: Sentence splits not found at {splits_path}")
        print("Please run: python prepare_sentence_splits.py")
        return
    
    splits = load_sentence_splits(splits_path)
    
    # Create datasets from sentence-level splits
    print("\n[Data] Creating datasets from sentence-level splits...")
    train_dataset = BELTSentenceDataset(
        sentence_list=splits['train'],
        vocabulary=vocab,
        split='train',
        eeg_type='GD'
    )
    
    val_dataset = BELTSentenceDataset(
        sentence_list=splits['dev'],
        vocabulary=vocab,
        split='dev',
        eeg_type='GD'
    )
    
    test_dataset = BELTSentenceDataset(
        sentence_list=splits['test'],
        vocabulary=vocab,
        split='test',
        eeg_type='GD'
    )
    
    print(f"[Data] Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    print(f"[Data] Split ratios: {splits['metadata']['train_ratio']:.1%} / {splits['metadata']['dev_ratio']:.1%} / {splits['metadata']['test_ratio']:.1%}")
    
    # Load BART embeddings (if contrastive loss is used)
    if config.get('use_contrastive_loss', False):
        print("\n[BART] Loading BART embeddings for contrastive loss...")
        bart_embeddings = load_bart_embeddings(config, vocab)
        bart_embeddings = bart_embeddings.to(device)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['evaluation']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory']
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['evaluation']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory']
    )
    
    # Create model
    print("\n[Model] Building BELT-Enhanced model...")
    model = BELTEnhancedModel(config).to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] Trainable parameters: {num_params:,}")
    
    # Optimizer (AdamW)
    print("\n[Training] Setting up optimizer and scheduler...")
    optimizer_cfg = config['training']['optimizer']
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=optimizer_cfg['lr'],
        weight_decay=optimizer_cfg['weight_decay'],
        betas=tuple(optimizer_cfg['betas']),
        eps=optimizer_cfg['eps']
    )
    
    # Scheduler (Warmup + Cosine)
    scheduler_cfg = config['training']['scheduler']
    scheduler = WarmupCosineSchedule(
        optimizer=optimizer,
        warmup_epochs=scheduler_cfg['warmup_epochs'],
        total_epochs=config['training']['num_epochs'],
        min_lr=scheduler_cfg['min_lr']
    )
    
    # Loss functions
    loss_cfg = config['training']['loss']
    
    # Classification loss (with label smoothing)
    if loss_cfg.get('use_label_smoothing', False):
        criterion_ce = LabelSmoothingCrossEntropy(
            num_classes=config['model']['classifier']['num_classes'],
            smoothing=loss_cfg['label_smoothing']
        )
        print(f"[Loss] Using Label Smoothing (ε={loss_cfg['label_smoothing']})")
    elif loss_cfg.get('use_focal_loss', False):
        criterion_ce = FocalLoss(
            alpha=loss_cfg['focal_alpha'],
            gamma=loss_cfg['focal_gamma']
        )
        print(f"[Loss] Using Focal Loss (α={loss_cfg['focal_alpha']}, γ={loss_cfg['focal_gamma']})")
    else:
        criterion_ce = nn.CrossEntropyLoss()
        print("[Loss] Using standard Cross Entropy")
    
    # Contrastive loss
    criterion_cl = ContrastiveLoss(temperature=loss_cfg['temperature'])
    
    # VQ loss
    criterion_vq = VectorQuantizerLoss()
    
    criterion_dict = {
        'ce': criterion_ce,
        'cl': criterion_cl,
        'vq': criterion_vq
    }
    
    # Print configuration summary
    print("\n" + "="*80)
    print("BELT-ENHANCED CONFIGURATION SUMMARY")
    print("="*80)
    print(f"Model: {config['model']['name']}")
    print(f"Optimizer: {optimizer_cfg['name'].upper()}")
    print(f"Learning Rate: {optimizer_cfg['lr']}")
    print(f"Scheduler: {scheduler_cfg['name']}")
    print(f"Warmup Epochs: {scheduler_cfg['warmup_epochs']}")
    print(f"Gradient Clipping: {config['training']['gradient_clipping']['enabled']}")
    print(f"MixUp: {config['data'].get('use_mixup', False)}")
    print(f"DropPath: {config['model']['encoder'].get('use_drop_path', False)}")
    print(f"Multi-Sample Dropout: {config['model']['classifier'].get('use_multi_sample_dropout', False)}")
    print("="*80)
    
    # Create save directories
    os.makedirs(config['training']['save_dir'], exist_ok=True)
    os.makedirs(config['training']['log_dir'], exist_ok=True)
    
    # Training loop
    if args.mode == 'train':
        print("\n[Training] Starting training...")
        
        best_val_acc = 0.0
        
        for epoch in range(1, config['training']['num_epochs'] + 1):
            print(f"\n{'='*80}")
            print(f"EPOCH {epoch}/{config['training']['num_epochs']}")
            print(f"{'='*80}")
            
            # Train
            train_loss = train_epoch(
                model, train_loader, optimizer, scheduler,
                criterion_dict, device, config, epoch
            )
            
            # Validate
            val_loss, val_results = evaluate(
                model, val_loader, criterion_dict, device, config
            )
            
            # Print results
            print(f"\nEpoch {epoch} Results:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Val Top-1:  {val_results['top1']:.2f}%")
            print(f"  Val Top-3:  {val_results['top3']:.2f}%")
            print(f"  Val Top-5:  {val_results['top5']:.2f}%")
            print(f"  Val Top-10: {val_results['top10']:.2f}%")
            
            # Save checkpoint
            if epoch % config['training']['save_every'] == 0:
                checkpoint_path = os.path.join(
                    config['training']['save_dir'],
                    f'checkpoint_epoch_{epoch}.pt'
                )
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_results': val_results
                }, checkpoint_path)
                print(f"\n[Save] Checkpoint saved to: {checkpoint_path}")
            
            # Save best model
            if config['training']['save_best'] and val_results['top10'] > best_val_acc:
                best_val_acc = val_results['top10']
                best_path = os.path.join(
                    config['training']['save_dir'],
                    'best_model.pt'
                )
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'val_results': val_results
                }, best_path)
                print(f"[Save] Best model saved (Top-10: {best_val_acc:.2f}%)")
        
        # Final test evaluation
        print("\n" + "="*80)
        print("FINAL TEST EVALUATION")
        print("="*80)
        
        # Load best model
        best_checkpoint = torch.load(best_path)
        model.load_state_dict(best_checkpoint['model_state_dict'])
        
        test_loss, test_results = evaluate(
            model, test_loader, criterion_dict, device, config
        )
        
        print(f"\nTest Results:")
        print(f"  Test Loss: {test_loss:.4f}")
        print(f"  Test Top-1:  {test_results['top1']:.2f}%")
        print(f"  Test Top-3:  {test_results['top3']:.2f}%")
        print(f"  Test Top-5:  {test_results['top5']:.2f}%")
        print(f"  Test Top-10: {test_results['top10']:.2f}%")
        
        print("\n[Training] Training complete!")


if __name__ == "__main__":
    main()
