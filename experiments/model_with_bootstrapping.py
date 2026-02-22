"""
Model 2: Full BELT - With Bootstrapping
L = L_ce + α*L_cl^w + λ*L_vq (ALL losses)

Expected Top-10 Accuracy: ~31.04%
"""

import sys
import os
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import yaml
from pathlib import Path

# Import components
from data import Vocabulary
from data.sentence_dataset import create_sentence_dataloaders
from models import DConformer, VectorQuantizer, MLPClassifier
from training import ContrastiveLoss, BELTLosses, BELTTrainer


def main():
    """Run Model 2 (Full BELT with Bootstrapping)"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train BELT model')
    parser.add_argument('--no-contrastive', action='store_true',
                       help='Disable contrastive loss (test if CE improves)')
    parser.add_argument('--alpha', type=float, default=None,
                       help='Override contrastive loss weight (e.g., 0.05 for weak contrastive)')
    args = parser.parse_args()
    
    print("="*80)
    print("MODEL 2: FULL BELT (WITH BOOTSTRAPPING)")
    print("="*80)
    if args.no_contrastive:
        print("Training with: L = L_ce + lambda*L_vq (NO CONTRASTIVE)")
        print("Expected: CE should decrease if contrastive was blocking")
    elif args.alpha is not None:
        print(f"Training with: L = L_ce + {args.alpha}*L_cl^w + lambda*L_vq")
        print(f"Testing with weak contrastive (alpha={args.alpha})")
    else:
        print("Training with: L = L_ce + alpha*L_cl^w + lambda*L_vq")
        print("Expected Top-10 Accuracy: ~31.04%")
    print("="*80)
    
    # Load configuration
    config_path = Path("config/belt_config.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Use full BELT settings
    config['experiments']['full_belt']['use_contrastive'] = True
    save_dir = config['experiments']['full_belt']['save_dir']
    
    print(f"\nConfiguration loaded from: {config_path}")
    print(f"Save directory: {save_dir}")
    
    # Load vocabulary
    print("\n" + "-"*80)
    print("LOADING VOCABULARY")
    print("-"*80)
    vocab = Vocabulary(vocab_size=500)
    vocab_path = "data/vocabulary_top500.pkl"
    
    if Path(vocab_path).exists():
        vocab.load(vocab_path)
    else:
        print(f"Vocabulary not found at {vocab_path}")
        print("Please run: python data/vocabulary.py")
        return
    
    # Create dataloaders with sentence-level 80/10/10 splits
    print("\n" + "-"*80)
    print("CREATING DATALOADERS (SENTENCE-LEVEL 80/10/10 SPLITS)")
    print("-"*80)
    print("Using sentence-level splits for fair comparison with BELT paper")
    
    splits_path = "data/sentence_splits.pkl"
    if not Path(splits_path).exists():
        print(f"\nError: Sentence splits not found at {splits_path}")
        print("Please run: python prepare_sentence_splits.py")
        return
    
    train_loader, dev_loader, test_loader = create_sentence_dataloaders(
        vocabulary=vocab,
        batch_size=config['training']['batch_size'],
        num_workers=0,  # Set to 0 for Windows to avoid CPU overload
        splits_path=splits_path,
        eeg_type="GD"
    )
    
    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_loader.dataset)} samples")
    print(f"  Dev: {len(dev_loader.dataset)} samples")
    print(f"  Test: {len(test_loader.dataset)} samples")
        # VERIFY DATA/LABEL ALIGNMENT
    print("\n" + "-"*80)
    print("VERIFYING DATA/LABEL ALIGNMENT (10 samples)")
    print("-"*80)
    first_batch = next(iter(train_loader))
    eeg_batch, label_batch, word_batch = first_batch
    for i in range(min(10, len(label_batch))):
        label_id = label_batch[i].item()
        word_from_vocab = vocab.id_to_word[label_id]
        word_from_dataset = word_batch[i]
        match = "✓" if word_from_vocab == word_from_dataset else "✗ MISMATCH!"
        print(f"  [{i}] Label={label_id:3d} | Vocab='{word_from_vocab}' | Dataset='{word_from_dataset}' {match}")
        # Build models
    print("\n" + "-"*80)
    print("BUILDING MODELS")
    print("-"*80)
    
    # D-Conformer Encoder
    encoder = DConformer(
        d_model=config['model']['conformer']['d_model'],
        num_blocks=config['model']['conformer']['num_blocks'],
        num_heads=config['model']['conformer']['num_heads'],
        ffn_expansion=config['model']['conformer']['ffn_expansion'],
        conv_kernel_size=config['model']['conformer']['conv_kernel_size'],
        dropout=config['model']['conformer']['dropout']
    )
    print(f"D-Conformer: {sum(p.numel() for p in encoder.parameters()):,} parameters")
    
    # Vector Quantizer
    vector_quantizer = VectorQuantizer(
        input_dim=config['model']['vector_quantizer']['input_dim'],
        codebook_size=config['model']['vector_quantizer']['codebook_size'],
        codebook_dim=config['model']['vector_quantizer']['codebook_dim'],
        beta=config['model']['vector_quantizer']['beta']
    )
    print(f"Vector Quantizer: {sum(p.numel() for p in vector_quantizer.parameters()):,} parameters")
    
    # Classifier
    classifier = MLPClassifier(
        input_dim=config['model']['classifier']['input_dim'],
        hidden_dims=config['model']['classifier']['hidden_dims'],
        output_dim=config['model']['classifier']['output_dim'],
        dropout=config['model']['classifier']['dropout']
    )
    print(f"Classifier: {sum(p.numel() for p in classifier.parameters()):,} parameters")
    
    # Setup contrastive loss
    print("\n" + "-"*80)
    print("SETTING UP CONTRASTIVE LOSS")
    print("-"*80)
    
    contrastive_loss = None
    use_contrastive = not args.no_contrastive
    
    if use_contrastive:
        print("Loading BART model for word embeddings...")
        
        contrastive_loss = ContrastiveLoss(
            eeg_dim=config['model']['contrastive']['eeg_proj_dim'],
            word_dim=config['model']['contrastive']['word_proj_dim'],
            bart_model_name=config['model']['contrastive']['bart_model'],
            temperature=config['model']['contrastive']['temperature'],
            freeze_bart=config['model']['contrastive']['freeze_bart']
        )
        
        print(f"Contrastive loss initialized")
        print(f"  EEG projection: {config['model']['contrastive']['eeg_proj_dim']}")
        print(f"  Word projection: {config['model']['contrastive']['word_proj_dim']}")
        print(f"  Temperature: {config['model']['contrastive']['temperature']}")
        print(f"  BART frozen: {config['model']['contrastive']['freeze_bart']}")
    else:
        print("Contrastive loss DISABLED (testing if it blocks CE learning)")
    
    # Get alpha from args or config
    alpha = args.alpha if args.alpha is not None else config['training']['loss_weights']['alpha']
    
    total_params = (
        sum(p.numel() for p in encoder.parameters()) +
        sum(p.numel() for p in vector_quantizer.parameters()) +
        sum(p.numel() for p in classifier.parameters()) +
        sum(p.numel() for p in contrastive_loss.eeg_projection.parameters()) +
        sum(p.numel() for p in contrastive_loss.word_projection.parameters())
    )
    print(f"\nTotal trainable parameters: {total_params:,}")
    
    # Setup combined losses
    print("\n" + "-"*80)
    print("SETTING UP COMBINED LOSSES")
    print("-"*80)
    if use_contrastive:
        print(f"Using: L = L_ce + α*L_cl^w + λ*L_vq")
    else:
        print(f"Using: L = L_ce + λ*L_vq (no contrastive)")
    
    belt_losses = BELTLosses(
        alpha=alpha,
        lambda_vq=config['training']['loss_weights']['lambda'],
        use_contrastive=use_contrastive,
        contrastive_loss=contrastive_loss
    )
    
    print(f"Loss weights:")
    print(f"  λ (VQ): {config['training']['loss_weights']['lambda']}")
    if use_contrastive:
        print(f"  α (Contrastive): {alpha}")
    else:
        print(f"  α (Contrastive): 0.0 (disabled)")
    
    # Create trainer
    print("\n" + "-"*80)
    print("CREATING TRAINER")
    print("-"*80)
    
    trainer_config = {
        'device': config['hardware']['device'],
        'learning_rate': config['training']['learning_rate'],
        'momentum': config['training']['momentum'],
        'weight_decay': config['training']['weight_decay'],
        'epochs': config['training']['epochs'],
        'grad_clip': config['training']['grad_clip'],
        'log_interval': config['logging']['log_interval'],
        'save_best': config['training']['save_best'],
        'save_every': config['training']['save_every']
    }
    
    trainer = BELTTrainer(
        encoder=encoder,
        vector_quantizer=vector_quantizer,
        classifier=classifier,
        belt_losses=belt_losses,
        train_loader=train_loader,
        dev_loader=dev_loader,
        test_loader=test_loader,
        config=trainer_config,
        save_dir=save_dir
    )
    
    # Train
    print("\n" + "="*80)
    print("STARTING TRAINING")
    print("="*80)
    trainer.train(num_epochs=config['training']['epochs'])
    
    print("\n" + "="*80)
    print("MODEL 2 TRAINING COMPLETE")
    print("="*80)
    print(f"Results saved to: {save_dir}")
    print(f"Expected Top-10 Accuracy: ~31.04%")
    print(f"Actual Top-10 Accuracy: {trainer.best_dev_acc:.4f} ({trainer.best_dev_acc*100:.2f}%)")
    
    # Compare with expected
    expected_acc = config['experiments']['full_belt']['expected_top10']
    improvement = (trainer.best_dev_acc - expected_acc) * 100
    if improvement > 0:
        print(f"Improvement over expected: +{improvement:.2f}%")
    else:
        print(f"Difference from expected: {improvement:.2f}%")


if __name__ == "__main__":
    main()
