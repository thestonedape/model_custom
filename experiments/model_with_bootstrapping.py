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
import torch.nn as nn
import yaml
import json
from pathlib import Path

# Import components
from data import Vocabulary
from data.sentence_dataset import create_sentence_dataloaders, load_sentence_splits
from models import DConformer, IdentityQuantizer, VectorQuantizer, MLPClassifier
from training import ContrastiveLoss, BELTLosses, BELTTrainer
from training.losses import build_class_balanced_weights


def format_alpha_slug(alpha: float) -> str:
    """Create a filesystem-safe alpha identifier."""
    return f"{alpha:g}".replace('-', 'm').replace('.', 'p')


def resolve_run_settings(config: dict, args) -> tuple[float, bool, str]:
    """Resolve loss settings and result directory without overwriting unrelated runs."""
    default_alpha = float(config['training']['loss_weights']['alpha'])
    alpha = float(args.alpha) if args.alpha is not None else default_alpha
    use_contrastive = (not args.no_contrastive) and alpha > 0.0
    
    if not use_contrastive:
        save_dir = config['experiments']['ablation']['save_dir']
    elif args.alpha is None or abs(alpha - default_alpha) < 1e-12:
        save_dir = config['experiments']['full_belt']['save_dir']
    else:
        save_dir = str(Path("results") / f"alpha_{format_alpha_slug(alpha)}_results")
    
    return alpha, use_contrastive, save_dir


def validate_tasks(config: dict, vocab: Vocabulary, splits_path: str, config_path: Path) -> bool:
    """Fail fast if vocabulary or sentence splits were generated for the wrong task set."""
    expected_tasks = list(config['data']['tasks'])
    splits = load_sentence_splits(splits_path)
    split_tasks = list(splits.get('metadata', {}).get('tasks', []))
    
    if not split_tasks:
        print("\nError: sentence split metadata is missing the task list.")
        print(f"Please regenerate sentence splits with: python prepare_sentence_splits.py --config {config_path}")
        return False
    
    if split_tasks != expected_tasks:
        print("\nError: sentence split tasks do not match the baseline config.")
        print(f"  Config tasks: {expected_tasks}")
        print(f"  Split tasks:  {split_tasks}")
        print(f"Please regenerate sentence splits with: python prepare_sentence_splits.py --config {config_path}")
        return False
    
    vocab_tasks = getattr(vocab, 'metadata', {}).get('tasks')
    if vocab_tasks and list(vocab_tasks) != expected_tasks:
        print("\nError: vocabulary tasks do not match the baseline config.")
        print(f"  Config tasks:     {expected_tasks}")
        print(f"  Vocabulary tasks: {list(vocab_tasks)}")
        print(f"Please regenerate the vocabulary with: python prepare_data.py --config {config_path}")
        return False
    
    if not vocab_tasks:
        print("\nWarning: vocabulary metadata is missing, so task consistency cannot be verified.")
        print("Regenerating it with prepare_data.py will make future comparisons safer.")
    
    return True


def build_encoder(raw_input_dim: int, config: dict):
    """Build the encoder and add a learned projection when input features are widened."""
    conformer_cfg = config['model']['conformer']
    d_model = int(conformer_cfg['d_model'])

    backbone = DConformer(
        d_model=d_model,
        num_blocks=conformer_cfg['num_blocks'],
        num_heads=conformer_cfg['num_heads'],
        ffn_expansion=conformer_cfg['ffn_expansion'],
        conv_kernel_size=conformer_cfg['conv_kernel_size'],
        dropout=conformer_cfg['dropout']
    )

    if raw_input_dim == d_model:
        return backbone

    print(f"Adding input projection: {raw_input_dim} -> {d_model}")
    return nn.Sequential(
        nn.LayerNorm(raw_input_dim),
        nn.Linear(raw_input_dim, d_model),
        nn.GELU(),
        backbone
    )


def use_vector_quantizer(config: dict) -> bool:
    """Whether this run should use the quantizer in the forward path."""
    return bool(config.get('model', {}).get('vector_quantizer', {}).get('enabled', True))


def build_vector_quantizer(config: dict):
    """Build the active quantizer module and return its output dimension."""
    vq_cfg = config['model']['vector_quantizer']
    if not use_vector_quantizer(config):
        input_dim = int(config['model']['conformer']['d_model'])
        print("Vector Quantizer: DISABLED (true no-VQ architecture)")
        return IdentityQuantizer(input_dim=input_dim), input_dim

    quantizer = VectorQuantizer(
        input_dim=config['model']['conformer']['d_model'],
        codebook_size=vq_cfg['codebook_size'],
        codebook_dim=vq_cfg['codebook_dim'],
        beta=vq_cfg['beta']
    )
    return quantizer, int(vq_cfg['codebook_dim'])


def build_ce_weights(train_dataset, config: dict, num_classes: int):
    """Build optional class-balanced CE weights from the train split."""
    class_balance_cfg = config.get('training', {}).get('class_balanced_loss', {})
    if not class_balance_cfg.get('enabled', False):
        return None, "standard", None

    if not hasattr(train_dataset, 'get_label_distribution'):
        print("Warning: train dataset does not expose label counts; falling back to standard CE.")
        return None, "standard", None

    beta = float(class_balance_cfg.get('beta', 0.9999))
    label_counts = train_dataset.get_label_distribution()
    class_weights = build_class_balanced_weights(
        label_counts=label_counts,
        num_classes=num_classes,
        beta=beta
    )

    non_zero_weights = class_weights[class_weights > 0]
    print("\nClass-balanced CE enabled")
    print(f"  Beta: {beta}")
    print(f"  Classes with weights: {(class_weights > 0).sum().item()} / {num_classes}")
    print(f"  Weight range: {non_zero_weights.min().item():.4f} -> {non_zero_weights.max().item():.4f}")

    return class_weights, "class_balanced", beta


def main():
    """Run Model 2 (Full BELT with Bootstrapping)"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train BELT model')
    parser.add_argument('--config', type=str, default='config/belt_config.yaml',
                       help='Path to the config file')
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
    config_path = Path(args.config)
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    alpha, use_contrastive = None, None
    alpha, use_contrastive, save_dir = resolve_run_settings(config, args)
    config['training']['loss_weights']['alpha'] = alpha
    if 'contrastive_schedule' in config.get('training', {}):
        config['training']['contrastive_schedule']['target_alpha'] = alpha
    
    print(f"\nConfiguration loaded from: {config_path}")
    print(f"Save directory: {save_dir}")
    
    # Load vocabulary
    print("\n" + "-"*80)
    print("LOADING VOCABULARY")
    print("-"*80)
    vocab = Vocabulary(vocab_size=500)
    vocab_path = config['data'].get('vocab_path', "data/vocabulary_top500.pkl")
    
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
    
    splits_path = config['data'].get('splits_path', "data/sentence_splits.pkl")
    if not Path(splits_path).exists():
        print(f"\nError: Sentence splits not found at {splits_path}")
        print(f"Please run: python prepare_sentence_splits.py --config {config_path}")
        return
    
    if not validate_tasks(config, vocab, splits_path, config_path):
        return
    
    train_loader, dev_loader, test_loader = create_sentence_dataloaders(
        vocabulary=vocab,
        batch_size=config['training']['batch_size'],
        num_workers=0,  # Set to 0 for Windows to avoid CPU overload
        splits_path=splits_path,
        eeg_type="GD",
        pin_memory=bool(config['hardware'].get('pin_memory', False) and torch.cuda.is_available()),
        eeg_types=config['data'].get('eeg_types'),
        include_n_fixations=bool(config['data'].get('include_n_fixations', False)),
        use_weighted_sampler=bool(config['data'].get('use_weighted_sampler', False))
    )
    
    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_loader.dataset)} samples")
    print(f"  Dev: {len(dev_loader.dataset)} samples")
    print(f"  Test: {len(test_loader.dataset)} samples")
    feature_dim = train_loader.dataset[0][0].numel()
    print(f"  Input feature dim: {feature_dim}")
    print(f"  Active EEG types: {config['data'].get('eeg_types', ['GD'])}")
    print(f"  Include nFixations: {bool(config['data'].get('include_n_fixations', False))}")
    print(f"  Use vector quantizer: {use_vector_quantizer(config)}")
        # VERIFY DATA/LABEL ALIGNMENT
    print("\n" + "-"*80)
    print("VERIFYING DATA/LABEL ALIGNMENT (10 samples)")
    print("-"*80)
    first_batch = next(iter(train_loader))
    eeg_batch, label_batch, word_batch = first_batch
    for i in range(min(10, len(label_batch))):
        label_id = label_batch[i].item()
        word_from_vocab = vocab.get_word_from_index(label_id)
        word_from_dataset = word_batch[i]
        match = "✓" if word_from_vocab == word_from_dataset else "✗ MISMATCH!"
        print(f"  [{i}] Label={label_id:3d} | Vocab='{word_from_vocab}' | Dataset='{word_from_dataset}' {match}")
        # Build models
    print("\n" + "-"*80)
    print("BUILDING MODELS")
    print("-"*80)
    
    # D-Conformer Encoder
    encoder = build_encoder(feature_dim, config)
    print(f"D-Conformer: {sum(p.numel() for p in encoder.parameters()):,} parameters")
    
    # Quantizer / identity passthrough
    vector_quantizer, model_feature_dim = build_vector_quantizer(config)
    print(f"Vector Quantizer: {sum(p.numel() for p in vector_quantizer.parameters()):,} parameters")
    
    # Classifier
    classifier = MLPClassifier(
        input_dim=model_feature_dim,
        hidden_dims=config['model']['classifier']['hidden_dims'],
        output_dim=config['model']['classifier']['output_dim'],
        dropout=config['model']['classifier']['dropout']
    )
    print(f"Classifier: {sum(p.numel() for p in classifier.parameters()):,} parameters")
    print(f"  Classifier input dim: {model_feature_dim}")
    
    # Setup contrastive loss
    print("\n" + "-"*80)
    print("SETTING UP CONTRASTIVE LOSS")
    print("-"*80)
    
    contrastive_loss = None
    
    if use_contrastive:
        print("Loading BART model for word embeddings...")
        
        contrastive_loss = ContrastiveLoss(
            eeg_dim=model_feature_dim,
            word_dim=config['model']['contrastive']['word_proj_dim'],
            bart_model_name=config['model']['contrastive']['bart_model'],
            temperature=config['model']['contrastive']['temperature'],
            freeze_bart=config['model']['contrastive']['freeze_bart']
        )
        
        print(f"Contrastive loss initialized")
        print(f"  EEG projection: {model_feature_dim}")
        print(f"  Word projection: {config['model']['contrastive']['word_proj_dim']}")
        print(f"  Temperature: {config['model']['contrastive']['temperature']}")
        print(f"  BART frozen: {config['model']['contrastive']['freeze_bart']}")
    else:
        print("Contrastive loss DISABLED (ablation mode)")
    
    total_params = (
        sum(p.numel() for p in encoder.parameters()) +
        sum(p.numel() for p in vector_quantizer.parameters()) +
        sum(p.numel() for p in classifier.parameters())
    )
    if use_contrastive and contrastive_loss is not None:
        total_params += (
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

    class_weights, ce_weighting, class_balance_beta = build_ce_weights(
        train_loader.dataset,
        config,
        num_classes=config['model']['classifier']['output_dim']
    )
    
    belt_losses = BELTLosses(
        alpha=alpha,
        lambda_vq=config['training']['loss_weights']['lambda'],
        use_contrastive=use_contrastive,
        contrastive_loss=contrastive_loss,
        class_weights=class_weights,
        ce_weighting=ce_weighting,
        class_balance_beta=class_balance_beta
    )
    
    print(f"Loss weights:")
    print(f"  λ (VQ): {config['training']['loss_weights']['lambda']}")
    print(f"  CE weighting: {ce_weighting}")
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
        'save_every': config['training']['save_every'],
        'run_name': 'full_belt' if use_contrastive else 'ablation',
        'use_vector_quantizer': use_vector_quantizer(config),
        'contrastive_schedule': {
            'enabled': bool(use_contrastive and config['training'].get('contrastive_schedule', {}).get('enabled', False)),
            'start_alpha': float(config['training'].get('contrastive_schedule', {}).get('start_alpha', 0.0)),
            'target_alpha': float(config['training'].get('contrastive_schedule', {}).get('target_alpha', alpha)),
            'warmup_epochs': int(config['training'].get('contrastive_schedule', {}).get('warmup_epochs', 0)),
            'ramp_epochs': int(config['training'].get('contrastive_schedule', {}).get('ramp_epochs', 0))
        }
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
    
    results_path = Path(save_dir) / "final_results.json"
    if results_path.exists():
        with open(results_path, 'r', encoding='utf-8') as f:
            final_results = json.load(f)
        test_metrics = final_results.get('test_metrics', {})
        print("FINAL ACCURACY SUMMARY")
        print(f"  Best Dev Top-10: {final_results.get('best_dev_acc', 0.0)*100:.2f}%")
        print(f"  Test Top-1:      {test_metrics.get('top1_acc', 0.0)*100:.2f}%")
        print(f"  Test Top-5:      {test_metrics.get('top5_acc', 0.0)*100:.2f}%")
        print(f"  Test Top-10:     {test_metrics.get('top10_acc', 0.0)*100:.2f}%")
    else:
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
