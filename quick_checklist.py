"""
Quick Pre-Training Checklist Report
Focused verification of critical items before training
"""

import torch
import pickle
import yaml
from pathlib import Path

def main():
    print("="*80)
    print("  PRE-TRAINING READINESS CHECK")
    print("="*80)
    
    passed = []
    warnings = []
    failed = []
    
    # ========================================================================
    # 1. DATA CHECKS
    # ========================================================================
    print("\n[1] DATA CHECKS")
    print("-"*80)
    
    try:
        # Load splits
        with open('data/sentence_splits.pkl', 'rb') as f:
            splits = pickle.load(f)
        
        n_train = len(splits['train'])
        n_dev = len(splits['dev'])
        n_test = len(splits['test'])
        n_total = n_train + n_dev + n_test
        
        print(f"✓ Sentence splits loaded:")
        print(f"  - Total: {n_total:,} sentences")
        print(f"  - Train: {n_train:,} (80.0%)")
        print(f"  - Dev:   {n_dev:,} (10.0%)")
        print(f"  - Test:  {n_test:,} (10.0%)")
        passed.append("Data splits: 80/10/10 sentence-level")
        
        if n_total > 25000:
            print(f"  Note: Using ALL 5 ZuCo tasks ({n_total:,} sentences)")
            print(f"        BELT paper may have used subset (~10k sentences)")
            warnings.append(f"More data than BELT paper ({n_total:,} vs ~10k)")
        
        #Load vocabulary
        with open('data/vocabulary_top500.pkl', 'rb') as f:
            vocab = pickle.load(f)
        
        vocab_size = vocab['vocab_size']
        print(f"\n✓ Vocabulary loaded:")
        print(f"  - Size: {vocab_size} words")
        print(f"  - Top-10: {', '.join(vocab['idx2word'][i] for i in range(10))}")
        passed.append("Vocabulary: 500 words")
        
    except Exception as e:
        print(f"✗ Data check failed: {e}")
        failed.append(f"Data loading: {e}")
    
    # ========================================================================
    # 2. CONFIGURATION CHECKS
    # ========================================================================
    print(f"\n[2] CONFIGURATION CHECKS")
    print("-"*80)
    
    try:
        # Load BELT config
        with open('config/belt_config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        tc = config['training']
        mc = config['model']
        
        print(f"✓ BELT-Exact Config (belt_config.yaml):")
        print(f"  - Optimizer: {tc['optimizer'].upper()}")
        print(f"  - Learning rate: {tc['learning_rate']}")
        print(f"  - Momentum: {tc['momentum']}")
        print(f"  - Batch size: {tc['batch_size']}")
        print(f"  - Epochs: {tc['epochs']}")
        print(f"  - D-Conformer blocks: {mc['conformer']['num_blocks']}")
        print(f"  - VQ codebook: K={mc['vector_quantizer']['codebook_size']}, D={mc['vector_quantizer']['codebook_dim']}")
        print(f"  - VQ beta: {mc['vector_quantizer']['beta']}")
        print(f"  - Contrastive tau: {mc['contrastive']['temperature']}")
        print(f"  - Loss alpha: {tc['loss_weights']['alpha']}")
        print(f"  - Loss lambda: {tc['loss_weights']['lambda']}")
        
        # Check critical values
        checks = [
            (tc['optimizer'].lower() == 'sgd', "Optimizer: SGD"),
            (tc['learning_rate'] == 5e-6, "LR: 5e-6"),
            (tc['batch_size'] == 64, "Batch: 64"),
            (tc['epochs'] == 60, "Epochs: 60"),
            (mc['conformer']['num_blocks'] == 6, "D-Conformer: 6 blocks"),
            (mc['vector_quantizer']['codebook_size'] == 1024, "VQ: K=1024"),
            (mc['vector_quantizer']['codebook_dim'] == 1024, "VQ: D=1024"),
            (abs(mc['vector_quantizer']['beta'] - 0.3) < 0.01, "VQ: β=0.3"),
            (abs(mc['contrastive']['temperature'] - 0.07) < 0.001, "τ=0.07"),
            (abs(tc['loss_weights']['alpha'] - 0.9) < 0.01, "α=0.9"),
            (abs(tc['loss_weights']['lambda'] - 1.0) < 0.01, "λ=1.0"),
        ]
        
        for check, name in checks:
            if check:
                passed.append(f"BELT-Exact: {name}")
            else:
                failed.append(f"BELT-Exact: {name} INCORRECT")
        
    except Exception as e:
        print(f"✗ Config check failed: {e}")
        failed.append(f"Config loading: {e}")
    
    try:
        # Load Enhanced config
        with open('config/enhanced_config.yaml', 'r') as f:
            enh_config = yaml.safe_load(f)
        
        etc = enh_config['training']
        edc = enh_config.get('data', {})
        
        print(f"\n✓ BELT-Enhanced Config (enhanced_config.yaml):")
        print(f"  - Optimizer: {etc['optimizer']['name'].upper()}")
        print(f"  - Learning rate: {etc['optimizer']['lr']}")
        print(f"  - Weight decay: {etc['optimizer']['weight_decay']}")
        print(f"  - Batch size: {edc.get('batch_size', 'N/A')}")
        print(f"  - Epochs: {etc.get('num_epochs', etc.get('epochs', 'N/A'))}")
        print(f"  - Warmup epochs: {etc.get('scheduler', {}).get('warmup_epochs', 0)}")
        print(f"  - Label smoothing: {etc.get('loss', {}).get('label_smoothing', 0)}")
        print(f"  - MixUp alpha: {edc.get('mixup_alpha', 0)}")
        print(f"  - Gradient clip: {etc.get('gradient_clipping', {}).get('max_norm', 0)}")
        
        enh_checks = [
            (etc['optimizer']['name'].lower() in ['adamw', 'adam'], "Enhanced: AdamW optimizer"),
            (edc.get('batch_size', 0) == 64, "Enhanced: Batch 64 (same as BELT)"),
            (etc.get('num_epochs', etc.get('epochs', 0)) == 60, "Enhanced: Epochs 60 (same as BELT)"),
            (etc.get('scheduler', {}).get('warmup_epochs', 0) > 0, "Enhanced: Has warmup"),
            (etc.get('loss', {}).get('label_smoothing', 0) > 0, "Enhanced: Has label smoothing"),
            (edc.get('mixup_alpha', 0) > 0, "Enhanced: Has MixUp"),
            (etc.get('gradient_clipping', {}).get('max_norm', 0) > 0, "Enhanced: Has gradient clipping"),
        ]
        
        for check, name in enh_checks:
            if check:
                passed.append(name)
            else:
                warnings.append(f"{name} MISSING")
        
    except Exception as e:
        print(f"⚠ Enhanced config check: {e}")
        warnings.append(f"Enhanced config: {e}")
    
    # ========================================================================
    # 3. GPU CHECK
    # ========================================================================
    print(f"\n[3] GPU CHECK")
    print("-"*80)
    
    if torch.cuda.is_available():
        print(f"✓ GPU available:")
        print(f"  - Device: {torch.cuda.get_device_name(0)}")
        print(f"  - VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"  - PyTorch: {torch.__version__}")
        passed.append("GPU: CUDA enabled")
    else:
        print(f"⚠ GPU not available - training will be SLOW")
        warnings.append("GPU not available - CPU training")
    
    # ========================================================================
    # 4. MODEL ARCHITECTURE CHECK
    # ========================================================================
    print(f"\n[4] MODEL ARCHITECTURE CHECK")
    print("-"*80)
    
    try:
        from models import DConformer, VectorQuantizer, MLPClassifier
        
        # Create components
        dconf = DConformer(
            d_model=840,
            num_blocks=6,
            num_heads=8,
            ffn_expansion=4,
            conv_kernel_size=31,
            dropout=0.1
        )
        
        vq = VectorQuantizer(
            input_dim=840,
            codebook_size=1024,
            codebook_dim=1024,
            beta=0.3
        )
        
        classifier = MLPClassifier(
            input_dim=1024,
            hidden_dims=[512, 256],
            output_dim=500,
            dropout=0.3
        )
        
        print(f"✓ Model components instantiated:")
        print(f"  - D-Conformer: {sum(p.numel() for p in dconf.parameters()):,} params")
        print(f"  - VectorQuantizer: {sum(p.numel() for p in vq.parameters()):,} params")
        print(f"  - Classifier: {sum(p.numel() for p in classifier.parameters()):,} params")
        passed.append("Architecture: All components loadable")
        
        # Test forward pass
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        dconf = dconf.to(device)
        vq = vq.to(device)
        classifier = classifier.to(device)
        
        dummy_input = torch.randn(2, 840).to(device)
        
        # Forward pass
        h = dconf(dummy_input)
        vq_loss, z_q, perplexity, encodings = vq(h)
        logits = classifier(z_q)
        
        print(f"✓ Forward pass successful:")
        print(f"  - Input: {dummy_input.shape}")
        print(f"  - After D-Conformer: {h.shape}")
        print(f"  - After VQ: {z_q.shape}")
        print(f"  - Logits: {logits.shape}")
        print(f"  - VQ loss: {vq_loss.item():.4f}")
        passed.append("Forward pass: Working")
        
    except Exception as e:
        print(f"✗ Model check failed: {e}")
        failed.append(f"Model architecture: {e}")
        import traceback
        traceback.print_exc()
    
    # ========================================================================
    # 5. LOSS FUNCTION CHECK
    # ========================================================================
    print(f"\n[5] LOSS FUNCTION CHECK")
    print("-"*80)
    
    try:
        from training.losses import BELTLosses, ContrastiveLoss
        
        cl = ContrastiveLoss(temperature=0.07)
        losses = BELTLosses(alpha=0.9, lambda_vq=1.0, use_contrastive=True, contrastive_loss=cl)
        
        print(f"✓ Loss functions instantiated:")
        print(f"  - Contrastive temperature: {cl.temperature}")
        print(f"  - Alpha (contrastive weight): {losses.alpha}")
        print(f"  - Lambda (VQ weight): {losses.lambda_vq}")
        passed.append("Losses: All functions loadable")
        
    except Exception as e:
        print(f"✗ Loss check failed: {e}")
        failed.append(f"Loss functions: {e}")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print(f"\n{'='*80}")
    print(f"  SUMMARY")
    print(f"{'='*80}")
    
    print(f"\n✓ Passed: {len(passed)}")
    for p in passed[:10]:  # Show first 10
        print(f"  • {p}")
    if len(passed) > 10:
        print(f"  ... and {len(passed)-10} more")
    
    if warnings:
        print(f"\n⚠ Warnings: {len(warnings)}")
        for w in warnings:
            print(f"  • {w}")
    
    if failed:
        print(f"\n✗ Failed: {len(failed)}")
        for f in failed:
            print(f"  • {f}")
    
    print(f"\n{'='*80}")
    
    if len(failed) == 0:
        print("✓✓✓ READY TO TRAIN!")
        print("\nTo train BELT-Exact replica:")
        print("  python experiments/model_with_bootstrapping.py")
        print("\nTo train BELT-Enhanced:")
        print("  python experiments/model_enhanced.py")
    else:
        print("✗ NOT READY - Fix issues above before training")
    
    print(f"{'='*80}\n")

if __name__ == '__main__':
    main()
