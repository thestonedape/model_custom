"""
QUICK BELT VERIFICATION
Fast pre-training checks - run this right before training!

This is a lightweight version that checks only critical items.
For comprehensive verification, run: python verify_belt_implementation.py
"""

import sys
import torch
import torch.nn.functional as F
from pathlib import Path

# Add to path
sys.path.insert(0, str(Path(__file__).parent))

print("="*80)
print("BELT QUICK VERIFICATION")
print("="*80)
print("Checking critical items before training...\n")

errors = []
warnings = []

# ==============================================================================
# 1. DATA FILES
# ==============================================================================
print("[1/7] Checking data files...")

if Path("data/vocabulary_top500.pkl").exists():
    print("  ✓ Vocabulary file exists")
else:
    errors.append("Vocabulary file missing: run python prepare_data.py")
    print("  ✗ Vocabulary file missing")

if Path("data/splits.pkl").exists():
    print("  ✓ Splits file exists")
else:
    errors.append("Splits file missing: run python prepare_data.py")
    print("  ✗ Splits file missing")

pickle_dir = Path("dataset/ZuCo/task1-SR/pickle")
if pickle_dir.exists() and len(list(pickle_dir.glob("*.pickle"))) > 0:
    print(f"  ✓ Pickle files exist ({len(list(pickle_dir.glob('*.pickle')))} files)")
else:
    errors.append("Pickle files missing: run python prepare_data.py")
    print("  ✗ Pickle files missing")

# ==============================================================================
# 2. VOCABULARY SIZE
# ==============================================================================
print("\n[2/7] Checking vocabulary...")

try:
    from data.vocabulary import Vocabulary
    vocab = Vocabulary(vocab_size=500)
    vocab.load("data/vocabulary_top500.pkl")
    
    if len(vocab.word2idx) == 500:
        print(f"  ✓ Vocabulary size = 500")
    else:
        errors.append(f"Vocabulary size = {len(vocab.word2idx)}, expected 500")
        print(f"  ✗ Vocabulary size = {len(vocab.word2idx)}")
except Exception as e:
    errors.append(f"Vocabulary check failed: {str(e)}")
    print(f"  ✗ Vocabulary check failed")

# ==============================================================================
# 3. MODEL ARCHITECTURE
# ==============================================================================
print("\n[3/7] Checking model architecture...")

try:
    from models.dconformer import DConformer
    from models.vector_quantizer import VectorQuantizer
    from models.classifier import MLPClassifier
    
    # Check D-Conformer
    encoder = DConformer(d_model=840, num_blocks=6, num_heads=8, 
                        ffn_expansion=4, conv_kernel_size=31, dropout=0.1)
    
    if len(encoder.conformer_blocks) == 6:
        print("  ✓ D-Conformer has 6 blocks")
    else:
        errors.append(f"D-Conformer has {len(encoder.conformer_blocks)} blocks, not 6")
        print(f"  ✗ D-Conformer has {len(encoder.conformer_blocks)} blocks")
    
    # Check VQ
    vq = VectorQuantizer(input_dim=840, codebook_size=1024, 
                        codebook_dim=1024, beta=0.3)
    
    if vq.codebook.num_embeddings == 1024 and vq.codebook.embedding_dim == 1024:
        print("  ✓ VQ: K=1024, D=1024")
    else:
        errors.append(f"VQ config wrong: K={vq.codebook.num_embeddings}, D={vq.codebook.embedding_dim}")
        print(f"  ✗ VQ: K={vq.codebook.num_embeddings}, D={vq.codebook.embedding_dim}")
    
    if vq.beta == 0.3:
        print("  ✓ VQ: β=0.3")
    else:
        errors.append(f"VQ beta = {vq.beta}, not 0.3")
        print(f"  ✗ VQ: β={vq.beta}")
    
    # Check Classifier
    classifier = MLPClassifier(input_dim=1024, hidden_dims=[512, 256], 
                              output_dim=500, dropout=0.3)
    print("  ✓ Classifier: 1024→512→256→500")
    
except Exception as e:
    errors.append(f"Model architecture check failed: {str(e)}")
    print(f"  ✗ Model check failed: {str(e)}")

# ==============================================================================
# 4. FORWARD PASS
# ==============================================================================
print("\n[4/7] Testing forward pass...")

try:
    x = torch.randn(4, 840)
    
    # Forward
    h = encoder(x)
    vq_loss, quantized, perplexity, _ = vq(h)
    logits = classifier(quantized)
    
    # Check shapes
    if logits.shape == (4, 500):
        print(f"  ✓ Output shape: {logits.shape}")
    else:
        errors.append(f"Output shape wrong: {logits.shape}")
        print(f"  ✗ Output shape: {logits.shape}")
    
    # Check losses
    labels = torch.randint(0, 500, (4,))
    ce_loss = F.cross_entropy(logits, labels)
    
    if 0 < ce_loss.item() < 10:
        print(f"  ✓ CE loss: {ce_loss.item():.4f}")
    else:
        warnings.append(f"CE loss unusual: {ce_loss.item()}")
        print(f"  ⚠ CE loss: {ce_loss.item():.4f}")
    
    if 0 < vq_loss.item() < 10:
        print(f"  ✓ VQ loss: {vq_loss.item():.4f}")
    else:
        warnings.append(f"VQ loss unusual: {vq_loss.item()}")
        print(f"  ⚠ VQ loss: {vq_loss.item():.4f}")
    
except Exception as e:
    errors.append(f"Forward pass failed: {str(e)}")
    print(f"  ✗ Forward pass failed: {str(e)}")

# ==============================================================================
# 5. BACKWARD PASS
# ==============================================================================
print("\n[5/7] Testing backward pass...")

try:
    total_loss = ce_loss + vq_loss
    total_loss.backward()
    
    # Check gradients
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 
                  for p in list(encoder.parameters()) + 
                           list(vq.parameters()) + 
                           list(classifier.parameters()))
    
    if has_grad:
        print("  ✓ Gradients computed")
    else:
        errors.append("No gradients after backward pass")
        print("  ✗ No gradients")
    
    # Check for NaN
    has_nan = any(p.grad is not None and torch.isnan(p.grad).any() 
                 for p in list(encoder.parameters()) + 
                          list(vq.parameters()) + 
                          list(classifier.parameters()))
    
    if not has_nan:
        print("  ✓ No NaN in gradients")
    else:
        errors.append("NaN in gradients")
        print("  ✗ NaN in gradients")
    
except Exception as e:
    errors.append(f"Backward pass failed: {str(e)}")
    print(f"  ✗ Backward pass failed: {str(e)}")

# ==============================================================================
# 6. CONFIGURATION
# ==============================================================================
print("\n[6/7] Checking configuration...")

try:
    import yaml
    with open("config/belt_config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    checks = {
        'epochs': (config['training']['epochs'], 60),
        'batch_size': (config['training']['batch_size'], 64),
        'learning_rate': (config['training']['learning_rate'], 5e-6),
        'alpha': (config['training']['loss_weights']['alpha'], 0.9),
        'lambda': (config['training']['loss_weights']['lambda'], 1.0),
    }
    
    all_match = True
    for param, (actual, expected) in checks.items():
        if actual == expected:
            print(f"  ✓ {param}: {actual}")
        else:
            warnings.append(f"{param}: {actual} (BELT uses {expected})")
            print(f"  ⚠ {param}: {actual} (BELT: {expected})")
            all_match = False
    
    if all_match:
        print("  ✓ All hyperparameters match BELT paper")
    
except Exception as e:
    warnings.append(f"Config check failed: {str(e)}")
    print(f"  ⚠ Config check failed")

# ==============================================================================
# 7. TRAINING SCRIPTS
# ==============================================================================
print("\n[7/7] Checking training scripts...")

if Path("experiments/model_with_bootstrapping.py").exists():
    print("  ✓ Training script exists")
else:
    errors.append("Training script missing")
    print("  ✗ Training script missing")

# ==============================================================================
# SUMMARY
# ==============================================================================
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

if errors:
    print(f"\n❌ FOUND {len(errors)} ERROR(S):\n")
    for i, error in enumerate(errors, 1):
        print(f"  {i}. {error}")
    print("\n⚠️  PLEASE FIX ERRORS BEFORE TRAINING!")
    print("\n" + "="*80)
    sys.exit(1)
elif warnings:
    print(f"\n⚠️  FOUND {len(warnings)} WARNING(S):\n")
    for i, warning in enumerate(warnings, 1):
        print(f"  {i}. {warning}")
    print("\n✓ No critical errors")
    print("✅ YOU CAN PROCEED WITH TRAINING")
else:
    print("\n✅ ALL CHECKS PASSED!")
    print("\n🎉 READY TO TRAIN!")
    print("\nNext steps:")
    print("  1. python experiments/model_with_bootstrapping.py")
    print("  2. Monitor training for ~60 epochs")
    print("  3. Expected: Top-10 accuracy ~31.04%")

print("\n" + "="*80)
print()
