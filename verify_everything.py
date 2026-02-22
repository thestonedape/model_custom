"""
Comprehensive Pre-Training Verification Script
Checks EVERYTHING before training starts

Run this before training to catch errors early!
"""

import sys
import os
from pathlib import Path

# Ensure we're in model_custom directory
if not Path("data").exists() or not Path("models").exists():
    print("[ERROR] Please run this script from the model_custom directory!")
    print("  cd model_custom")
    print("  python verify_everything.py")
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).parent))

print("="*80)
print("BELT PRE-TRAINING VERIFICATION")
print("="*80)

errors = []
warnings = []

# ==============================================================================
# STEP 1: VERIFY DATASET FILES
# ==============================================================================
print("\n" + "="*80)
print("STEP 1: VERIFYING DATASET STRUCTURE")
print("="*80)

required_paths = [
    "dataset/ZuCo/task1-SR/Matlab_files",
    "dataset/ZuCo/task2-NR/Matlab_files",
    "dataset/ZuCo/task3-TSR/Matlab_files",
]

for path in required_paths:
    if Path(path).exists():
        # Count .mat files
        mat_files = list(Path(path).glob("*.mat"))
        print(f"✓ {path}: {len(mat_files)} .mat files")
    else:
        errors.append(f"Missing: {path}")
        print(f"✗ {path}: NOT FOUND")

# ==============================================================================
# STEP 2: VERIFY PYTHON MODULES
# ==============================================================================
print("\n" + "="*80)
print("STEP 2: VERIFYING PYTHON MODULES")
print("="*80)

modules_to_test = [
    ("data.vocabulary", "Vocabulary"),
    ("data.dataset", "BELTWordDataset"),
    ("data.splits", "create_splits"),
    ("models.dconformer", "DConformer"),
    ("models.vector_quantizer", "VectorQuantizer"),
    ("models.classifier", "MLPClassifier"),
    ("training.losses", "ContrastiveLoss"),
    ("training.enhanced_losses", "LabelSmoothingCrossEntropy"),
    ("training.schedulers", "WarmupCosineSchedule"),
    ("training.augmentation", "MixUp"),
    ("training.regularization", "DropPath"),
]

for module_name, class_name in modules_to_test:
    try:
        module = __import__(module_name, fromlist=[class_name])
        cls = getattr(module, class_name)
        print(f"✓ {module_name}.{class_name}")
    except Exception as e:
        errors.append(f"Import failed: {module_name}.{class_name} - {str(e)}")
        print(f"✗ {module_name}.{class_name}: {str(e)}")

# ==============================================================================
# STEP 3: VERIFY EXTERNAL DEPENDENCIES
# ==============================================================================
print("\n" + "="*80)
print("STEP 3: VERIFYING EXTERNAL DEPENDENCIES")
print("="*80)

dependencies = [
    ("torch", "PyTorch"),
    ("numpy", "NumPy"),
    ("scipy", "SciPy"),
    ("yaml", "PyYAML"),
    ("tqdm", "tqdm"),
    ("transformers", "Transformers"),
    ("pandas", "Pandas"),
]

for module_name, display_name in dependencies:
    try:
        module = __import__(module_name)
        version = getattr(module, "__version__", "unknown")
        print(f"✓ {display_name}: {version}")
    except ImportError:
        errors.append(f"Missing dependency: {display_name}")
        print(f"✗ {display_name}: NOT INSTALLED")

# ==============================================================================
# STEP 4: VERIFY CONFIGURATION FILES
# ==============================================================================
print("\n" + "="*80)
print("STEP 4: VERIFYING CONFIGURATION FILES")
print("="*80)

config_files = [
    "config/belt_config.yaml",
    "config/enhanced_config.yaml",
]

for config_file in config_files:
    if Path(config_file).exists():
        try:
            import yaml
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            # Check critical fields
            critical_fields = ['data', 'model', 'training']
            missing = [f for f in critical_fields if f not in config]
            
            if missing:
                errors.append(f"{config_file}: Missing fields {missing}")
                print(f"✗ {config_file}: Missing {missing}")
            else:
                print(f"✓ {config_file}: Valid")
        except Exception as e:
            errors.append(f"{config_file}: {str(e)}")
            print(f"✗ {config_file}: {str(e)}")
    else:
        errors.append(f"{config_file}: File not found")
        print(f"✗ {config_file}: NOT FOUND")

# ==============================================================================
# STEP 5: VERIFY MODEL INSTANTIATION
# ==============================================================================
print("\n" + "="*80)
print("STEP 5: VERIFYING MODEL INSTANTIATION")
print("="*80)

try:
    import torch
    from models.dconformer import DConformer
    from models.vector_quantizer import VectorQuantizer
    from models.classifier import MLPClassifier
    
    # Test D-Conformer
    encoder = DConformer(
        d_model=840,
        num_blocks=6,
        num_heads=8,
        ffn_expansion=4,
        conv_kernel_size=31,
        dropout=0.1
    )
    print(f"✓ D-Conformer: {sum(p.numel() for p in encoder.parameters()):,} parameters")
    
    # Test Vector Quantizer
    vq = VectorQuantizer(
        input_dim=840,
        codebook_size=1024,
        codebook_dim=1024,
        beta=0.3
    )
    print(f"✓ Vector Quantizer: {sum(p.numel() for p in vq.parameters()):,} parameters")
    
    # Test Classifier (input_dim should match VQ codebook_dim)
    classifier = MLPClassifier(
        input_dim=1024,  # Match VQ codebook_dim
        hidden_dims=[512, 256],
        output_dim=500,
        dropout=0.3
    )
    print(f"✓ MLP Classifier: {sum(p.numel() for p in classifier.parameters()):,} parameters")
    
except Exception as e:
    errors.append(f"Model instantiation failed: {str(e)}")
    print(f"✗ Model instantiation failed: {str(e)}")

# ==============================================================================
# STEP 6: VERIFY FORWARD PASS
# ==============================================================================
print("\n" + "="*80)
print("STEP 6: VERIFYING FORWARD PASS")
print("="*80)

try:
    import torch
    
    # Create dummy input
    batch_size = 4
    seq_len = 105
    input_dim = 840
    x = torch.randn(batch_size, seq_len, input_dim)
    
    # Forward through encoder
    encoded = encoder(x)
    print(f"✓ Encoder output: {encoded.shape}")
    
    # Average pool
    pooled = encoded.mean(dim=1)
    print(f"✓ Pooled output: {pooled.shape}")
    
    # Vector quantization
    vq_loss, quantized, perplexity, encodings = vq(pooled)
    print(f"✓ VQ output: {quantized.shape}, loss: {vq_loss.item():.4f}")
    
    # Classification
    logits = classifier(quantized)
    print(f"✓ Classifier output: {logits.shape}")
    
    # Check shapes
    if logits.shape != (batch_size, 500):
        errors.append(f"Wrong output shape: {logits.shape}, expected ({batch_size}, 500)")
        print(f"✗ Wrong output shape: {logits.shape}")
    else:
        print(f"✓ Output shape correct: {logits.shape}")
    
except Exception as e:
    errors.append(f"Forward pass failed: {str(e)}")
    print(f"✗ Forward pass failed: {str(e)}")

# ==============================================================================
# STEP 7: VERIFY LOSS FUNCTIONS
# ==============================================================================
print("\n" + "="*80)
print("STEP 7: VERIFYING LOSS FUNCTIONS")
print("="*80)

try:
    import torch
    import torch.nn as nn
    from training.losses import ContrastiveLoss
    from training.enhanced_losses import LabelSmoothingCrossEntropy
    
    # Test CE loss
    criterion_ce = nn.CrossEntropyLoss()
    labels = torch.randint(0, 500, (batch_size,))
    loss_ce = criterion_ce(logits, labels)
    print(f"✓ CrossEntropyLoss: {loss_ce.item():.4f}")
    
    # Test label smoothing
    criterion_smooth = LabelSmoothingCrossEntropy(epsilon=0.1)
    loss_smooth = criterion_smooth(logits, labels)
    print(f"✓ LabelSmoothingCE: {loss_smooth.item():.4f}")
    
    # Test contrastive loss
    criterion_cl = ContrastiveLoss(temperature=0.07)
    word_embeddings = torch.randn(batch_size, 1024)
    loss_contrastive = criterion_cl(quantized, word_embeddings, labels)
    print(f"✓ ContrastiveLoss: {loss_contrastive.item():.4f}")
    
    # Test VQ loss
    print(f"✓ VQ Loss: {vq_loss.item():.4f}")
    
except Exception as e:
    errors.append(f"Loss function failed: {str(e)}")
    print(f"✗ Loss function failed: {str(e)}")

# ==============================================================================
# STEP 8: VERIFY BACKWARD PASS
# ==============================================================================
print("\n" + "="*80)
print("STEP 8: VERIFYING BACKWARD PASS")
print("="*80)

try:
    # Total loss
    alpha = 0.1
    lambda_vq = 1.0
    total_loss = loss_ce + alpha * loss_contrastive + lambda_vq * vq_loss
    print(f"✓ Total loss: {total_loss.item():.4f}")
    
    # Backward pass
    total_loss.backward()
    print("✓ Backward pass successful")
    
    # Check gradients
    has_grads = any(p.grad is not None and p.grad.abs().sum() > 0 
                    for p in list(encoder.parameters()) + list(vq.parameters()) + list(classifier.parameters()))
    if has_grads:
        print("✓ Gradients computed")
    else:
        warnings.append("No gradients found after backward pass")
        print("⚠ Warning: No gradients found")
    
except Exception as e:
    errors.append(f"Backward pass failed: {str(e)}")
    print(f"✗ Backward pass failed: {str(e)}")

# ==============================================================================
# STEP 9: VERIFY TRAINING SCRIPTS
# ==============================================================================
print("\n" + "="*80)
print("STEP 9: VERIFYING TRAINING SCRIPTS")
print("="*80)

training_scripts = [
    "experiments/model_with_bootstrapping.py",
    "experiments/model_enhanced.py",
]

for script in training_scripts:
    if Path(script).exists():
        print(f"✓ {script}: Found")
    else:
        errors.append(f"{script}: File not found")
        print(f"✗ {script}: NOT FOUND")

# ==============================================================================
# STEP 10: VERIFY PATHS IN SCRIPTS
# ==============================================================================
print("\n" + "="*80)
print("STEP 10: VERIFYING PATHS IN SCRIPTS")
print("="*80)

# Check prepare_data.py uses correct paths
with open("prepare_data.py", 'r') as f:
    content = f.read()
    
    if 'dataset/ZuCo' in content:
        print("✓ prepare_data.py: Uses local dataset path")
    else:
        warnings.append("prepare_data.py may have incorrect dataset path")
        print("⚠ prepare_data.py: Check dataset path")
    
    if "sys.path.insert(0, '.')" in content or "sys.path.insert(0, str(Path(__file__).parent))" in content:
        print("✓ prepare_data.py: Standalone imports configured")
    else:
        warnings.append("prepare_data.py may have import issues")
        print("⚠ prepare_data.py: Check import paths")

# ==============================================================================
# FINAL SUMMARY
# ==============================================================================
print("\n" + "="*80)
print("VERIFICATION SUMMARY")
print("="*80)

if errors:
    print(f"\n❌ FOUND {len(errors)} ERROR(S):")
    for i, error in enumerate(errors, 1):
        print(f"  {i}. {error}")
    print("\n⚠ PLEASE FIX ERRORS BEFORE TRAINING!")
    sys.exit(1)
elif warnings:
    print(f"\n⚠ FOUND {len(warnings)} WARNING(S):")
    for i, warning in enumerate(warnings, 1):
        print(f"  {i}. {warning}")
    print("\n✓ No critical errors, but check warnings")
    print("✅ YOU CAN PROCEED WITH TRAINING (with caution)")
else:
    print("\n✅ ALL CHECKS PASSED!")
    print("\n🎉 EVERYTHING LOOKS GOOD!")
    print("\nYou can now train models:")
    print("  • python prepare_data.py")
    print("  • python experiments/model_enhanced.py --config config/enhanced_config.yaml --mode train")
    print("  • Or use train_all.bat / train_all.sh")

print("\n" + "="*80)
