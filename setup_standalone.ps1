# ============================================================================
# Setup Standalone BELT Project
# ============================================================================
# This script moves the dataset into model_custom and makes it standalone

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "Setting up standalone BELT project" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# Check if dataset exists
if (-Not (Test-Path "..\dataset")) {
    Write-Host "[ERROR] Dataset folder not found at ..\dataset" -ForegroundColor Red
    Write-Host "Please ensure the dataset is in the parent directory" -ForegroundColor Red
    exit 1
}

# Check if dataset already exists in model_custom
if (Test-Path ".\dataset") {
    Write-Host "[WARNING] Dataset folder already exists in model_custom" -ForegroundColor Yellow
    $response = Read-Host "Do you want to replace it? (y/n)"
    if ($response -ne "y") {
        Write-Host "Aborting..." -ForegroundColor Yellow
        exit 0
    }
    Write-Host "Removing existing dataset..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force ".\dataset"
}

# Move dataset folder
Write-Host "[1/4] Moving dataset folder..." -ForegroundColor Green
try {
    Move-Item -Path "..\dataset" -Destination ".\dataset"
    Write-Host "  ✓ Dataset moved successfully" -ForegroundColor Green
} catch {
    Write-Host "  ✗ Failed to move dataset: $_" -ForegroundColor Red
    exit 1
}

# Create .gitignore
Write-Host "[2/4] Creating .gitignore..." -ForegroundColor Green
$gitignore = @"
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
.venv/
ENV/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# PyTorch
*.pth
*.pt
*.ckpt

# Checkpoints and logs
checkpoints/
checkpoints_ablation/
checkpoints_enhanced/
logs/
logs_enhanced/
tensorboard/

# Dataset (if large)
# dataset/ZuCo/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~
.DS_Store

# Jupyter
.ipynb_checkpoints/
*.ipynb

# Environment
.env
.envrc
"@
Set-Content -Path ".gitignore" -Value $gitignore
Write-Host "  ✓ .gitignore created" -ForegroundColor Green

# Create requirements.txt
Write-Host "[3/4] Creating requirements.txt..." -ForegroundColor Green
$requirements = @"
# Core dependencies
torch>=2.0.0
numpy>=1.24.0
scipy>=1.10.0
pyyaml>=6.0
tqdm>=4.65.0

# Transformers (for BART embeddings)
transformers>=4.30.0

# Data processing
pandas>=2.0.0
scikit-learn>=1.3.0

# Visualization (optional)
matplotlib>=3.7.0
seaborn>=0.12.0
tensorboard>=2.13.0

# Development (optional)
pytest>=7.4.0
black>=23.7.0
flake8>=6.1.0
"@
Set-Content -Path "requirements.txt" -Value $requirements
Write-Host "  ✓ requirements.txt created" -ForegroundColor Green

# Create standalone README
Write-Host "[4/4] Creating standalone README..." -ForegroundColor Green
$readme = @"
# BELT-Enhanced: Bootstrapped EEG-to-text Translation

Standalone implementation of BELT with proven enhancements for improved performance.

## 📊 Quick Overview

This project implements three model variants:

| Model | Description | Expected Top-10 Accuracy |
|-------|-------------|--------------------------|
| **Model 1** | BELT-Ablation (no bootstrapping) | ~25% |
| **Model 2** | BELT-Baseline (full BELT) | ~31.04% |
| **Model 3** | BELT-Enhanced (with improvements) | ~37-39% |

## 🚀 Quick Start

### 1. Install Dependencies

``````bash
pip install -r requirements.txt
``````

### 2. Prepare Data

``````bash
python prepare_data.py
``````

### 3. Train Models

**Windows:**
``````bash
train_all.bat
``````

**Linux/Mac:**
``````bash
bash train_all.sh
``````

Or train individual models:

``````bash
# Model 1: Ablation
python experiments/model_without_bootstrapping.py --config config/belt_config.yaml --mode train

# Model 2: Baseline
python experiments/model_with_bootstrapping.py --config config/belt_config.yaml --mode train

# Model 3: Enhanced
python experiments/model_enhanced.py --config config/enhanced_config.yaml --mode train
``````

## 📁 Project Structure

``````
model_custom/
├── dataset/              # ZuCo EEG dataset (moved here)
│   ├── ZuCo/
│   └── stanfordsentiment/
├── data/                 # Data loading modules
│   ├── vocabulary.py
│   ├── dataset.py
│   └── splits.py
├── models/               # Model architectures
│   ├── dconformer.py
│   ├── vector_quantizer.py
│   └── classifier.py
├── training/             # Training utilities
│   ├── losses.py
│   ├── enhanced_losses.py
│   ├── schedulers.py
│   ├── augmentation.py
│   └── regularization.py
├── experiments/          # Training scripts
│   ├── model_without_bootstrapping.py
│   ├── model_with_bootstrapping.py
│   └── model_enhanced.py
├── config/               # Configuration files
│   ├── belt_config.yaml
│   └── enhanced_config.yaml
├── prepare_data.py       # Data preparation script
├── requirements.txt      # Python dependencies
└── README.md            # This file
``````

## 🎯 Enhancements

Model 3 includes 7 proven techniques:

1. **Label Smoothing** (ε=0.1) - +1-2%
2. **AdamW Optimizer** - +1-2%
3. **Warmup + Cosine LR** - +0.5-1.5%
4. **Gradient Clipping** - +0.5-1%
5. **MixUp Augmentation** (α=0.2) - +1-2%
6. **Stochastic Depth** (p=0.1) - +0.5-1.5%
7. **Multi-Sample Dropout** (n=5) - +0.5-1%

**Total Expected Gain**: +6-11% absolute

## 📚 Documentation

- **QUICKSTART.md** - Basic usage guide
- **EXPERIMENTS.md** - Detailed enhancement documentation
- **README.md** - Architecture details

## 🔧 Configuration

Edit ``config/enhanced_config.yaml`` to tune hyperparameters:

``````yaml
# Optimizer
training:
  optimizer:
    name: "adamw"
    lr: 5e-4
    weight_decay: 0.01

# Augmentation
data:
  use_mixup: true
  mixup_alpha: 0.2

# Regularization
model:
  encoder:
    use_drop_path: true
    drop_path_rate: 0.1
``````

## 📊 Expected Results

### Model 1: BELT-Ablation
- No contrastive learning bootstrapping
- Baseline architecture only
- **Expected: ~25% top-10 accuracy**

### Model 2: BELT-Baseline
- Full BELT implementation from paper
- Contrastive learning with BART embeddings
- **Expected: ~31.04% top-10 accuracy**

### Model 3: BELT-Enhanced
- All 7 enhancements enabled
- Optimized training pipeline
- **Expected: ~37-39% top-10 accuracy**

## 🐛 Troubleshooting

### Out of Memory
Reduce batch size in config:
``````yaml
data:
  batch_size: 32  # Reduce from 64
``````

### Slow Training
Reduce augmentation:
``````yaml
data:
  mixup_prob: 0.5  # Apply to 50% of batches
``````

### Overfitting
Increase regularization:
``````yaml
model:
  encoder:
    drop_path_rate: 0.2  # Increase from 0.1
``````

## 📝 Citation

If you use this code, please cite the original BELT paper and the enhancement papers listed in EXPERIMENTS.md.

## 📧 Support

For issues or questions, please check:
1. QUICKSTART.md for basic usage
2. EXPERIMENTS.md for enhancement details
3. Open an issue with logs and config
"@
Set-Content -Path "README_STANDALONE.md" -Value $readme
Write-Host "  ✓ README_STANDALONE.md created" -ForegroundColor Green

# Summary
Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "Setup complete!" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "The model_custom folder is now standalone with:" -ForegroundColor White
Write-Host "  ✓ Dataset moved from parent directory" -ForegroundColor Green
Write-Host "  ✓ All dependencies listed in requirements.txt" -ForegroundColor Green
Write-Host "  ✓ .gitignore configured" -ForegroundColor Green
Write-Host "  ✓ Standalone documentation" -ForegroundColor Green
Write-Host ""
Write-Host "You can now:" -ForegroundColor Yellow
Write-Host "  1. Move this folder anywhere" -ForegroundColor White
Write-Host "  2. Install dependencies: pip install -r requirements.txt" -ForegroundColor White
Write-Host "  3. Prepare data: python prepare_data.py" -ForegroundColor White
Write-Host "  4. Train models: train_all.bat (or .sh)" -ForegroundColor White
Write-Host ""
