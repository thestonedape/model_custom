# BELT Model - Quick Start Guide

Complete implementation of BELT (Boosting with EEG Language Transformer) for word classification.

## 🚀 Quick Start (3 Steps)

### Step 1: Prepare Data (5 minutes)

```bash
# Activate your virtual environment
# Windows:
.venv\Scripts\activate

# Run data preparation
python model_custom/prepare_data.py
```

This will:
- ✓ Build vocabulary (top-500 words from ZuCo)
- ✓ Create 80/10/10 train/dev/test splits
- ✓ Save to `model_custom/data/`

### Step 2: Train Model 1 - Ablation (Hours)

```bash
# Model 1: Without bootstrapping (L_ce + L_vq only)
python model_custom/experiments/model_without_bootstrapping.py
```

**Expected Result:** ~25% Top-10 accuracy  
**Training Time:** ~2-4 hours (depends on GPU)  
**Saves to:** `results/ablation_results/`

### Step 3: Train Model 2 - Full BELT (Hours)

```bash
# Model 2: With bootstrapping (L_ce + α*L_cl + λ*L_vq)
python model_custom/experiments/model_with_bootstrapping.py
```

**Expected Result:** ~31.04% Top-10 accuracy  
**Training Time:** ~3-5 hours (depends on GPU)  
**Saves to:** `results/main_results/`

---

## 📊 What You Get

### After Training:

**Model 1 Results** (`results/ablation_results/`):
- `best_model.pt` - Best checkpoint
- `final_results.json` - Test metrics
- `training_history.json` - Per-epoch metrics

**Model 2 Results** (`results/main_results/`):
- `best_model.pt` - Best checkpoint
- `final_results.json` - Test metrics
- `training_history.json` - Per-epoch metrics

### Expected Performance:

| Model | Loss Function | Top-10 Accuracy |
|-------|--------------|-----------------|
| Model 1 (Ablation) | L_ce + L_vq | ~25% |
| Model 2 (Full BELT) | L_ce + α*L_cl + λ*L_vq | **~31.04%** |
| **Improvement** | | **+5.78%** |

---

## 🧪 Test Individual Components

Before full training, you can test each module:

```bash
# Data components
python model_custom/data/vocabulary.py
python model_custom/data/splits.py
python model_custom/data/dataset.py

# Model components
python model_custom/models/conformer_block.py
python model_custom/models/dconformer.py
python model_custom/models/vector_quantizer.py
python model_custom/models/classifier.py

# Training components
python model_custom/training/losses.py
python model_custom/training/metrics.py
```

---

## 📁 Project Structure

```
model_custom/
├── prepare_data.py              # [RUN FIRST] Data preparation
│
├── experiments/
│   ├── model_without_bootstrapping.py    # Model 1 (ablation)
│   └── model_with_bootstrapping.py       # Model 2 (full BELT)
│
├── data/
│   ├── vocabulary.py           # Top-500 word selection
│   ├── dataset.py              # BELT dataset class
│   ├── splits.py               # 80/10/10 splitting
│   └── [Generated files]
│       ├── vocabulary_top500.pkl
│       └── splits.pkl
│
├── models/
│   ├── dconformer.py          # 6-layer Conformer encoder
│   ├── conformer_block.py     # Conformer block
│   ├── convolution_module.py  # Conv module
│   ├── vector_quantizer.py    # VQ with codebook
│   └── classifier.py          # MLP head
│
├── training/
│   ├── losses.py              # L_ce, L_vq, L_cl
│   ├── metrics.py             # Top-K accuracy
│   └── trainer.py             # Training loop
│
├── config/
│   └── belt_config.yaml       # All hyperparameters
│
└── results/
    ├── ablation_results/      # Model 1 outputs
    └── main_results/          # Model 2 outputs
```

---

## ⚙️ Configuration

Edit `model_custom/config/belt_config.yaml` to change:

**Training:**
- `epochs: 60` (default)
- `batch_size: 64`
- `learning_rate: 5.0e-6`
- `optimizer: sgd`

**Model:**
- `num_blocks: 6` (Conformer layers)
- `codebook_size: 1024`
- `vocab_size: 500`

**Loss Weights:**
- `alpha: 0.9` (contrastive weight)
- `lambda: 1.0` (VQ weight)

---

## 🔍 Monitor Training

Training logs show:
- Batch-level progress every 10 batches
- Epoch-level metrics (loss, Top-1/5/10 accuracy)
- Best model tracking
- Checkpoint saving

Example output:
```
EPOCH 1/60
================================================================================
Learning rate: 5.00e-06
  Batch 0/150 | Loss: 6.2145 | Time: 2.3s
  Batch 10/150 | Loss: 6.1823 | Time: 25.1s
  ...

Epoch 1 Train Metrics:
  Top-1 Acc:  0.0234 (2.34%)
  Top-5 Acc:  0.0891 (8.91%)
  Top-10 Acc: 0.1456 (14.56%)
  Total Loss: 6.1234

Evaluating on dev set...
Dev Metrics:
  Top-1 Acc:  0.0256 (2.56%)
  Top-5 Acc:  0.0923 (9.23%)
  Top-10 Acc: 0.1512 (15.12%)

*** New best model! Top-10 Acc: 0.1512 (15.12%) ***
```

---

## 🎯 Troubleshooting

### Out of Memory (OOM)?
```yaml
# In belt_config.yaml, reduce:
batch_size: 32  # or 16
```

### Training too slow?
```yaml
# Reduce number of blocks:
num_blocks: 4  # instead of 6

# Or reduce epochs:
epochs: 30  # for quick testing
```

### Data not found?
```bash
# Make sure you ran data preparation:
python model_custom/prepare_data.py

# Check that these exist:
ls model_custom/data/vocabulary_top500.pkl
ls model_custom/data/splits.pkl
```

---

## 📈 Compare Results

After training both models:

```python
import json

# Load results
with open('results/ablation_results/final_results.json') as f:
    ablation = json.load(f)

with open('results/main_results/final_results.json') as f:
    full_belt = json.load(f)

# Compare
print(f"Model 1 (Ablation): {ablation['test_metrics']['top10_acc']:.4f}")
print(f"Model 2 (Full BELT): {full_belt['test_metrics']['top10_acc']:.4f}")
print(f"Improvement: +{(full_belt['test_metrics']['top10_acc'] - ablation['test_metrics']['top10_acc'])*100:.2f}%")
```

---

## 🎓 Paper Reference

This implementation matches:
- **BELT Paper**: Section III-D.3 (EEG-to-word Classification)
- **Table I**: Architecture specifications
- **Equation 7**: Combined loss L = L_ce + α*L_cl^w + λ*L_vq
- **Table VI**: Ablation study results

---

## 📝 Notes

1. **Training method**: JOINT optimization (all losses together from epoch 1)
2. **Not staged**: No pre-training or fine-tuning phases
3. **EEG type**: Uses GD (Gaze Duration) features by default
4. **Vocabulary**: Top-500 most frequent words
5. **Dataset**: Combines task1-SR, task2-NR, task3-TSR from ZuCo

---

## ✅ Checklist

- [ ] Data prepared (`python model_custom/prepare_data.py`)
- [ ] Virtual environment activated
- [ ] GPU available (optional but recommended)
- [ ] Model 1 trained (ablation baseline)
- [ ] Model 2 trained (full BELT)
- [ ] Results compared

**Ready to start? Run:**
```bash
python model_custom/prepare_data.py
```
