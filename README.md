# EEG-to-Word Decoding Experiments on ZuCo

This repository started as a BELT-style EEG-to-word classification implementation and has since grown into a broader research sandbox for:

- BELT-style processed-feature baselines
- raw fixation-level EEG + eye-tracking modeling
- grouped word-level raw modeling
- hybrid processed + raw architectures

The project currently targets **500-way word classification** on ZuCo with **Top-1 / Top-5 / Top-10** evaluation, with **Top-10 accuracy** as the main comparison metric.

## Current Status

What we know so far:

- Strong processed BELT-style runs sit in roughly the **29.4 to 29.55\% Top-10** band.
- Simple training-trick upgrades did **not** break that ceiling in a meaningful way.
- Raw fixation-only modeling was too noisy and plateaued much lower.
- Grouping raw fixations by word was a major improvement and pushed raw modeling to about **28.24\% Top-10**.
- A simple processed+raw hybrid with CE-only training also plateaued near the **28.2\%** band.

The current working hypothesis is:

- the remaining bottleneck is no longer basic preprocessing alone
- it is mostly in **representation quality, fusion design, and language-aware supervision**

The strongest next direction is likely:

- **BELT core processed encoder + grouped raw EEG+ET branch + stronger contextual semantic alignment**

## Main Pipelines in This Repo

### 1. BELT-style processed-feature pipeline

This is the closest path to the original repo goal:

- processed EEG summary features (`840 = 105 electrodes x 8 bands`)
- D-Conformer encoder
- optional vector quantizer
- optional BART-based contrastive branch
- MLP classifier

Main files:

- [config/belt_config.yaml](/C:/Users/n1sha/Desktop/model_custom/config/belt_config.yaml)
- [experiments/model_with_bootstrapping.py](/C:/Users/n1sha/Desktop/model_custom/experiments/model_with_bootstrapping.py)
- [models/dconformer.py](/C:/Users/n1sha/Desktop/model_custom/models/dconformer.py)
- [models/vector_quantizer.py](/C:/Users/n1sha/Desktop/model_custom/models/vector_quantizer.py)
- [training/losses.py](/C:/Users/n1sha/Desktop/model_custom/training/losses.py)

### 2. Raw multimodal pipeline

This path uses:

- raw fixation-level EEG
- raw fixation-level eye-tracking
- fixation metrics
- temporal patching/downsampling
- transformer-based temporal modeling

Main files:

- [data/raw_fixation_dataset.py](/C:/Users/n1sha/Desktop/model_custom/data/raw_fixation_dataset.py)
- [data/raw_fixation_torch_dataset.py](/C:/Users/n1sha/Desktop/model_custom/data/raw_fixation_torch_dataset.py)
- [scripts/build_word_level_cache.py](/C:/Users/n1sha/Desktop/model_custom/scripts/build_word_level_cache.py)
- [models/raw_multimodal.py](/C:/Users/n1sha/Desktop/model_custom/models/raw_multimodal.py)
- [experiments/raw_multimodal_baseline.py](/C:/Users/n1sha/Desktop/model_custom/experiments/raw_multimodal_baseline.py)

### 3. Hybrid processed + raw pipeline

This path joins:

- processed BELT-style summary EEG features
- grouped raw word-level EEG
- grouped raw eye-tracking
- fixation metrics

and fuses them with a gated hybrid model.

Main files:

- [scripts/build_hybrid_word_cache.py](/C:/Users/n1sha/Desktop/model_custom/scripts/build_hybrid_word_cache.py)
- [data/hybrid_word_torch_dataset.py](/C:/Users/n1sha/Desktop/model_custom/data/hybrid_word_torch_dataset.py)
- [models/hybrid_processed_raw.py](/C:/Users/n1sha/Desktop/model_custom/models/hybrid_processed_raw.py)
- [experiments/hybrid_processed_raw_baseline.py](/C:/Users/n1sha/Desktop/model_custom/experiments/hybrid_processed_raw_baseline.py)
- [config/hybrid_processed_raw_mixed.yaml](/C:/Users/n1sha/Desktop/model_custom/config/hybrid_processed_raw_mixed.yaml)

## Repository Layout

```text
model_custom/
├── config/                  # experiment configs
├── data/                    # dataset code, splits, vocab, torch datasets
├── dataset/                 # local ZuCo data and related resources
├── experiments/             # train/eval entry points
├── models/                  # model definitions
├── scripts/                 # cache builders and utilities
├── training/                # loss utilities and training support
├── results/                 # active experiment outputs
└── approach_findings_documentation.tex
```

## Useful Commands

### Prepare classic processed artifacts

```powershell
.\.venv\Scripts\python.exe prepare_data.py
.\.venv\Scripts\python.exe prepare_sentence_splits.py
```

### Train BELT-style processed model

```powershell
.\.venv\Scripts\python.exe experiments\model_with_bootstrapping.py --config config\belt_config.yaml
```

### Train raw grouped word-level model

```powershell
.\.venv\Scripts\python.exe experiments\raw_multimodal_baseline.py --config config\raw_multimodal_mixed_word_patched.yaml
```

### Resume raw grouped run

```powershell
.\.venv\Scripts\python.exe experiments\raw_multimodal_baseline.py --config config\raw_multimodal_mixed_word_patched.yaml --resume results\raw_multimodal_mixed_word_patched\checkpoint_epoch_4.pt
```

### Train hybrid processed+raw model

```powershell
.\.venv\Scripts\python.exe experiments\hybrid_processed_raw_baseline.py --config config\hybrid_processed_raw_mixed.yaml
```

## Active Data Artifacts

The current hybrid pipeline relies on hybrid joined caches in `data/cache/`:

- `hybrid_word_task2_NR.pkl`
- `hybrid_word_task2_NR_2_0_part1.pkl`
- `hybrid_word_task2_NR_2_0_part2.pkl`
- `hybrid_word_task2_NR_2_0_part3.pkl`
- `hybrid_word_task3_TSR.pkl`
- `hybrid_word_task3_TSR_2_0_part1.pkl`
- `hybrid_word_task3_TSR_2_0_part2a.pkl`
- `hybrid_word_task3_TSR_2_0_part2b.pkl`
- `hybrid_word_task3_TSR_2_0_part3a.pkl`
- `hybrid_word_task3_TSR_2_0_part3b.pkl`

These are built from the raw grouped caches plus the processed EEG summaries by joining on:

- task
- subject id
- sentence index
- word index

## Important Findings

### What helped

- moving from isolated raw fixations to grouped word-level raw samples
- normalizing raw EEG and eye-tracking sequences
- preserving temporal structure instead of using only summary features

### What did not help enough

- weighted sampling
- class-balanced CE
- simple stronger regularization stacks
- CE-only raw fusion
- CE-only hybrid fusion

### What is still missing

- stronger processed branch inside the hybrid model
- contextual semantic alignment instead of isolated-word matching
- better use of sentence context
- better multimodal fusion than simple gating

## Documentation

The main project write-up is:

- [approach_findings_documentation.tex](/C:/Users/n1sha/Desktop/model_custom/approach_findings_documentation.tex)

It includes:

- the project history
- why each change was made
- current architecture diagrams
- what failed and why
- the real bottlenecks
- the most promising research directions and search keywords

## Research Direction Going Forward

The current most defensible next-step architecture is:

- BELT-style processed branch with the real D-Conformer
- grouped raw EEG+ET word-level branch
- stronger fusion
- contextual semantic alignment on top of the fused representation

In short:

- not raw-only
- not CE-only hybrid
- not more small training tricks

The project is now at the point where **architecture and supervision quality matter more than another round of minor tuning**.

## Notes

- This repository is actively experimental.
- Some older docs such as [QUICKSTART.md](/C:/Users/n1sha/Desktop/model_custom/QUICKSTART.md) and [EXPERIMENTS.md](/C:/Users/n1sha/Desktop/model_custom/EXPERIMENTS.md) still reflect older BELT-only ambitions and are no longer the best summary of the current state.
- The README and the LaTeX documentation should be treated as the main source of truth going forward.
