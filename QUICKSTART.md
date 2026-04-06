# Quick Start

This quick start reflects the repository as it exists now, not the earlier BELT-only plan.

## 1. Environment

Windows PowerShell:

```powershell
.\.venv\Scripts\activate
```

Install dependencies if needed:

```powershell
pip install -r requirements.txt
```

## 2. Prepare Processed Artifacts

If you need the classic processed-feature pipeline artifacts:

```powershell
.\.venv\Scripts\python.exe prepare_data.py
.\.venv\Scripts\python.exe prepare_sentence_splits.py
```

This creates vocabulary and split files in `data/`.

## 3. Main Training Paths

### A. BELT-style processed baseline

```powershell
.\.venv\Scripts\python.exe experiments\model_with_bootstrapping.py --config config\belt_config.yaml
```

Use this for:

- processed BELT-style baselines
- VQ / no-VQ comparisons
- contrastive-loss ablations

### B. Raw grouped word-level baseline

```powershell
.\.venv\Scripts\python.exe experiments\raw_multimodal_baseline.py --config config\raw_multimodal_mixed_word_patched.yaml
```

Use this for:

- grouped raw EEG + eye-tracking experiments
- temporal modeling over word-level grouped raw signals

Resume example:

```powershell
.\.venv\Scripts\python.exe experiments\raw_multimodal_baseline.py --config config\raw_multimodal_mixed_word_patched.yaml --resume results\raw_multimodal_mixed_word_patched\checkpoint_epoch_4.pt
```

### C. Hybrid processed + raw baseline

```powershell
.\.venv\Scripts\python.exe experiments\hybrid_processed_raw_baseline.py --config config\hybrid_processed_raw_mixed.yaml
```

Use this for:

- combined processed summary + grouped raw EEG+ET experiments

## 4. Smoke Tests

If you only want to verify that a pipeline runs end-to-end:

```powershell
.\.venv\Scripts\python.exe experiments\raw_multimodal_baseline.py --config config\raw_multimodal_word_smoke.yaml
.\.venv\Scripts\python.exe experiments\hybrid_processed_raw_baseline.py --config config\hybrid_processed_raw_smoke.yaml
```

These are plumbing checks, not meaningful benchmark runs.

## 5. Current Anchors

The useful reference points right now are:

- processed BELT-style best band: about `29.4` to `29.55%` Top-10
- grouped raw word-level best band: about `28.24%` Top-10
- simple hybrid CE-only band: about `28.2%` Top-10

Interpretation:

- raw signal helps
- grouping fixations helps a lot
- simple fusion alone is not enough to beat the processed baseline

## 6. What To Read First

If you need the full project context, start here:

- [README.md](/C:/Users/n1sha/Desktop/model_custom/README.md)
- [approach_findings_documentation.tex](/C:/Users/n1sha/Desktop/model_custom/approach_findings_documentation.tex)

Those two are the current source of truth.

## 7. Most Likely Next Architecture

The strongest next implementation direction is:

- BELT-style processed branch with real D-Conformer
- grouped raw EEG+ET branch
- stronger fusion
- contextual semantic alignment

So if you are resuming work after a break, that is the mainline to keep in mind.
