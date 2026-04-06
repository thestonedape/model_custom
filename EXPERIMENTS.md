# Experiments and Findings

This file summarizes the main experiment families in the repository and what they taught us.

## 1. Processed BELT-Style Experiments

These experiments use processed EEG summary features and BELT-style training code.

Relevant files:

- [config/belt_config.yaml](/C:/Users/n1sha/Desktop/model_custom/config/belt_config.yaml)
- [experiments/model_with_bootstrapping.py](/C:/Users/n1sha/Desktop/model_custom/experiments/model_with_bootstrapping.py)

What was tried:

- BELT-style processed baseline
- no-contrastive ablations
- true no-VQ runs
- weak-contrastive warmup variants
- richer summary-feature inputs
- class-balanced CE
- weighted sampling

What we learned:

- the processed baseline is strong and stable
- simple training tricks do not break the `~29.5%` Top-10 ceiling
- weighted sampling and class-balanced CE were not helpful
- current contrastive usage was not the missing breakthrough

## 2. Raw Fixation-Level Experiments

These experiments moved away from summary features and into raw signal.

Relevant files:

- [data/raw_fixation_dataset.py](/C:/Users/n1sha/Desktop/model_custom/data/raw_fixation_dataset.py)
- [data/raw_fixation_torch_dataset.py](/C:/Users/n1sha/Desktop/model_custom/data/raw_fixation_torch_dataset.py)
- [models/raw_multimodal.py](/C:/Users/n1sha/Desktop/model_custom/models/raw_multimodal.py)
- [experiments/raw_multimodal_baseline.py](/C:/Users/n1sha/Desktop/model_custom/experiments/raw_multimodal_baseline.py)

What was tried:

- raw fixation-level EEG
- raw fixation-level eye-tracking
- temporal transformer baseline

What we learned:

- raw-only fixation-level modeling is too noisy
- preserving time is useful, but isolated fixations are a weak prediction unit

## 3. Grouped Raw Word-Level Experiments

This was the first major raw-data improvement.

Relevant files:

- [scripts/build_word_level_cache.py](/C:/Users/n1sha/Desktop/model_custom/scripts/build_word_level_cache.py)
- [config/raw_multimodal_mixed_word_patched.yaml](/C:/Users/n1sha/Desktop/model_custom/config/raw_multimodal_mixed_word_patched.yaml)

What changed:

- multiple fixations for the same word were grouped
- raw EEG and ET were normalized
- temporal patching/downsampling was added

What we learned:

- this was a real gain
- grouped raw word-level modeling reached about `28.24%` Top-10
- preprocessing mattered a lot, but still did not surpass the processed baseline

## 4. Hybrid Processed + Raw Experiments

These experiments tried to combine low-noise processed features with richer raw signals.

Relevant files:

- [scripts/build_hybrid_word_cache.py](/C:/Users/n1sha/Desktop/model_custom/scripts/build_hybrid_word_cache.py)
- [data/hybrid_word_torch_dataset.py](/C:/Users/n1sha/Desktop/model_custom/data/hybrid_word_torch_dataset.py)
- [models/hybrid_processed_raw.py](/C:/Users/n1sha/Desktop/model_custom/models/hybrid_processed_raw.py)
- [experiments/hybrid_processed_raw_baseline.py](/C:/Users/n1sha/Desktop/model_custom/experiments/hybrid_processed_raw_baseline.py)
- [config/hybrid_processed_raw_mixed.yaml](/C:/Users/n1sha/Desktop/model_custom/config/hybrid_processed_raw_mixed.yaml)

What was tried:

- processed summary EEG branch
- grouped raw EEG branch
- grouped raw ET branch
- fixation metrics
- gated fusion
- CE-only training

What we learned:

- hybrid training is stable
- simple fusion does not automatically beat the processed baseline
- CE-only hybrid also plateaued around the high-`27` to low-`28` band

## 5. Current Working Diagnosis

The most important conclusion right now is:

- we are no longer mainly blocked by basic preprocessing mistakes
- we are now more blocked by representation design, fusion quality, and supervision quality

In practical terms:

- raw-only is not enough
- simple hybrid is not enough
- the next gain likely requires a better processed branch plus stronger language-aware supervision

## 6. Best Next Experiment

The strongest next architecture to implement is:

- BELT-style processed branch with the real D-Conformer
- grouped raw EEG+ET branch
- stronger fusion
- contextual semantic alignment

This is the first setup that really combines:

- BELT’s strongest encoder prior
- the strongest raw improvement found in this repo
- the missing language-side signal

## 7. What Not To Spend Time On Next

Low-priority directions for the immediate next cycle:

- more weighted-sampler variants
- more class-balanced CE variants
- more CE-only raw-only sweeps
- more CE-only hybrid variants
- blindly adding VQ or focal loss before fixing the representation and supervision

## 8. Full Write-Up

For the full rationale, architecture diagrams, and research keywords, see:

- [approach_findings_documentation.tex](/C:/Users/n1sha/Desktop/model_custom/approach_findings_documentation.tex)
