"""
Build a hybrid word-level cache that joins grouped raw samples with processed
BELT-style summary EEG features.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Build hybrid processed+raw word caches")
    parser.add_argument("--input", required=True, help="Input raw word-level cache pickle")
    parser.add_argument("--output", required=True, help="Output hybrid cache pickle")
    parser.add_argument("--dataset-root", default="dataset/ZuCo", help="Root directory for processed ZuCo pickles")
    parser.add_argument("--eeg-types", nargs="+", default=["GD"], help="Processed EEG summary types to join")
    parser.add_argument("--include-n-fixations", action="store_true", help="Append nFixations to processed vector")
    return parser.parse_args()


def _task_pickle_path(dataset_root: str, task: str) -> Path:
    return Path(dataset_root) / task / "pickle" / f"{task}-dataset.pickle"


def _extract_single_eeg_features(word_data: Dict[str, object], eeg_type: str) -> Optional[np.ndarray]:
    try:
        eeg_dict = word_data["word_level_EEG"][eeg_type]
        bands = ["t1", "t2", "a1", "a2", "b1", "b2", "g1", "g2"]
        features = []
        for band in bands:
            band_key = f"{eeg_type}_{band}"
            if band_key not in eeg_dict:
                return None
            band_data = eeg_dict[band_key]
            if isinstance(band_data, list):
                band_data = np.asarray(band_data)
            features.append(np.asarray(band_data, dtype=np.float32))
        fused = np.concatenate(features, axis=0)
        if fused.shape[0] != 840 or np.isnan(fused).any() or np.isinf(fused).any():
            return None
        return fused.astype(np.float32)
    except (KeyError, TypeError, ValueError):
        return None


def _extract_processed_eeg(
    word_data: Dict[str, object],
    eeg_types: Sequence[str],
    include_n_fixations: bool,
) -> Optional[np.ndarray]:
    features: List[np.ndarray] = []
    for eeg_type in eeg_types:
        eeg = _extract_single_eeg_features(word_data, eeg_type)
        if eeg is None:
            return None
        features.append(eeg)
    if include_n_fixations:
        features.append(np.asarray([float(word_data.get("nFixations", 0.0))], dtype=np.float32))
    fused = np.concatenate(features, axis=0)
    if np.isnan(fused).any() or np.isinf(fused).any():
        return None
    return fused.astype(np.float32)


def _normalize_token(text: object) -> str:
    return " ".join(str(text).strip().split()).lower()


def _resolve_processed_word_index(
    sentence_data: Dict[str, object],
    raw_word_idx: int,
    sample_word: str,
) -> Optional[int]:
    tokens_with_mask = sentence_data.get("word_tokens_with_mask")
    words = sentence_data.get("word")
    if tokens_with_mask is not None and words is not None and raw_word_idx < len(tokens_with_mask):
        token = _normalize_token(tokens_with_mask[raw_word_idx])
        if token and token != "[mask]":
            processed_idx = sum(1 for value in tokens_with_mask[: raw_word_idx + 1] if _normalize_token(value) != "[mask]") - 1
            if 0 <= processed_idx < len(words) and words[processed_idx] is not None:
                processed_word = _normalize_token(words[processed_idx].get("content", ""))
                if processed_word == sample_word:
                    return processed_idx

    words = sentence_data.get("word")
    if words is not None and raw_word_idx < len(words) and words[raw_word_idx] is not None:
        processed_word = _normalize_token(words[raw_word_idx].get("content", ""))
        if processed_word == sample_word:
            return raw_word_idx

    return None


def build_hybrid_cache(
    input_path: str,
    output_path: str,
    dataset_root: str,
    eeg_types: Sequence[str],
    include_n_fixations: bool,
) -> Dict[str, object]:
    input_file = Path(input_path)
    output_file = Path(output_path)

    with open(input_file, "rb") as handle:
        payload = pickle.load(handle)

    task_names = sorted({sample["task"] for sample in payload["samples"]})
    processed_payloads: Dict[str, Dict[str, object]] = {}
    for task in task_names:
        processed_path = _task_pickle_path(dataset_root=dataset_root, task=task)
        if not processed_path.exists():
            raise FileNotFoundError(f"Processed pickle missing for task {task}: {processed_path}")
        with open(processed_path, "rb") as handle:
            processed_payloads[task] = pickle.load(handle)

    hybrid_samples: List[Dict[str, object]] = []
    missing_subject = 0
    missing_sentence = 0
    missing_word = 0
    missing_processed = 0
    word_mismatch = 0

    for sample in payload["samples"]:
        task = sample["task"]
        subject_id = sample["subject_id"]
        sentence_idx = int(sample["sentence_idx"])
        word_idx = int(sample["word_idx"])
        processed_task = processed_payloads[task]

        if subject_id not in processed_task or processed_task[subject_id] is None:
            missing_subject += 1
            continue
        sentences = processed_task[subject_id]
        if sentence_idx >= len(sentences) or sentences[sentence_idx] is None:
            missing_sentence += 1
            continue
        sentence_data = sentences[sentence_idx]
        words = sentence_data.get("word")
        if words is None:
            missing_word += 1
            continue
        sample_word = _normalize_token(sample["word"])
        processed_word_idx = _resolve_processed_word_index(
            sentence_data=sentence_data,
            raw_word_idx=word_idx,
            sample_word=sample_word,
        )
        if processed_word_idx is None:
            word_mismatch += 1
            continue
        if processed_word_idx >= len(words) or words[processed_word_idx] is None:
            missing_word += 1
            continue
        word_data = words[processed_word_idx]

        processed_eeg = _extract_processed_eeg(
            word_data=word_data,
            eeg_types=eeg_types,
            include_n_fixations=include_n_fixations,
        )
        if processed_eeg is None:
            missing_processed += 1
            continue

        hybrid_sample = dict(sample)
        hybrid_sample["processed_eeg"] = processed_eeg
        hybrid_samples.append(hybrid_sample)

    summary = {
        "source_cache": str(input_file),
        "hybrid_sample_count": len(hybrid_samples),
        "raw_sample_count": len(payload["samples"]),
        "missing_subject": missing_subject,
        "missing_sentence": missing_sentence,
        "missing_word": missing_word,
        "missing_processed": missing_processed,
        "word_mismatch": word_mismatch,
        "processed_eeg_types": list(eeg_types),
        "include_n_fixations": include_n_fixations,
        "processed_feature_dim": int(hybrid_samples[0]["processed_eeg"].shape[0]) if hybrid_samples else 0,
    }

    output_payload = {
        "metadata": {
            **payload["metadata"],
            "input_cache": str(input_file),
            "sample_granularity": "word",
            "cache_type": "hybrid_processed_raw",
            "processed_eeg_types": list(eeg_types),
            "include_n_fixations": include_n_fixations,
        },
        "summary": summary,
        "samples": hybrid_samples,
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "wb") as handle:
        pickle.dump(output_payload, handle, protocol=pickle.HIGHEST_PROTOCOL)

    summary_path = output_file.with_suffix(output_file.suffix + ".summary.txt")
    with open(summary_path, "w", encoding="utf-8") as handle:
        for key, value in summary.items():
            handle.write(f"{key}: {value}\n")

    return output_payload


def main():
    args = parse_args()
    print("=" * 80)
    print("BUILDING HYBRID WORD CACHE")
    print("=" * 80)
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Processed EEG types: {args.eeg_types}")
    print(f"Include nFixations: {args.include_n_fixations}")

    payload = build_hybrid_cache(
        input_path=args.input,
        output_path=args.output,
        dataset_root=args.dataset_root,
        eeg_types=args.eeg_types,
        include_n_fixations=args.include_n_fixations,
    )

    print("\n" + "=" * 80)
    print("BUILD COMPLETE")
    print("=" * 80)
    print(payload["summary"])
    print(f"\nSaved cache to: {Path(args.output)}")


if __name__ == "__main__":
    main()
