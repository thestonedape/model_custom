"""
Build a word-level raw cache by grouping fixation-level raw samples.

The current raw pipeline treats each fixation as a separate example, which is
often too noisy for word decoding. This script groups all fixations for the same
word instance and concatenates their EEG/ET windows in fixation order.
"""

from __future__ import annotations

import argparse
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Build word-level raw caches from fixation-level caches")
    parser.add_argument("--input", required=True, help="Input fixation-level cache pickle")
    parser.add_argument("--output", required=True, help="Output word-level cache pickle")
    return parser.parse_args()


def _group_key(sample: Dict[str, object]) -> Tuple[object, ...]:
    return (
        sample["task"],
        sample["version"],
        sample["subject_id"],
        int(sample["sentence_idx"]),
        int(sample["word_idx"]),
        sample["word"],
        sample["sentence_text"],
        sample.get("source_path"),
    )


def _aggregate_metrics(samples: List[Dict[str, object]]) -> Dict[str, float | None]:
    metric_names = set()
    for sample in samples:
        metric_names.update(sample["metrics"].keys())

    result: Dict[str, float | None] = {}
    for metric in sorted(metric_names):
        values = []
        for sample in samples:
            value = sample["metrics"].get(metric)
            if value is None:
                continue
            try:
                numeric = float(value)
            except Exception:
                continue
            if np.isfinite(numeric):
                values.append(numeric)
        result[metric] = float(np.mean(values)) if values else None
    return result


def _aggregate_mean_pupil(samples: List[Dict[str, object]]) -> float | None:
    values = []
    for sample in samples:
        value = sample.get("mean_pupil_size")
        if value is None:
            continue
        try:
            numeric = float(value)
        except Exception:
            continue
        if np.isfinite(numeric):
            values.append(numeric)
    return float(np.mean(values)) if values else None


def build_word_level_cache(input_path: str, output_path: str) -> Dict[str, object]:
    input_file = Path(input_path)
    output_file = Path(output_path)

    with open(input_file, "rb") as handle:
        payload = pickle.load(handle)

    grouped: Dict[Tuple[object, ...], List[Dict[str, object]]] = defaultdict(list)
    for sample in payload["samples"]:
        grouped[_group_key(sample)].append(sample)

    word_samples: List[Dict[str, object]] = []
    fixation_counts: List[int] = []
    eeg_lengths: List[int] = []
    et_lengths: List[int] = []

    for _, samples in grouped.items():
        samples = sorted(samples, key=lambda item: int(item["fixation_idx"]))
        first = samples[0]
        eeg = np.concatenate([np.asarray(sample["eeg"], dtype=np.float32) for sample in samples], axis=0)
        et = np.concatenate([np.asarray(sample["et"], dtype=np.float32) for sample in samples], axis=0)
        fixation_count = len(samples)

        word_sample = dict(first)
        word_sample["eeg"] = eeg
        word_sample["et"] = et
        word_sample["fixation_idx"] = 0
        word_sample["fixation_count"] = fixation_count
        word_sample["fixation_lengths"] = [int(np.asarray(sample["eeg"]).shape[0]) for sample in samples]
        word_sample["n_fixations"] = fixation_count
        word_sample["metrics"] = _aggregate_metrics(samples)
        word_sample["mean_pupil_size"] = _aggregate_mean_pupil(samples)

        word_samples.append(word_sample)
        fixation_counts.append(fixation_count)
        eeg_lengths.append(int(eeg.shape[0]))
        et_lengths.append(int(et.shape[0]))

    summary = {
        "source_cache": str(input_file),
        "grouping": "word_level_concatenated_fixations",
        "fixation_sample_count": len(payload["samples"]),
        "word_sample_count": len(word_samples),
        "fixation_count": {
            "min": min(fixation_counts) if fixation_counts else 0,
            "max": max(fixation_counts) if fixation_counts else 0,
            "mean": float(np.mean(fixation_counts)) if fixation_counts else 0.0,
        },
        "eeg_timesteps": {
            "min": min(eeg_lengths) if eeg_lengths else 0,
            "max": max(eeg_lengths) if eeg_lengths else 0,
            "mean": float(np.mean(eeg_lengths)) if eeg_lengths else 0.0,
        },
        "et_timesteps": {
            "min": min(et_lengths) if et_lengths else 0,
            "max": max(et_lengths) if et_lengths else 0,
            "mean": float(np.mean(et_lengths)) if et_lengths else 0.0,
        },
    }

    output_payload = {
        "metadata": {
            **payload["metadata"],
            "input_cache": str(input_file),
            "sample_granularity": "word",
            "grouping": "concatenated_fixations",
        },
        "summary": summary,
        "samples": word_samples,
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
    print("BUILDING WORD-LEVEL RAW CACHE")
    print("=" * 80)
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")

    payload = build_word_level_cache(input_path=args.input, output_path=args.output)

    print("\n" + "=" * 80)
    print("BUILD COMPLETE")
    print("=" * 80)
    print(payload["summary"])
    print(f"\nSaved cache to: {Path(args.output)}")


if __name__ == "__main__":
    main()
