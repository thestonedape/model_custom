"""
PyTorch dataset utilities for hybrid processed+raw word-level caches.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .raw_fixation_torch_dataset import METRIC_ORDER, _pad_sequence_list
from .vocabulary import Vocabulary


class HybridWordCacheDataset(Dataset):
    def __init__(
        self,
        cache_path: Optional[str] = None,
        cache_paths: Optional[Sequence[str]] = None,
        vocab_path: Optional[str] = None,
        splits_path: Optional[str] = None,
        split_name: Optional[str] = None,
        tasks: Optional[Sequence[str]] = None,
        normalize_eeg: bool = False,
        normalize_et: bool = False,
        normalization_eps: float = 1e-5,
    ):
        resolved_paths: List[Path] = []
        if cache_paths is not None:
            resolved_paths.extend(Path(path) for path in cache_paths)
        if cache_path is not None:
            resolved_paths.append(Path(cache_path))
        if not resolved_paths:
            raise ValueError("Provide cache_path or cache_paths")

        self.cache_paths = resolved_paths
        self.cache_path = resolved_paths[0]
        self.normalize_eeg = normalize_eeg
        self.normalize_et = normalize_et
        self.normalization_eps = normalization_eps
        self.metadata: List[Dict[str, object]] = []
        self.summary: List[Dict[str, object]] = []
        samples: List[Dict[str, object]] = []

        for path in resolved_paths:
            with open(path, "rb") as handle:
                payload = pickle.load(handle)
            self.metadata.append(payload["metadata"])
            self.summary.append(payload["summary"])
            samples.extend(payload["samples"])

        if tasks is not None:
            task_set = set(tasks)
            samples = [sample for sample in samples if sample["task"] in task_set]

        self.vocabulary: Optional[Vocabulary] = None
        if vocab_path is not None:
            self.vocabulary = Vocabulary(vocab_size=500)
            self.vocabulary.load(vocab_path)
            samples = [sample for sample in samples if self.vocabulary.is_in_vocabulary(sample["word"])]

        if splits_path is not None and split_name is not None:
            allowed_refs = self._load_allowed_refs(splits_path=splits_path, split_name=split_name)
            samples = [
                sample for sample in samples
                if (sample["task"], sample["subject_id"], int(sample["sentence_idx"])) in allowed_refs
            ]

        self.samples = samples

    @staticmethod
    def _load_allowed_refs(splits_path: str, split_name: str) -> set[tuple[str, str, int]]:
        with open(splits_path, "rb") as handle:
            splits = pickle.load(handle)
        refs = set()
        for file_path, subject_id, sentence_idx in splits[split_name]:
            task = Path(file_path).parent.parent.name
            refs.add((task, str(subject_id), int(sentence_idx)))
        return refs

    def __len__(self) -> int:
        return len(self.samples)

    def _normalize_sequence(self, array: np.ndarray) -> np.ndarray:
        array = np.asarray(array, dtype=np.float32)
        mean = array.mean(axis=0, keepdims=True)
        std = array.std(axis=0, keepdims=True)
        std = np.maximum(std, self.normalization_eps)
        return (array - mean) / std

    def __getitem__(self, idx: int) -> Dict[str, object]:
        sample = self.samples[idx]
        eeg = np.asarray(sample["eeg"], dtype=np.float32)
        et = np.asarray(sample["et"], dtype=np.float32)
        if self.normalize_eeg:
            eeg = self._normalize_sequence(eeg)
        if self.normalize_et:
            et = self._normalize_sequence(et)
        metrics = np.asarray(
            [sample["metrics"].get(metric) if sample["metrics"].get(metric) is not None else 0.0 for metric in METRIC_ORDER],
            dtype=np.float32,
        )
        label = None
        if self.vocabulary is not None:
            label = int(self.vocabulary.get_word_index(sample["word"]))
        return {
            "processed_eeg": np.asarray(sample["processed_eeg"], dtype=np.float32),
            "eeg": eeg,
            "et": et,
            "sentence_raw_eeg": sample["sentence_raw_eeg"],
            "word": sample["word"],
            "label": label,
            "sentence_text": sample["sentence_text"],
            "task": sample["task"],
            "version": sample["version"],
            "subject_id": sample["subject_id"],
            "source_path": sample.get("source_path"),
            "sentence_idx": int(sample["sentence_idx"]),
            "word_idx": int(sample["word_idx"]),
            "fixation_idx": int(sample["fixation_idx"]),
            "n_fixations": int(sample["n_fixations"]),
            "fixation_positions": np.asarray(sample["fixation_positions"], dtype=np.float32),
            "mean_pupil_size": float(sample["mean_pupil_size"]) if sample["mean_pupil_size"] is not None else 0.0,
            "metrics": metrics,
        }


def collate_hybrid_word_batch(batch: List[Dict[str, object]]) -> Dict[str, object]:
    eeg_sequences = [item["eeg"] for item in batch]
    et_sequences = [item["et"] for item in batch]
    sentence_raw_sequences = [item["sentence_raw_eeg"] for item in batch if item["sentence_raw_eeg"] is not None]

    eeg, eeg_mask = _pad_sequence_list(eeg_sequences, feature_dim=105)
    et, et_mask = _pad_sequence_list(et_sequences, feature_dim=4)

    sentence_raw = None
    sentence_raw_mask = None
    if len(sentence_raw_sequences) == len(batch):
        sentence_raw, sentence_raw_mask = _pad_sequence_list(sentence_raw_sequences, feature_dim=105)

    processed_eeg = torch.tensor(np.stack([item["processed_eeg"] for item in batch]), dtype=torch.float32)
    metrics = torch.tensor(np.stack([item["metrics"] for item in batch]), dtype=torch.float32)
    mean_pupil_size = torch.tensor([item["mean_pupil_size"] for item in batch], dtype=torch.float32)
    fixation_counts = torch.tensor([item["n_fixations"] for item in batch], dtype=torch.long)
    lengths = torch.tensor([item["eeg"].shape[0] for item in batch], dtype=torch.long)
    labels = None
    if batch[0]["label"] is not None:
        labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)

    return {
        "processed_eeg": processed_eeg,
        "eeg": eeg,
        "eeg_mask": eeg_mask,
        "et": et,
        "et_mask": et_mask,
        "sentence_raw_eeg": sentence_raw,
        "sentence_raw_mask": sentence_raw_mask,
        "labels": labels,
        "metrics": metrics,
        "mean_pupil_size": mean_pupil_size,
        "n_fixations": fixation_counts,
        "lengths": lengths,
        "word": [item["word"] for item in batch],
        "sentence_text": [item["sentence_text"] for item in batch],
        "task": [item["task"] for item in batch],
        "version": [item["version"] for item in batch],
        "subject_id": [item["subject_id"] for item in batch],
        "source_path": [item["source_path"] for item in batch],
        "sentence_idx": torch.tensor([item["sentence_idx"] for item in batch], dtype=torch.long),
        "word_idx": torch.tensor([item["word_idx"] for item in batch], dtype=torch.long),
        "fixation_idx": torch.tensor([item["fixation_idx"] for item in batch], dtype=torch.long),
        "fixation_positions": [item["fixation_positions"] for item in batch],
    }
