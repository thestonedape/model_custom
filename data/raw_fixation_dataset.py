"""
Raw fixation-level ZuCo dataset utilities.

This module reads the original MATLAB files from ZuCo 1.0 and 2.0 and emits a
common multimodal sample schema built around per-fixation EEG and eye-tracking
windows. The goal is to preserve much more signal than the legacy GD/FFD/TRT
summary pipeline used by BELT.
"""

from __future__ import annotations

import pickle
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Generator, Iterable, List, Optional, Sequence

import h5py
import numpy as np
from scipy.io import loadmat


WORD_METRIC_KEYS = ("FFD", "GD", "TRT", "GPT", "SFD")
EXPECTED_EEG_CHANNELS = 105
EXPECTED_ET_CHANNELS = 4


def _parse_subject_id(mat_path: Path) -> str:
    match = re.match(r"results(?P<subject>[A-Za-z0-9]+)_.+", mat_path.stem)
    if match:
        return match.group("subject")
    return mat_path.stem


def _normalize_text(text: object) -> str:
    return " ".join(str(text).strip().split())


def _normalize_scalar(value: object) -> Optional[float]:
    if value is None:
        return None
    array = np.asarray(value)
    if array.size == 0:
        return None
    return float(array.reshape(-1)[0])


def _normalize_vector(value: object) -> List[float]:
    if value is None:
        return []
    array = np.asarray(value)
    if array.size == 0:
        return []
    return [float(x) for x in array.reshape(-1)]


def _normalize_eeg_window(window: np.ndarray) -> np.ndarray:
    array = np.asarray(window)
    if array.ndim != 2:
        raise ValueError(f"EEG window must be 2D, got shape {array.shape}")
    if array.shape[0] == EXPECTED_EEG_CHANNELS:
        array = array.T
    elif array.shape[1] != EXPECTED_EEG_CHANNELS:
        raise ValueError(f"EEG window must contain {EXPECTED_EEG_CHANNELS} channels, got {array.shape}")
    return array.astype(np.float32, copy=False)


def _normalize_et_window(window: np.ndarray) -> np.ndarray:
    array = np.asarray(window)
    if array.ndim != 2:
        raise ValueError(f"ET window must be 2D, got shape {array.shape}")
    if array.shape[0] == EXPECTED_ET_CHANNELS:
        array = array.T
    elif array.shape[1] != EXPECTED_ET_CHANNELS:
        raise ValueError(f"ET window must contain {EXPECTED_ET_CHANNELS} channels, got {array.shape}")
    return array.astype(np.float32, copy=False)


def _v1_cell_sequence(value: object, expected_channels: int) -> List[np.ndarray]:
    """
    Normalize v1 MATLAB cell/array storage into a list of 2D fixation windows.

    ZuCo 1.0 stores single-fixation words as plain 2D numeric arrays and
    multi-fixation words as object-cell arrays. We preserve both cases here.
    """
    if value is None:
        return []

    if isinstance(value, (list, tuple)):
        return [np.asarray(item) for item in value]

    array = np.asarray(value)
    if array.size == 0:
        return []

    if array.dtype == object:
        return [np.asarray(item) for item in array.reshape(-1)]

    if array.ndim == 2:
        return [array]

    if array.ndim == 3:
        if array.shape[0] == expected_channels:
            return [array[:, :, idx] for idx in range(array.shape[2])]
        if array.shape[1] == expected_channels:
            return [array[idx, :, :] for idx in range(array.shape[0])]
        if array.shape[2] == expected_channels:
            return [array[:, idx, :] for idx in range(array.shape[1])]

    return [array]


def _iter_mat_files(dataset_root: str, tasks: Sequence[str]) -> List[Path]:
    root = Path(dataset_root)
    mat_files: List[Path] = []
    for task in tasks:
        task_dir = root / task / "Matlab_files"
        if not task_dir.exists():
            continue
        mat_files.extend(sorted(task_dir.glob("*.mat")))
    return mat_files


@dataclass
class RawFixationBuildSummary:
    tasks: List[str]
    mat_files_seen: int = 0
    sentences_seen: int = 0
    words_seen: int = 0
    samples_emitted: int = 0
    skipped_missing_sentence: int = 0
    skipped_missing_word: int = 0
    skipped_missing_raw: int = 0
    skipped_shape: int = 0
    skipped_other: int = 0

    def to_dict(self) -> Dict[str, object]:
        return {
            "tasks": list(self.tasks),
            "mat_files_seen": self.mat_files_seen,
            "sentences_seen": self.sentences_seen,
            "words_seen": self.words_seen,
            "samples_emitted": self.samples_emitted,
            "skipped_missing_sentence": self.skipped_missing_sentence,
            "skipped_missing_word": self.skipped_missing_word,
            "skipped_missing_raw": self.skipped_missing_raw,
            "skipped_shape": self.skipped_shape,
            "skipped_other": self.skipped_other,
        }


def _v2_deref_text(file_handle: h5py.File, ref) -> str:
    chars = np.asarray(file_handle[ref][()]).reshape(-1)
    return "".join(chr(int(value)) for value in chars if int(value) != 0)


def _v2_deref_array(file_handle: h5py.File, ref) -> np.ndarray:
    return np.asarray(file_handle[ref][()])


def _v2_deref_scalar(file_handle: h5py.File, ref) -> Optional[float]:
    return _normalize_scalar(_v2_deref_array(file_handle, ref))


def _v2_deref_vector(file_handle: h5py.File, ref) -> List[float]:
    return _normalize_vector(_v2_deref_array(file_handle, ref))


def _v2_resolve_cell_windows(
    file_handle: h5py.File,
    ref,
    expected_channels: int,
) -> List[np.ndarray]:
    """
    Resolve a MATLAB cell array stored in a v7.3 MAT file into a list of arrays.

    Some ZuCo 2.0 rawEEG/rawET cells are stored as uint64 zeros, which appear to
    indicate missing fixation windows. Those are filtered out here.
    """
    cell = file_handle[ref]
    if isinstance(cell, h5py.Dataset) and cell.dtype != object:
        array = np.asarray(cell[()])
        if array.size == 0:
            return []
        flat = array.reshape(-1)
        try:
            flat_float = flat.astype(np.float64, copy=False)
            if np.all(~np.isfinite(flat_float)) or np.all(flat_float == 0.0):
                return []
        except Exception:
            pass
        if array.ndim == 2 and (array.shape[0] == expected_channels or array.shape[1] == expected_channels):
            return [array]
        raise ValueError(f"Unsupported non-cell raw window shape: {array.shape}")

    entries = np.asarray(cell[()]).reshape(-1)
    windows: List[np.ndarray] = []
    for entry in entries:
        if isinstance(entry, h5py.Reference):
            windows.append(_v2_deref_array(file_handle, entry))
            continue
        try:
            entry_value = float(entry)
            if not np.isfinite(entry_value) or entry_value == 0.0:
                continue
        except Exception:
            pass
        raise ValueError(f"Unsupported v2 cell entry type: {type(entry).__name__}")
    return windows


def _iter_v1_sentence_words(sentence) -> Iterable[object]:
    words = getattr(sentence, "word", None)
    if words is None:
        return []
    if isinstance(words, np.ndarray):
        return [word for word in words.reshape(-1) if word is not None]
    return [words]


def _extract_v1_metrics(word) -> Dict[str, Optional[float]]:
    return {metric.lower(): _normalize_scalar(getattr(word, metric, None)) for metric in WORD_METRIC_KEYS}


def _extract_v2_metrics(file_handle: h5py.File, word_group, word_idx: int) -> Dict[str, Optional[float]]:
    metrics: Dict[str, Optional[float]] = {}
    for metric in WORD_METRIC_KEYS:
        if metric not in word_group:
            metrics[metric.lower()] = None
            continue
        ref = word_group[metric][word_idx, 0]
        metrics[metric.lower()] = _v2_deref_scalar(file_handle, ref)
    return metrics


def iter_raw_fixation_samples(
    dataset_root: str,
    tasks: Sequence[str],
    include_sentence_raw: bool = False,
    file_start: int = 0,
    max_files: Optional[int] = None,
    max_sentences: Optional[int] = None,
    max_samples: Optional[int] = None,
    summary: Optional[RawFixationBuildSummary] = None,
) -> Generator[Dict[str, object], None, RawFixationBuildSummary]:
    """
    Iterate over fixation-level multimodal samples from raw ZuCo MATLAB files.

    Sample schema:
        {
            "task": str,
            "version": "1.0" | "2.0",
            "subject_id": str,
            "sentence_idx": int,
            "word_idx": int,
            "fixation_idx": int,
            "sentence_text": str,
            "word": str,
            "n_fixations": int,
            "fixation_positions": List[float],
            "mean_pupil_size": Optional[float],
            "metrics": {"ffd": ..., "gd": ..., "trt": ..., "gpt": ..., "sfd": ...},
            "eeg": np.ndarray[T, 105],
            "et": np.ndarray[T, 4],
            "sentence_raw_eeg": Optional[np.ndarray[T, 105]],
        }
    """
    if summary is None:
        summary = RawFixationBuildSummary(tasks=list(tasks))

    mat_files = _iter_mat_files(dataset_root=dataset_root, tasks=tasks)
    if file_start:
        mat_files = mat_files[file_start:]
    if max_files is not None:
        mat_files = mat_files[:max_files]

    sentences_processed = 0
    emitted = 0

    for mat_path in mat_files:
        summary.mat_files_seen += 1
        task_name = mat_path.parent.parent.name
        version = "2.0" if task_name.endswith("-2.0") else "1.0"
        subject_id = _parse_subject_id(mat_path)

        if version == "1.0":
            mat = loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)
            sentences = mat["sentenceData"]
            if not isinstance(sentences, np.ndarray):
                sentences = np.asarray([sentences], dtype=object)

            for sentence_idx, sentence in enumerate(sentences.reshape(-1)):
                if sentence is None:
                    continue
                summary.sentences_seen += 1
                sentences_processed += 1
                if max_sentences is not None and sentences_processed > max_sentences:
                    return summary

                sentence_text = _normalize_text(getattr(sentence, "content", ""))
                sentence_raw = None
                if include_sentence_raw:
                    try:
                        sentence_raw = _normalize_eeg_window(getattr(sentence, "rawData", None))
                    except Exception:
                        sentence_raw = None

                for word_idx, word in enumerate(_iter_v1_sentence_words(sentence)):
                    summary.words_seen += 1
                    word_text = _normalize_text(getattr(word, "content", ""))
                    if not word_text:
                        summary.skipped_missing_word += 1
                        continue

                    raw_eeg_cells = getattr(word, "rawEEG", None)
                    raw_et_cells = getattr(word, "rawET", None)
                    if raw_eeg_cells is None or raw_et_cells is None:
                        summary.skipped_missing_raw += 1
                        continue

                    raw_eeg_cells = _v1_cell_sequence(raw_eeg_cells, expected_channels=EXPECTED_EEG_CHANNELS)
                    raw_et_cells = _v1_cell_sequence(raw_et_cells, expected_channels=EXPECTED_ET_CHANNELS)
                    if len(raw_eeg_cells) == 0 or len(raw_et_cells) == 0:
                        summary.skipped_missing_raw += 1
                        continue

                    n_fixations = int(_normalize_scalar(getattr(word, "nFixations", len(raw_eeg_cells))) or 0)
                    fixation_positions = _normalize_vector(getattr(word, "fixPositions", None))
                    mean_pupil_size = _normalize_scalar(getattr(word, "meanPupilSize", None))
                    metrics = _extract_v1_metrics(word)

                    for fixation_idx, (raw_eeg, raw_et) in enumerate(zip(raw_eeg_cells, raw_et_cells)):
                        try:
                            eeg = _normalize_eeg_window(raw_eeg)
                            et = _normalize_et_window(raw_et)
                        except ValueError:
                            summary.skipped_shape += 1
                            continue
                        except Exception:
                            summary.skipped_other += 1
                            continue

                        sample = {
                            "source_path": str(mat_path),
                            "task": task_name,
                            "version": version,
                            "subject_id": subject_id,
                            "sentence_idx": int(sentence_idx),
                            "word_idx": int(word_idx),
                            "fixation_idx": int(fixation_idx),
                            "sentence_text": sentence_text,
                            "word": word_text,
                            "n_fixations": int(n_fixations),
                            "fixation_positions": fixation_positions,
                            "mean_pupil_size": mean_pupil_size,
                            "metrics": metrics,
                            "eeg": eeg,
                            "et": et,
                            "sentence_raw_eeg": sentence_raw,
                        }
                        summary.samples_emitted += 1
                        emitted += 1
                        yield sample
                        if max_samples is not None and emitted >= max_samples:
                            return summary

        else:
            with h5py.File(mat_path, "r") as file_handle:
                sentence_group = file_handle["sentenceData"]
                total_sentences = sentence_group["content"].shape[0]

                for sentence_idx in range(total_sentences):
                    summary.sentences_seen += 1
                    sentences_processed += 1
                    if max_sentences is not None and sentences_processed > max_sentences:
                        return summary

                    sentence_text = _normalize_text(
                        _v2_deref_text(file_handle, sentence_group["content"][sentence_idx, 0])
                    )
                    sentence_raw = None
                    if include_sentence_raw:
                        try:
                            sentence_raw = _normalize_eeg_window(
                                _v2_deref_array(file_handle, sentence_group["rawData"][sentence_idx, 0])
                            )
                        except Exception:
                            sentence_raw = None

                    word_group = file_handle[sentence_group["word"][sentence_idx, 0]]
                    if not isinstance(word_group, h5py.Group):
                        summary.skipped_missing_sentence += 1
                        continue
                    if "content" not in word_group or "rawEEG" not in word_group or "rawET" not in word_group:
                        summary.skipped_missing_sentence += 1
                        continue
                    total_words = word_group["content"].shape[0]

                    for word_idx in range(total_words):
                        summary.words_seen += 1
                        word_text = _normalize_text(_v2_deref_text(file_handle, word_group["content"][word_idx, 0]))
                        if not word_text:
                            summary.skipped_missing_word += 1
                            continue

                        raw_eeg_cells = _v2_resolve_cell_windows(
                            file_handle,
                            word_group["rawEEG"][word_idx, 0],
                            expected_channels=EXPECTED_EEG_CHANNELS,
                        )
                        raw_et_cells = _v2_resolve_cell_windows(
                            file_handle,
                            word_group["rawET"][word_idx, 0],
                            expected_channels=EXPECTED_ET_CHANNELS,
                        )
                        if len(raw_eeg_cells) == 0 or len(raw_et_cells) == 0:
                            summary.skipped_missing_raw += 1
                            continue

                        n_fixations = int(_v2_deref_scalar(file_handle, word_group["nFixations"][word_idx, 0]) or 0)
                        fixation_positions = _v2_deref_vector(file_handle, word_group["fixPositions"][word_idx, 0])
                        mean_pupil_size = _v2_deref_scalar(file_handle, word_group["meanPupilSize"][word_idx, 0])
                        metrics = _extract_v2_metrics(file_handle, word_group, word_idx)

                        total_fixations = min(len(raw_eeg_cells), len(raw_et_cells))
                        for fixation_idx in range(total_fixations):
                            try:
                                eeg = _normalize_eeg_window(raw_eeg_cells[fixation_idx])
                                et = _normalize_et_window(raw_et_cells[fixation_idx])
                            except ValueError:
                                summary.skipped_shape += 1
                                continue
                            except Exception:
                                summary.skipped_other += 1
                                continue

                            sample = {
                                "source_path": str(mat_path),
                                "task": task_name,
                                "version": version,
                                "subject_id": subject_id,
                                "sentence_idx": int(sentence_idx),
                                "word_idx": int(word_idx),
                                "fixation_idx": int(fixation_idx),
                                "sentence_text": sentence_text,
                                "word": word_text,
                                "n_fixations": int(n_fixations),
                                "fixation_positions": fixation_positions,
                                "mean_pupil_size": mean_pupil_size,
                                "metrics": metrics,
                                "eeg": eeg,
                                "et": et,
                                "sentence_raw_eeg": sentence_raw,
                            }
                            summary.samples_emitted += 1
                            emitted += 1
                            yield sample
                            if max_samples is not None and emitted >= max_samples:
                                return summary

    return summary


def summarize_fixation_samples(samples: Sequence[Dict[str, object]], summary: RawFixationBuildSummary) -> Dict[str, object]:
    eeg_lengths = [int(sample["eeg"].shape[0]) for sample in samples]
    et_lengths = [int(sample["et"].shape[0]) for sample in samples]
    tasks: Dict[str, int] = {}
    versions: Dict[str, int] = {}

    for sample in samples:
        tasks[sample["task"]] = tasks.get(sample["task"], 0) + 1
        versions[sample["version"]] = versions.get(sample["version"], 0) + 1

    return {
        "build": summary.to_dict(),
        "sample_count": len(samples),
        "tasks": tasks,
        "versions": versions,
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


def build_fixation_cache(
    dataset_root: str,
    tasks: Sequence[str],
    output_path: str,
    include_sentence_raw: bool = False,
    file_start: int = 0,
    max_files: Optional[int] = None,
    max_sentences: Optional[int] = None,
    max_samples: Optional[int] = None,
) -> Dict[str, object]:
    summary = RawFixationBuildSummary(tasks=list(tasks))
    samples = list(
        iter_raw_fixation_samples(
            dataset_root=dataset_root,
            tasks=tasks,
            include_sentence_raw=include_sentence_raw,
            file_start=file_start,
            max_files=max_files,
            max_sentences=max_sentences,
            max_samples=max_samples,
            summary=summary,
        )
    )
    payload = {
        "metadata": {
            "dataset_root": dataset_root,
            "tasks": list(tasks),
            "include_sentence_raw": bool(include_sentence_raw),
            "file_start": file_start,
            "max_files": max_files,
            "max_sentences": max_sentences,
            "max_samples": max_samples,
        },
        "summary": summarize_fixation_samples(samples, summary),
        "samples": samples,
    }

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)

    summary_path = output.with_suffix(output.suffix + ".summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "metadata": payload["metadata"],
                "summary": payload["summary"],
            },
            handle,
            indent=2,
        )
    return payload
