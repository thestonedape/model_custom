"""
Audit the local BELT/ZuCo preprocessing assumptions used by this repo.

This prints:
- available task versions and sentence/word counts
- current vocabulary/split metadata
- train/dev/test sentence-text overlap for sentence-level splits
- the structure of the already-processed local pickle files
"""

import argparse
import pickle
from pathlib import Path


def load_pickle(path: Path):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"Failed to load {path}. This audit should be run from the project venv "
            f"that matches the serialized pickle dependencies. Original error: {exc}"
        ) from exc


def summarize_task(task_path: Path):
    dataset_path = task_path / "pickle" / f"{task_path.name}-dataset.pickle"
    if not dataset_path.exists():
        return None

    data = load_pickle(dataset_path)
    valid_sentences = 0
    total_words = 0
    for sentences in data.values():
        if not sentences:
            continue
        for sentence in sentences:
            if sentence is None or "word" not in sentence:
                continue
            valid_sentences += 1
            total_words += sum(1 for word in sentence["word"] if word is not None)

    return {
        "task": task_path.name,
        "subjects": len(data),
        "valid_sentences": valid_sentences,
        "total_words": total_words,
        "dataset_path": str(dataset_path),
    }


def sentence_text_overlap(splits_path: Path):
    splits = load_pickle(splits_path)
    cache = {}

    def collect(split_items):
        texts = []
        for file_path, subject_id, sent_idx in split_items:
            if file_path not in cache:
                cache[file_path] = load_pickle(Path(file_path))
            data = cache[file_path]
            if isinstance(data, dict):
                if subject_id not in data or data[subject_id] is None:
                    continue
                sentence = data[subject_id][sent_idx]
            else:
                sentence = data[sent_idx]
            if sentence is None or "content" not in sentence:
                continue
            texts.append(sentence["content"].strip().lower())
        return texts

    train_texts = collect(splits["train"])
    dev_texts = collect(splits["dev"])
    test_texts = collect(splits["test"])

    train_set = set(train_texts)
    dev_set = set(dev_texts)
    test_set = set(test_texts)

    return {
        "metadata": splits.get("metadata", {}),
        "entries": {
            "train": len(train_texts),
            "dev": len(dev_texts),
            "test": len(test_texts),
        },
        "unique_texts": {
            "train": len(train_set),
            "dev": len(dev_set),
            "test": len(test_set),
        },
        "overlap": {
            "train_dev": len(train_set & dev_set),
            "train_test": len(train_set & test_set),
            "dev_test": len(dev_set & test_set),
            "all_three": len(train_set & dev_set & test_set),
        },
    }


def inspect_processed_pickle(example_path: Path):
    data = load_pickle(example_path)
    subject_id = next(iter(data))
    sentence = next(s for s in data[subject_id] if s and "word" in s and s["word"])
    word = next(w for w in sentence["word"] if w)
    return {
        "sentence_keys": sorted(sentence.keys()),
        "word_keys": sorted(word.keys()),
        "word_level_EEG_keys": sorted(word["word_level_EEG"].keys()),
    }


def main():
    parser = argparse.ArgumentParser(description="Audit BELT data assumptions in the local repo")
    parser.add_argument("--dataset-root", default="dataset/ZuCo")
    parser.add_argument("--splits-path", default="data/sentence_splits.pkl")
    parser.add_argument("--vocab-path", default="data/vocabulary_top500.pkl")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    print("=" * 80)
    print("LOCAL BELT DATA AUDIT")
    print("=" * 80)

    print("\nAvailable task versions")
    for task_dir in sorted(p for p in dataset_root.iterdir() if p.is_dir()):
        summary = summarize_task(task_dir)
        if summary is None:
            continue
        print(summary)

    vocab_path = Path(args.vocab_path)
    if vocab_path.exists():
        vocab = load_pickle(vocab_path)
        print("\nVocabulary metadata")
        print(vocab.get("metadata", {}))
        print({"vocab_size": vocab.get("vocab_size"), "stored_words": len(vocab.get("word2idx", {}))})

    splits_path = Path(args.splits_path)
    if splits_path.exists():
        print("\nSentence split overlap audit")
        print(sentence_text_overlap(splits_path))

    example_pickle = dataset_root / "task2-NR" / "pickle" / "task2-NR-dataset.pickle"
    if example_pickle.exists():
        print("\nProcessed pickle structure")
        print(inspect_processed_pickle(example_pickle))


if __name__ == "__main__":
    main()
