"""
Sentence-Level Splitting for BELT-style experiments.

Supports:
- sentence instance splits: shuffle each (file, subject, sentence_idx) entry
- unique sentence text splits: keep identical sentence texts in the same split
"""

import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import random


SentenceRef = Tuple[str, int, int]


def _normalize_sentence_text(sentence_data) -> str:
    """Normalize sentence content for unique-text grouping."""
    content = sentence_data.get('content', '') if sentence_data else ''
    return " ".join(str(content).strip().lower().split())


def _load_sentence_refs(dataset_root: Path, tasks: List[str]) -> Tuple[List[SentenceRef], Dict[str, List[SentenceRef]]]:
    """
    Load sentence references and group them by normalized sentence text.

    Returns:
        all_sentences: flat list of sentence references
        sentence_groups: mapping from normalized sentence text to all matching refs
    """
    print("Loading all sentences from all tasks...")
    all_sentences: List[SentenceRef] = []
    sentence_groups: Dict[str, List[SentenceRef]] = {}

    for task in tasks:
        pickle_dir = dataset_root / task / "pickle"
        if not pickle_dir.exists():
            continue

        pickle_files = sorted(list(pickle_dir.glob("*.pickle")))

        for pickle_file in pickle_files:
            print(f"  Processing: {pickle_file}")

            with open(pickle_file, 'rb') as f:
                data = pickle.load(f)

            if isinstance(data, dict):
                for subject_id, sentences in data.items():
                    if sentences is None:
                        continue
                    for sent_idx, sentence_data in enumerate(sentences):
                        if sentence_data is None:
                            continue
                        ref = (str(pickle_file), subject_id, sent_idx)
                        all_sentences.append(ref)
                        sentence_groups.setdefault(_normalize_sentence_text(sentence_data), []).append(ref)
            else:
                for sent_idx, sentence_data in enumerate(data):
                    if sentence_data is None:
                        continue
                    ref = (str(pickle_file), None, sent_idx)
                    all_sentences.append(ref)
                    sentence_groups.setdefault(_normalize_sentence_text(sentence_data), []).append(ref)

    return all_sentences, sentence_groups


def _split_sentence_instances(
    all_sentences: List[SentenceRef],
    train_ratio: float,
    dev_ratio: float
) -> Tuple[List[SentenceRef], List[SentenceRef], List[SentenceRef]]:
    """Split sentence references directly, allowing repeated sentence text across splits."""
    random.shuffle(all_sentences)
    total_sentences = len(all_sentences)
    train_end = int(total_sentences * train_ratio)
    dev_end = int(total_sentences * (train_ratio + dev_ratio))
    return (
        all_sentences[:train_end],
        all_sentences[train_end:dev_end],
        all_sentences[dev_end:]
    )


def _split_unique_sentence_texts(
    sentence_groups: Dict[str, List[SentenceRef]],
    train_ratio: float,
    dev_ratio: float
) -> Tuple[List[SentenceRef], List[SentenceRef], List[SentenceRef]]:
    """Split by unique normalized sentence text so duplicate texts stay in one split."""
    sentence_texts = list(sentence_groups.keys())
    random.shuffle(sentence_texts)

    total_texts = len(sentence_texts)
    train_end = int(total_texts * train_ratio)
    dev_end = int(total_texts * (train_ratio + dev_ratio))

    train_texts = sentence_texts[:train_end]
    dev_texts = sentence_texts[train_end:dev_end]
    test_texts = sentence_texts[dev_end:]

    def expand(texts: List[str]) -> List[SentenceRef]:
        refs: List[SentenceRef] = []
        for text in texts:
            refs.extend(sentence_groups[text])
        return refs

    return expand(train_texts), expand(dev_texts), expand(test_texts)


def create_sentence_splits(
    dataset_root: str,
    tasks: List[str],
    train_ratio: float = 0.8,
    dev_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_seed: int = 42,
    split_mode: str = "sentence_instance",
    save_path: str = None
) -> Dict[str, List[Tuple[str, int, int]]]:
    """
    Create train/dev/test splits at sentence level.
    
    Args:
        dataset_root: Root directory of ZuCo dataset
        tasks: List of task names to include
        train_ratio: Fraction for training (default: 0.8)
        dev_ratio: Fraction for development (default: 0.1)
        test_ratio: Fraction for testing (default: 0.1)
        random_seed: Random seed for reproducibility
        split_mode:
            - "sentence_instance": shuffle each sentence instance independently
            - "unique_sentence_text": keep identical sentence texts in the same split
        save_path: Path to save splits (optional)
        
    Returns:
        Dictionary with 'train', 'dev', 'test' keys containing lists of
        (pickle_file_path, subject_id, sentence_index) tuples
    """
    assert abs(train_ratio + dev_ratio + test_ratio - 1.0) < 1e-6, \
        "Split ratios must sum to 1.0"
    
    # Set random seed
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    dataset_root = Path(dataset_root)
    
    if split_mode not in {"sentence_instance", "unique_sentence_text"}:
        raise ValueError(f"Unsupported split_mode: {split_mode}")

    # Step 1: Collect all sentences with their locations
    all_sentences, sentence_groups = _load_sentence_refs(dataset_root=dataset_root, tasks=tasks)
    total_sentences = len(all_sentences)
    print(f"\nTotal sentences collected: {total_sentences:,}")

    print(f"Split mode: {split_mode}")
    if split_mode == "sentence_instance":
        train_sentences, dev_sentences, test_sentences = _split_sentence_instances(
            all_sentences=all_sentences,
            train_ratio=train_ratio,
            dev_ratio=dev_ratio
        )
        print(f"Sentence instances shuffled with seed {random_seed}")
    else:
        train_sentences, dev_sentences, test_sentences = _split_unique_sentence_texts(
            sentence_groups=sentence_groups,
            train_ratio=train_ratio,
            dev_ratio=dev_ratio
        )
        print(f"Unique sentence texts shuffled with seed {random_seed}")
        print(f"Total unique sentence texts: {len(sentence_groups):,}")
    
    # Step 4: Create splits dictionary
    splits = {
        'train': train_sentences,
        'dev': dev_sentences,
        'test': test_sentences,
        'metadata': {
            'total_sentences': total_sentences,
            'train_ratio': len(train_sentences) / total_sentences,
            'dev_ratio': len(dev_sentences) / total_sentences,
            'test_ratio': len(test_sentences) / total_sentences,
            'random_seed': random_seed,
            'tasks': tasks,
            'split_type': split_mode,
            'split_mode': split_mode,
            'unique_sentence_texts': len(sentence_groups)
        }
    }
    
    print(f"\nSentence-level splits created:")
    print(f"  Train: {len(train_sentences):,} sentences ({len(train_sentences)/total_sentences:.1%})")
    print(f"  Dev:   {len(dev_sentences):,} sentences ({len(dev_sentences)/total_sentences:.1%})")
    print(f"  Test:  {len(test_sentences):,} sentences ({len(test_sentences)/total_sentences:.1%})")
    
    # Step 5: Save if requested
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'wb') as f:
            pickle.dump(splits, f)
        
        print(f"\nSplits saved to: {save_path}")
    
    return splits


def analyze_sentence_splits(splits_path: str):
    """Analyze sentence-level splits"""
    with open(splits_path, 'rb') as f:
        splits = pickle.load(f)
    
    print("\n" + "="*80)
    print("SENTENCE-LEVEL SPLITS ANALYSIS")
    print("="*80)
    
    for split_name in ['train', 'dev', 'test']:
        sentences = splits[split_name]
        
        # Count unique files used in this split
        unique_files = set(file for file, _, _ in sentences)
        
        print(f"\n{split_name.upper()}:")
        print(f"  Total sentences: {len(sentences):,}")
        print(f"  Unique files: {len(unique_files)}")
        print(f"  Ratio: {splits['metadata'][f'{split_name}_ratio']:.1%}")
    
    print(f"\nMetadata:")
    print(f"  Total sentences: {splits['metadata']['total_sentences']:,}")
    print(f"  Random seed: {splits['metadata']['random_seed']}")
    print(f"  Split type: {splits['metadata'].get('split_type', 'unknown')}")
    print(f"  Unique sentence texts: {splits['metadata'].get('unique_sentence_texts', 'unknown')}")


if __name__ == "__main__":
    # Test the sentence-level splitting
    splits = create_sentence_splits(
        dataset_root="dataset/ZuCo",
        tasks=['task1-SR', 'task2-NR', 'task2-NR-2.0', 'task3-TSR', 'task3-TSR-2.0'],
        save_path="data/sentence_splits.pkl"
    )
    
    analyze_sentence_splits("data/sentence_splits.pkl")
