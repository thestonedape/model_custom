"""
Data Splitting for BELT
Creates 80/10/10 train/dev/test splits by sentence index
"""

import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import random


def create_splits(
    dataset_root: str,
    tasks: List[str],
    train_ratio: float = 0.8,
    dev_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_seed: int = 42,
    save_path: str = None
) -> Dict[str, List[int]]:
    """
    Create train/dev/test splits for ZuCo dataset
    
    Splits by pickle file indices (cross-sentence setting as in BELT paper)
    
    Args:
        dataset_root: Root directory of ZuCo dataset
        tasks: List of task names to include
        train_ratio: Fraction for training (default: 0.8)
        dev_ratio: Fraction for development (default: 0.1)
        test_ratio: Fraction for testing (default: 0.1)
        random_seed: Random seed for reproducibility
        save_path: Path to save splits (optional)
        
    Returns:
        Dictionary with 'train', 'dev', 'test' keys containing file indices
    """
    assert abs(train_ratio + dev_ratio + test_ratio - 1.0) < 1e-6, \
        "Split ratios must sum to 1.0"
    
    # Set random seed
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    dataset_root = Path(dataset_root)
    
    # Collect all pickle files
    all_pickle_files = []
    for task in tasks:
        pickle_dir = dataset_root / task / "pickle"
        if pickle_dir.exists():
            pickle_files = sorted(list(pickle_dir.glob("*.pickle")))
            all_pickle_files.extend(pickle_files)
    
    num_files = len(all_pickle_files)
    print(f"Total pickle files found: {num_files}")
    
    # Create shuffled indices
    all_indices = list(range(num_files))
    random.shuffle(all_indices)
    
    # Split indices with improved logic for small datasets
    # Ensure each split gets at least 1 file when we have 3+ files
    if num_files >= 10:
        # For larger datasets, use exact ratios
        train_end = int(num_files * train_ratio)
        dev_end = train_end + int(num_files * dev_ratio)
    elif num_files >= 3:
        # For small datasets (3-9 files), ensure minimum 1 per split
        # Allocate remaining after reserving 1 for dev and test
        reserved = 2  # 1 for dev, 1 for test
        train_end = num_files - reserved
        dev_end = train_end + 1  # 1 file for dev
        # Remaining goes to test (at least 1 guaranteed)
    else:
        # Very small datasets (1-2 files): best effort
        train_end = int(num_files * train_ratio)
        dev_end = train_end + int(num_files * dev_ratio)
    
    train_indices = sorted(all_indices[:train_end])
    dev_indices = sorted(all_indices[train_end:dev_end])
    test_indices = sorted(all_indices[dev_end:])
    
    splits = {
        'train': train_indices,
        'dev': dev_indices,
        'test': test_indices,
        'metadata': {
            'total_files': num_files,
            'train_ratio': train_ratio,
            'dev_ratio': dev_ratio,
            'test_ratio': test_ratio,
            'random_seed': random_seed,
            'tasks': tasks
        }
    }
    
    print(f"\nSplit sizes:")
    print(f"  Train: {len(train_indices)} files ({len(train_indices)/num_files:.1%})")
    print(f"  Dev: {len(dev_indices)} files ({len(dev_indices)/num_files:.1%})")
    print(f"  Test: {len(test_indices)} files ({len(test_indices)/num_files:.1%})")
    
    # Save splits
    if save_path:
        with open(save_path, 'wb') as f:
            pickle.dump(splits, f)
        print(f"\nSplits saved to: {save_path}")
    
    return splits


def load_splits(load_path: str) -> Dict[str, List[int]]:
    """
    Load splits from file
    
    Args:
        load_path: Path to saved splits file
        
    Returns:
        Dictionary with 'train', 'dev', 'test' keys
    """
    with open(load_path, 'rb') as f:
        splits = pickle.load(f)
    
    print(f"Loaded splits from: {load_path}")
    print(f"  Train: {len(splits['train'])} files")
    print(f"  Dev: {len(splits['dev'])} files")
    print(f"  Test: {len(splits['test'])} files")
    
    if 'metadata' in splits:
        print(f"\nMetadata:")
        for key, value in splits['metadata'].items():
            print(f"  {key}: {value}")
    
    return splits


def count_words_per_split(
    dataset_root: str,
    tasks: List[str],
    splits: Dict[str, List[int]],
    vocabulary
) -> Dict[str, Dict]:
    """
    Count word instances in each split
    
    Args:
        dataset_root: Root directory of ZuCo dataset
        tasks: List of task names
        splits: Dictionary with split indices
        vocabulary: Vocabulary object
        
    Returns:
        Dictionary with statistics per split
    """
    dataset_root = Path(dataset_root)
    
    # Collect all pickle files
    all_pickle_files = []
    for task in tasks:
        pickle_dir = dataset_root / task / "pickle"
        if pickle_dir.exists():
            all_pickle_files.extend(sorted(list(pickle_dir.glob("*.pickle"))))
    
    all_pickle_files = [str(p) for p in all_pickle_files]
    
    # Count words per split
    split_stats = {}
    
    for split_name in ['train', 'dev', 'test']:
        split_indices = splits[split_name]
        split_files = [all_pickle_files[i] for i in split_indices]
        
        word_count = 0
        in_vocab_count = 0
        
        for pickle_file in split_files:
            with open(pickle_file, 'rb') as f:
                data = pickle.load(f)
            
            # Data structure: {subject_id: [sentences]}
            for subject_id, sentences in data.items():
                for sentence_data in sentences:
                    if sentence_data is None or 'word' not in sentence_data:
                        continue
                    for word_data in sentence_data['word']:
                        if word_data is None or 'content' not in word_data:
                            continue
                        word = word_data['content'].lower()
                        word_count += 1
                        
                        if vocabulary.is_in_vocabulary(word):
                            in_vocab_count += 1
        
        split_stats[split_name] = {
            'num_files': len(split_files),
            'total_words': word_count,
            'in_vocab_words': in_vocab_count,
            'coverage': in_vocab_count / word_count if word_count > 0 else 0
        }
    
    return split_stats


if __name__ == "__main__":
    """Test split creation"""
    
    print("="*80)
    print("CREATING BELT DATA SPLITS (80/10/10)")
    print("="*80)
    
    # Create splits
    splits = create_splits(
        dataset_root="dataset/ZuCo",
        tasks=["task1-SR", "task2-NR", "task3-TSR"],
        train_ratio=0.8,
        dev_ratio=0.1,
        test_ratio=0.1,
        random_seed=42,
        save_path="model_custom/data/splits.pkl"
    )
    
    # Count word instances per split
    print("\n" + "="*80)
    print("WORD STATISTICS PER SPLIT")
    print("="*80)
    
    from vocabulary import Vocabulary
    
    vocab = Vocabulary(vocab_size=500)
    vocab.load("model_custom/data/vocabulary_top500.pkl")
    
    stats = count_words_per_split(
        dataset_root="dataset/ZuCo",
        tasks=["task1-SR", "task2-NR", "task3-TSR"],
        splits=splits,
        vocabulary=vocab
    )
    
    for split_name, split_stats in stats.items():
        print(f"\n{split_name.upper()} Split:")
        for key, value in split_stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.2%}")
            else:
                print(f"  {key}: {value}")
    
    # Test loading
    print("\n" + "="*80)
    print("TESTING SPLIT LOADING")
    print("="*80)
    
    loaded_splits = load_splits("model_custom/data/splits.pkl")
    print(f"\nVerification: splits match = {splits == loaded_splits}")
