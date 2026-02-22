"""
Sentence-Level Splitting for BELT
Creates 80/10/10 train/dev/test splits at sentence level (not file level)
"""

import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import random


def create_sentence_splits(
    dataset_root: str,
    tasks: List[str],
    train_ratio: float = 0.8,
    dev_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_seed: int = 42,
    save_path: str = None
) -> Dict[str, List[Tuple[str, int, int]]]:
    """
    Create train/dev/test splits at SENTENCE level (not file level)
    
    This properly implements 80/10/10 splits as in BELT paper by:
    1. Loading ALL sentences from ALL files
    2. Shuffling sentences
    3. Splitting shuffled sentences 80/10/10
    
    Args:
        dataset_root: Root directory of ZuCo dataset
        tasks: List of task names to include
        train_ratio: Fraction for training (default: 0.8)
        dev_ratio: Fraction for development (default: 0.1)
        test_ratio: Fraction for testing (default: 0.1)
        random_seed: Random seed for reproducibility
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
    
    # Step 1: Collect all sentences with their locations
    print("Loading all sentences from all tasks...")
    all_sentences = []  # List of (file_path, subject_id, sentence_idx)
    
    for task in tasks:
        pickle_dir = dataset_root / task / "pickle"
        if not pickle_dir.exists():
            continue
        
        pickle_files = sorted(list(pickle_dir.glob("*.pickle")))
        
        for pickle_file in pickle_files:
            print(f"  Processing: {pickle_file}")
            
            with open(pickle_file, 'rb') as f:
                data = pickle.load(f)
            
            # Handle nested structure: {subject_id: [sentences]}
            if isinstance(data, dict):
                for subject_id, sentences in data.items():
                    if sentences is not None:
                        for sent_idx in range(len(sentences)):
                            # Store location: (file, subject, sentence_index)
                            all_sentences.append((str(pickle_file), subject_id, sent_idx))
            else:
                # Flat list structure
                for sent_idx in range(len(data)):
                    all_sentences.append((str(pickle_file), None, sent_idx))
    
    total_sentences = len(all_sentences)
    print(f"\nTotal sentences collected: {total_sentences:,}")
    
    # Step 2: Shuffle sentences
    random.shuffle(all_sentences)
    print(f"Sentences shuffled with seed {random_seed}")
    
    # Step 3: Split sentences 80/10/10
    train_end = int(total_sentences * train_ratio)
    dev_end = int(total_sentences * (train_ratio + dev_ratio))
    
    train_sentences = all_sentences[:train_end]
    dev_sentences = all_sentences[train_end:dev_end]
    test_sentences = all_sentences[dev_end:]
    
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
            'split_type': 'sentence_level'  # Important marker!
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


if __name__ == "__main__":
    # Test the sentence-level splitting
    splits = create_sentence_splits(
        dataset_root="dataset/ZuCo",
        tasks=['task1-SR', 'task2-NR', 'task2-NR-2.0', 'task3-TSR', 'task3-TSR-2.0'],
        save_path="data/sentence_splits.pkl"
    )
    
    analyze_sentence_splits("data/sentence_splits.pkl")
