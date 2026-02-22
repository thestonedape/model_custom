"""
Data Preparation Script
Prepares vocabulary and data splits for BELT training
Run this BEFORE training models
"""

import sys
from pathlib import Path

# Add current directory to path (for standalone usage)
sys.path.insert(0, str(Path(__file__).parent))

from data import build_zuco_vocabulary, create_splits


def main():
    """Prepare all data for training"""
    
    print("="*80)
    print("BELT DATA PREPARATION")
    print("="*80)
    
    # Configuration (updated for standalone structure)
    dataset_root = "dataset/ZuCo"
    tasks = ["task1-SR", "task2-NR", "task2-NR-2.0", "task3-TSR", "task3-TSR-2.0"]
    vocab_size = 500
    
    vocab_save_path = "data/vocabulary_top500.pkl"
    splits_save_path = "data/splits.pkl"
    
    # Step 1: Build vocabulary
    print("\n" + "="*80)
    print("STEP 1: BUILDING VOCABULARY")
    print("="*80)
    print(f"Dataset root: {dataset_root}")
    print(f"Tasks: {tasks}")
    print(f"Vocabulary size: {vocab_size}")
    
    vocab = build_zuco_vocabulary(
        dataset_root=dataset_root,
        tasks=tasks,
        vocab_size=vocab_size,
        save_path=vocab_save_path
    )
    
    print(f"\n✓ Vocabulary created and saved to: {vocab_save_path}")
    
    # Print vocabulary statistics
    stats = vocab.get_statistics()
    print(f"\nVocabulary Statistics:")
    print(f"  Vocabulary size: {stats['vocab_size']}")
    print(f"  Total unique words: {stats['total_unique_words']}")
    print(f"  Total word instances: {stats['total_word_instances']}")
    print(f"  Coverage: {stats['coverage']:.2%}")
    
    # Step 2: Create data splits
    print("\n" + "="*80)
    print("STEP 2: CREATING DATA SPLITS")
    print("="*80)
    print("Train: 80% | Dev: 10% | Test: 10%")
    
    splits = create_splits(
        dataset_root=dataset_root,
        tasks=tasks,
        train_ratio=0.8,
        dev_ratio=0.1,
        test_ratio=0.1,
        random_seed=42,
        save_path=splits_save_path
    )
    
    print(f"\n✓ Splits created and saved to: {splits_save_path}")
    
    # Step 3: Count word instances per split
    print("\n" + "="*80)
    print("STEP 3: ANALYZING SPLITS")
    print("="*80)
    
    from data.splits import count_words_per_split
    
    split_stats = count_words_per_split(
        dataset_root=dataset_root,
        tasks=tasks,
        splits=splits,
        vocabulary=vocab
    )
    
    print("\nWord Statistics per Split:")
    for split_name in ['train', 'dev', 'test']:
        stats = split_stats[split_name]
        print(f"\n{split_name.upper()}:")
        print(f"  Files: {stats['num_files']}")
        print(f"  Total words: {stats['total_words']}")
        print(f"  In-vocabulary words: {stats['in_vocab_words']}")
        print(f"  Coverage: {stats['coverage']:.2%}")
    
    # Summary
    print("\n" + "="*80)
    print("DATA PREPARATION COMPLETE!")
    print("="*80)
    print(f"\n✓ Vocabulary: {vocab_save_path}")
    print(f"✓ Splits: {splits_save_path}")
    print("\nYou can now run:")
    print("  - Model 1 (ablation): python model_custom/experiments/model_without_bootstrapping.py")
    print("  - Model 2 (full BELT): python model_custom/experiments/model_with_bootstrapping.py")


if __name__ == "__main__":
    main()
