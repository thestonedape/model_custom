"""
Data Preparation Script
Prepares vocabulary and data splits for BELT training
Run this BEFORE training models
"""

import argparse
import sys
from pathlib import Path
import yaml

# Add current directory to path (for standalone usage)
sys.path.insert(0, str(Path(__file__).parent))

from data import build_zuco_vocabulary, create_splits


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare BELT vocabulary and legacy file-level splits")
    parser.add_argument("--config", type=str, default="config/belt_config.yaml")
    return parser.parse_args()


def main():
    """Prepare all data for training"""
    args = parse_args()
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    print("="*80)
    print("BELT DATA PREPARATION")
    print("="*80)
    
    dataset_root = config['data']['dataset_path']
    tasks = list(config['data']['tasks'])
    vocab_size = int(config['data']['vocabulary_size'])
    random_seed = int(config['data'].get('random_seed', 42))
    
    vocab_save_path = config['data'].get('vocab_path', "data/vocabulary_top500.pkl")
    splits_save_path = config['data'].get('legacy_splits_path', "data/splits.pkl")
    
    # Step 1: Build vocabulary
    print("\n" + "="*80)
    print("STEP 1: BUILDING VOCABULARY")
    print("="*80)
    print(f"Dataset root: {dataset_root}")
    print(f"Tasks: {tasks}")
    print(f"Vocabulary size: {vocab_size}")
    print(f"Vocabulary path: {vocab_save_path}")
    
    vocab = build_zuco_vocabulary(
        dataset_root=dataset_root,
        tasks=tasks,
        vocab_size=vocab_size,
        save_path=vocab_save_path
    )
    
    print(f"\n[OK] Vocabulary created and saved to: {vocab_save_path}")
    
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
    print(f"Legacy splits path: {splits_save_path}")
    
    splits = create_splits(
        dataset_root=dataset_root,
        tasks=tasks,
        train_ratio=0.8,
        dev_ratio=0.1,
        test_ratio=0.1,
        random_seed=random_seed,
        save_path=splits_save_path
    )
    
    print(f"\n[OK] Splits created and saved to: {splits_save_path}")

    empty_splits = [name for name in ("train", "dev", "test") if len(splits.get(name, [])) == 0]
    if empty_splits:
        print(
            "[WARN] Legacy file-level splits are degenerate for this task setup: "
            f"{', '.join(empty_splits)} received 0 files."
        )
        print("       Use the sentence-level split workflow for actual BELT comparisons.")
    
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
    print(f"\n[OK] Vocabulary: {vocab_save_path}")
    print(f"[OK] Legacy file-level splits: {splits_save_path}")
    print("\nYou can now run:")
    print(f"  - Sentence splits for baseline training: python prepare_sentence_splits.py --config {args.config}")
    print(f"  - Baseline BELT training: python experiments/model_with_bootstrapping.py --config {args.config}")


if __name__ == "__main__":
    main()
