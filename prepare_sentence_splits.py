"""
Prepare sentence-level splits for BELT-style experiments.
"""

import argparse
import sys
from pathlib import Path
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from data.sentence_splits import create_sentence_splits, analyze_sentence_splits
from data.vocabulary import Vocabulary


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare sentence-level splits for BELT baseline training")
    parser.add_argument("--config", type=str, default="config/belt_config.yaml")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    print("="*80)
    print("BELT DATA PREPARATION - SENTENCE-LEVEL SPLITS")
    print("="*80)
    print()
    print("This creates configurable 80/10/10 splits at SENTENCE level")
    print("(not file level) for the task set and split mode in your config.")
    print()
    
    # Configuration
    dataset_root = config['data']['dataset_path']
    tasks = list(config['data']['tasks'])
    random_seed = int(config['data'].get('random_seed', 42))
    split_mode = config['data'].get('split_mode', 'sentence_instance')
    splits_path = config['data'].get('splits_path', "data/sentence_splits.pkl")
    vocab_path = config['data'].get('vocab_path', "data/vocabulary_top500.pkl")
    
    # Step 1: Create sentence-level splits
    print("="*80)
    print("STEP 1: CREATING SENTENCE-LEVEL SPLITS")
    print("="*80)
    print(f"Dataset root: {dataset_root}")
    print(f"Tasks: {tasks}")
    print(f"Split ratios: 80% train / 10% dev / 10% test")
    print(f"Split mode: {split_mode}")
    print(f"Save path: {splits_path}")
    print()
    
    splits = create_sentence_splits(
        dataset_root=dataset_root,
        tasks=tasks,
        train_ratio=0.8,
        dev_ratio=0.1,
        test_ratio=0.1,
        random_seed=random_seed,
        split_mode=split_mode,
        save_path=splits_path
    )
    
    # Step 2: Analyze the splits
    print("\n" + "="*80)
    print("STEP 2: ANALYZING SPLITS")
    print("="*80)
    
    analyze_sentence_splits(splits_path)
    
    # Step 3: Verify with vocabulary
    print("\n" + "="*80)
    print("STEP 3: VERIFYING WITH VOCABULARY")
    print("="*80)
    
    try:
        vocab = Vocabulary(vocab_size=500)
        vocab.load(vocab_path)
        print(f"[OK] Vocabulary loaded: {len(vocab.word2idx)} words")
    except FileNotFoundError:
        print(f"[WARN] Vocabulary not found at {vocab_path}. Run prepare_data.py first to create it.")
        print("  (The splits are ready, but you need vocabulary for training)")
    
    # Step 4: Summary
    print("\n" + "="*80)
    print("DATA PREPARATION COMPLETE!")
    print("="*80)
    print()
    print(f"[OK] Sentence-level splits created: {splits_path}")
    print()
    print("Split Summary:")
    print(f"  Train: {len(splits['train']):,} sentences ({splits['metadata']['train_ratio']:.1%})")
    print(f"  Dev:   {len(splits['dev']):,} sentences ({splits['metadata']['dev_ratio']:.1%})")
    print(f"  Test:  {len(splits['test']):,} sentences ({splits['metadata']['test_ratio']:.1%})")
    print()
    print("Configured split generation complete.")
    print()
    print("Next steps:")
    print(f"  1. Ensure vocabulary exists: python prepare_data.py --config {args.config}")
    print(f"  2. Run baseline training: python experiments/model_with_bootstrapping.py --config {args.config}")
    print()


if __name__ == "__main__":
    main()
