"""
Prepare Sentence-Level Splits (80/10/10) for BELT
This properly matches the BELT paper's data splitting approach
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from data.sentence_splits import create_sentence_splits, analyze_sentence_splits
from data.vocabulary import Vocabulary


def main():
    print("="*80)
    print("BELT DATA PREPARATION - SENTENCE-LEVEL SPLITS")
    print("="*80)
    print()
    print("This creates proper 80/10/10 splits at SENTENCE level")
    print("(not file level), matching the BELT paper exactly.")
    print()
    
    # Configuration
    dataset_root = "dataset/ZuCo"
    tasks = ['task1-SR', 'task2-NR', 'task2-NR-2.0', 'task3-TSR', 'task3-TSR-2.0']
    
    # Step 1: Create sentence-level splits
    print("="*80)
    print("STEP 1: CREATING SENTENCE-LEVEL SPLITS")
    print("="*80)
    print(f"Dataset root: {dataset_root}")
    print(f"Tasks: {tasks}")
    print(f"Split ratios: 80% train / 10% dev / 10% test")
    print()
    
    splits = create_sentence_splits(
        dataset_root=dataset_root,
        tasks=tasks,
        train_ratio=0.8,
        dev_ratio=0.1,
        test_ratio=0.1,
        random_seed=42,
        save_path="data/sentence_splits.pkl"
    )
    
    # Step 2: Analyze the splits
    print("\n" + "="*80)
    print("STEP 2: ANALYZING SPLITS")
    print("="*80)
    
    analyze_sentence_splits("data/sentence_splits.pkl")
    
    # Step 3: Verify with vocabulary
    print("\n" + "="*80)
    print("STEP 3: VERIFYING WITH VOCABULARY")
    print("="*80)
    
    try:
        vocab = Vocabulary(vocab_size=500)
        vocab.load("data/vocabulary_top500.pkl")
        print(f"✓ Vocabulary loaded: {len(vocab.word2idx)} words")
    except FileNotFoundError:
        print("⚠ Vocabulary not found. Run prepare_data.py first to create it.")
        print("  (The splits are ready, but you need vocabulary for training)")
    
    # Step 4: Summary
    print("\n" + "="*80)
    print("DATA PREPARATION COMPLETE!")
    print("="*80)
    print()
    print("✓ Sentence-level splits created: data/sentence_splits.pkl")
    print()
    print("Split Summary:")
    print(f"  Train: {len(splits['train']):,} sentences ({splits['metadata']['train_ratio']:.1%})")
    print(f"  Dev:   {len(splits['dev']):,} sentences ({splits['metadata']['dev_ratio']:.1%})")
    print(f"  Test:  {len(splits['test']):,} sentences ({splits['metadata']['test_ratio']:.1%})")
    print()
    print("This matches BELT paper's 80/10/10 split! ✓✓✓")
    print()
    print("Next steps:")
    print("  1. Ensure vocabulary exists: python prepare_data.py")
    print("  2. Update training script to use sentence_dataset.py")
    print("  3. Run training: python experiments/model_with_bootstrapping.py")
    print()


if __name__ == "__main__":
    main()
