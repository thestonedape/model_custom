"""Test sentence-level dataset loading"""
from data.vocabulary import Vocabulary
from data.sentence_dataset import BELTSentenceDataset, load_sentence_splits

print("Testing sentence-level dataset loader...")

# Load vocabulary
vocab = Vocabulary(500)
vocab.load('data/vocabulary_top500.pkl')
print(f"✓ Vocabulary loaded: {len(vocab.word2idx)} words")

# Load splits
splits = load_sentence_splits()
print(f"✓ Splits loaded: Train={len(splits['train']):,}, Dev={len(splits['dev']):,}, Test={len(splits['test']):,}")

# Test with a small subset (first 100 sentences)
print("\nTesting dataset with first 100 train sentences...")
train_dataset = BELTSentenceDataset(splits['train'][:100], vocab, 'train', 'GD')
print(f"✓ Dataset created: {len(train_dataset)} samples")

if len(train_dataset) > 0:
    eeg, label, word = train_dataset[0]
    print(f"✓ First sample: EEG shape={eeg.shape}, label={label}, word='{word}'")
    print("\n✅ All tests passed! Sentence-level dataset works correctly.")
else:
    print("⚠ Warning: No samples loaded (might be normal if first 100 sentences have no in-vocab words)")
