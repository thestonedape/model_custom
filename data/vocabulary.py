"""
Vocabulary Selection for BELT Word Classification
Selects the top-500 most frequent words from ZuCo dataset
"""

import pickle
from collections import Counter
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np


class Vocabulary:
    """Build and manage vocabulary of top-K most frequent words"""
    
    def __init__(self, vocab_size: int = 500):
        """
        Args:
            vocab_size: Number of most frequent words to keep (default: 500)
        """
        self.vocab_size = vocab_size
        self.word2idx: Dict[str, int] = {}
        self.idx2word: Dict[int, str] = {}
        self.word_counts: Counter = Counter()
        
    def build_from_pickle_files(self, dataset_paths: List[str]) -> None:
        """
        Build vocabulary from preprocessed ZuCo pickle files
        
        Args:
            dataset_paths: List of paths to pickle files
        """
        print(f"Building vocabulary from {len(dataset_paths)} datasets...")
        
        # Count word frequencies across all datasets
        for dataset_path in dataset_paths:
            print(f"  Processing: {dataset_path}")
            with open(dataset_path, 'rb') as f:
                data = pickle.load(f)
            
            # Data structure: {subject_id: [sentences]}
            # Iterate through all subjects
            for subject_id, sentences in data.items():
                # Count words in each sentence
                for sentence_data in sentences:
                    if sentence_data is None or 'word' not in sentence_data:
                        continue
                    for word_data in sentence_data['word']:
                        if word_data is None or 'content' not in word_data:
                            continue
                        word = word_data['content'].lower()  # Lowercase normalization
                        self.word_counts[word] += 1
        
        # Select top-K most frequent words
        most_common = self.word_counts.most_common(self.vocab_size)
        
        print(f"\nVocabulary Statistics:")
        print(f"  Total unique words: {len(self.word_counts)}")
        print(f"  Selected vocabulary size: {self.vocab_size}")
        print(f"  Coverage: {sum([count for _, count in most_common]) / sum(self.word_counts.values()):.2%}")
        
        # Build word2idx and idx2word mappings
        for idx, (word, count) in enumerate(most_common):
            self.word2idx[word] = idx
            self.idx2word[idx] = word
        
        # Print top-20 most frequent words
        print(f"\nTop-20 most frequent words:")
        for i, (word, count) in enumerate(most_common[:20]):
            print(f"  {i+1:2d}. '{word}': {count} occurrences")
    
    def get_word_index(self, word: str) -> int:
        """
        Get index for a word (returns -1 if not in vocabulary)
        
        Args:
            word: Word string
            
        Returns:
            Index (0 to vocab_size-1) or -1 if not in vocabulary
        """
        return self.word2idx.get(word.lower(), -1)
    
    def get_word_from_index(self, idx: int) -> str:
        """
        Get word from index
        
        Args:
            idx: Word index
            
        Returns:
            Word string
        """
        return self.idx2word.get(idx, "<UNK>")
    
    def is_in_vocabulary(self, word: str) -> bool:
        """Check if word is in vocabulary"""
        return word.lower() in self.word2idx
    
    def save(self, save_path: str) -> None:
        """Save vocabulary to file"""
        vocab_data = {
            'vocab_size': self.vocab_size,
            'word2idx': self.word2idx,
            'idx2word': self.idx2word,
            'word_counts': dict(self.word_counts)
        }
        with open(save_path, 'wb') as f:
            pickle.dump(vocab_data, f)
        print(f"Vocabulary saved to: {save_path}")
    
    def load(self, load_path: str) -> None:
        """Load vocabulary from file"""
        with open(load_path, 'rb') as f:
            vocab_data = pickle.load(f)
        
        self.vocab_size = vocab_data['vocab_size']
        self.word2idx = vocab_data['word2idx']
        self.idx2word = vocab_data['idx2word']
        self.word_counts = Counter(vocab_data['word_counts'])
        print(f"Vocabulary loaded from: {load_path}")
        print(f"  Vocabulary size: {self.vocab_size}")
    
    def get_statistics(self) -> Dict:
        """Get vocabulary statistics"""
        return {
            'vocab_size': self.vocab_size,
            'total_unique_words': len(self.word_counts),
            'total_word_instances': sum(self.word_counts.values()),
            'coverage': sum([self.word_counts[word] for word in self.word2idx.keys()]) / sum(self.word_counts.values()),
            'most_frequent': self.word_counts.most_common(10),
            'least_frequent_in_vocab': [(word, self.word_counts[word]) for word in list(self.word2idx.keys())[-10:]]
        }


def build_zuco_vocabulary(
    dataset_root: str = "dataset/ZuCo",
    tasks: List[str] = ["task1-SR", "task2-NR", "task3-TSR"],
    vocab_size: int = 500,
    save_path: str = None
) -> Vocabulary:
    """
    Build vocabulary from ZuCo dataset (convenience function)
    
    Args:
        dataset_root: Root directory of ZuCo dataset
        tasks: List of tasks to include
        vocab_size: Number of most frequent words (default: 500)
        save_path: Path to save vocabulary (optional)
    
    Returns:
        Vocabulary object
    """
    dataset_root = Path(dataset_root)
    
    # Find all pickle files
    pickle_files = []
    for task in tasks:
        task_dir = dataset_root / task / "pickle"  # Look in pickle subdirectory
        if task_dir.exists():
            pickle_files.extend(list(task_dir.glob("*.pickle")))
    
    pickle_paths = [str(p) for p in pickle_files]
    
    if not pickle_paths:
        raise ValueError(f"No pickle files found in {dataset_root} for tasks {tasks}")
    
    # Build vocabulary
    vocab = Vocabulary(vocab_size=vocab_size)
    vocab.build_from_pickle_files(pickle_paths)
    
    # Save if requested
    if save_path:
        vocab.save(save_path)
    
    return vocab


if __name__ == "__main__":
    """Test vocabulary building"""
    
    # Build vocabulary from ZuCo dataset
    print("="*80)
    print("BUILDING BELT VOCABULARY (TOP-500 WORDS)")
    print("="*80)
    
    vocab = build_zuco_vocabulary(
        dataset_root="dataset/ZuCo",
        tasks=["task1-SR", "task2-NR", "task3-TSR"],
        vocab_size=500,
        save_path="model_custom/data/vocabulary_top500.pkl"
    )
    
    # Print statistics
    print("\n" + "="*80)
    print("VOCABULARY STATISTICS")
    print("="*80)
    stats = vocab.get_statistics()
    for key, value in stats.items():
        if isinstance(value, list):
            print(f"\n{key}:")
            for item in value:
                print(f"  {item}")
        else:
            print(f"{key}: {value}")
    
    # Test word lookup
    print("\n" + "="*80)
    print("VOCABULARY LOOKUP TESTS")
    print("="*80)
    test_words = ["the", "brain", "activity", "reading", "uncommonword123"]
    for word in test_words:
        idx = vocab.get_word_index(word)
        in_vocab = vocab.is_in_vocabulary(word)
        print(f"'{word}': index={idx}, in_vocab={in_vocab}")
