"""
BELT Dataset Class for EEG-to-Word Classification
Loads preprocessed ZuCo data and provides (EEG, label, word) tuples
"""

import pickle
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np


class BELTWordDataset(Dataset):
    """
    Dataset for EEG-to-word classification
    
    Returns:
        eeg: (840,) tensor - 105 electrodes × 8 frequency bands
        label: int - word class (0 to vocab_size-1)
        word: str - original word text (for contrastive loss)
    """
    
    def __init__(
        self,
        pickle_files: List[str],
        vocabulary,
        split: str = "train",
        eeg_type: str = "GD"
    ):
        """
        Args:
            pickle_files: List of paths to preprocessed pickle files
            vocabulary: Vocabulary object with word2idx mapping
            split: "train", "dev", or "test"
            eeg_type: "FFD", "GD", or "TRT" (default: GD as in BELT paper)
        """
        self.vocabulary = vocabulary
        self.split = split
        self.eeg_type = eeg_type
        self.samples = []
        
        print(f"Loading {split} data from {len(pickle_files)} files...")
        print(f"Using EEG type: {eeg_type}")
        
        # Load and filter samples
        for pickle_file in pickle_files:
            with open(pickle_file, 'rb') as f:
                data = pickle.load(f)
            
            # Handle nested structure: {subject_id: [sentences]}
            if isinstance(data, dict):
                # Flatten: iterate through all subjects
                all_sentences = []
                for subject_id, sentences in data.items():
                    if sentences is not None:
                        all_sentences.extend(sentences)
                data = all_sentences
            
            for sentence_data in data:
                if sentence_data is None or 'word' not in sentence_data:
                    continue
                    
                for word_data in sentence_data['word']:
                    if word_data is None:
                        continue
                        
                    word = word_data['content'].lower()
                    
                    # Only include words in vocabulary
                    if not vocabulary.is_in_vocabulary(word):
                        continue
                    
                    # Get word label
                    label = vocabulary.get_word_index(word)
                    
                    # Get EEG features
                    eeg_features = self._extract_eeg_features(word_data, eeg_type)
                    
                    # Skip if EEG data is missing or invalid
                    if eeg_features is None:
                        continue
                    
                    self.samples.append({
                        'eeg': eeg_features,
                        'label': label,
                        'word': word,
                        'nFixations': word_data['nFixations']
                    })
        
        print(f"Loaded {len(self.samples)} samples for {split} split")
        
    def _extract_eeg_features(self, word_data: Dict, eeg_type: str) -> Optional[np.ndarray]:
        """
        Extract EEG features from word data
        
        Args:
            word_data: Dictionary containing word-level EEG data
            eeg_type: "FFD", "GD", or "TRT"
            
        Returns:
            (840,) array or None if invalid
        """
        try:
            eeg_dict = word_data['word_level_EEG'][eeg_type]
            
            # Concatenate all 8 frequency bands
            # Each band has 105 electrodes
            bands = ['t1', 't2', 'a1', 'a2', 'b1', 'b2', 'g1', 'g2']
            features = []
            
            for band in bands:
                # EEG type prefix is in the key (e.g., 'FFD_t1', 'GD_t1')
                band_key = f"{eeg_type}_{band}"
                
                if band_key not in eeg_dict:
                    return None
                    
                band_data = eeg_dict[band_key]
                if isinstance(band_data, list):
                    band_data = np.array(band_data)
                features.append(band_data)
            
            # Concatenate: (8 bands, 105 electrodes) → (840,)
            features = np.concatenate(features, axis=0)
            
            # Validate shape
            if features.shape[0] != 840:
                return None
            
            # Check for NaN or Inf
            if np.isnan(features).any() or np.isinf(features).any():
                return None
            
            return features.astype(np.float32)
        
        except (KeyError, ValueError, TypeError):
            return None
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        """
        Get a single sample
        
        Returns:
            eeg: (840,) FloatTensor
            label: int (word class)
            word: str (word text)
        """
        sample = self.samples[idx]
        
        eeg = torch.from_numpy(sample['eeg'])  # (840,)
        label = sample['label']
        word = sample['word']
        
        return eeg, label, word
    
    def get_label_distribution(self) -> Dict[int, int]:
        """Get distribution of labels in dataset"""
        from collections import Counter
        labels = [sample['label'] for sample in self.samples]
        return dict(Counter(labels))
    
    def get_statistics(self) -> Dict:
        """Get dataset statistics"""
        labels = [sample['label'] for sample in self.samples]
        eeg_features = np.stack([sample['eeg'] for sample in self.samples])
        
        return {
            'num_samples': len(self.samples),
            'num_unique_labels': len(set(labels)),
            'eeg_mean': eeg_features.mean(),
            'eeg_std': eeg_features.std(),
            'eeg_min': eeg_features.min(),
            'eeg_max': eeg_features.max(),
            'samples_per_word': {
                'mean': len(self.samples) / len(set(labels)),
                'min': min(self.get_label_distribution().values()),
                'max': max(self.get_label_distribution().values())
            }
        }


def create_dataloaders(
    dataset_root: str,
    vocabulary,
    tasks: List[str],
    splits_dict: Dict[str, List[int]],
    batch_size: int = 64,
    num_workers: int = 4,
    eeg_type: str = "GD"
) -> Tuple[torch.utils.data.DataLoader, ...]:
    """
    Create train, dev, test dataloaders
    
    Args:
        dataset_root: Root directory of ZuCo dataset
        vocabulary: Vocabulary object
        tasks: List of task names
        splits_dict: Dictionary with 'train', 'dev', 'test' file indices
        batch_size: Batch size
        num_workers: Number of data loading workers
        eeg_type: EEG feature type
        
    Returns:
        train_loader, dev_loader, test_loader
    """
    dataset_root = Path(dataset_root)
    
    # Collect all pickle files
    all_pickle_files = []
    for task in tasks:
        task_dir = dataset_root / task
        if task_dir.exists():
            all_pickle_files.extend(sorted(list(task_dir.glob("*.pickle"))))
    
    all_pickle_files = [str(p) for p in all_pickle_files]
    
    # Split files according to splits_dict
    train_files = [all_pickle_files[i] for i in splits_dict['train']]
    dev_files = [all_pickle_files[i] for i in splits_dict['dev']]
    test_files = [all_pickle_files[i] for i in splits_dict['test']]
    
    print(f"\nDataset split:")
    print(f"  Train: {len(train_files)} files")
    print(f"  Dev: {len(dev_files)} files")
    print(f"  Test: {len(test_files)} files")
    
    # Create datasets
    train_dataset = BELTWordDataset(train_files, vocabulary, split="train", eeg_type=eeg_type)
    dev_dataset = BELTWordDataset(dev_files, vocabulary, split="dev", eeg_type=eeg_type)
    test_dataset = BELTWordDataset(test_files, vocabulary, split="test", eeg_type=eeg_type)
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    dev_loader = torch.utils.data.DataLoader(
        dev_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, dev_loader, test_loader


if __name__ == "__main__":
    """Test dataset loading"""
    from vocabulary import Vocabulary
    
    print("="*80)
    print("TESTING BELT DATASET")
    print("="*80)
    
    # Load vocabulary
    vocab = Vocabulary(vocab_size=500)
    vocab.load("model_custom/data/vocabulary_top500.pkl")
    
    # Load a sample dataset
    from pathlib import Path
    pickle_files = list(Path("dataset/ZuCo/task1-SR").glob("*.pickle"))[:5]
    
    dataset = BELTWordDataset(
        pickle_files=[str(p) for p in pickle_files],
        vocabulary=vocab,
        split="train",
        eeg_type="GD"
    )
    
    print(f"\nDataset statistics:")
    stats = dataset.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print(f"\nSample data:")
    for i in range(3):
        eeg, label, word = dataset[i]
        print(f"  Sample {i}: word='{word}', label={label}, eeg_shape={eeg.shape}")
    
    print("\nDataLoader test:")
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    batch = next(iter(loader))
    eeg_batch, label_batch, word_batch = batch
    print(f"  Batch EEG shape: {eeg_batch.shape}")
    print(f"  Batch labels shape: {label_batch.shape}")
    print(f"  Batch words (first 5): {word_batch[:5]}")
