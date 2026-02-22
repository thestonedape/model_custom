"""
Sentence-Level Dataset for BELT
Works with sentence-level splits (80/10/10) instead of file-level splits
"""

import torch
from torch.utils.data import Dataset
import pickle
import numpy as np
from typing import List, Tuple, Optional
from pathlib import Path


class BELTSentenceDataset(Dataset):
    """
    Dataset that loads specific sentences based on sentence-level splits
    
    This properly implements BELT's data loading by:
    - Loading only sentences assigned to this split (train/dev/test)
    - Supporting 80/10/10 sentence-level splits (not file-level)
    """
    
    def __init__(
        self,
        sentence_list: List[Tuple[str, Optional[str], int]],
        vocabulary,
        split: str = 'train',
        eeg_type: str = 'GD'
    ):
        """
        Args:
            sentence_list: List of (file_path, subject_id, sentence_idx) tuples
            vocabulary: Vocabulary object
            split: 'train', 'dev', or 'test'
            eeg_type: 'GD', 'FFD', or 'TRT'
        """
        self.vocabulary = vocabulary
        self.split = split
        self.eeg_type = eeg_type
        self.samples = []
        
        # Cache for loaded pickle files to avoid repeated loading
        self._cache = {}
        
        print(f"Loading {split} data from sentence-level splits...")
        print(f"  Total sentences: {len(sentence_list):,}")
        print(f"  Using EEG type: {eeg_type}")
        
        # Load samples from specified sentences
        for file_path, subject_id, sent_idx in sentence_list:
            # Load pickle file (with caching)
            if file_path not in self._cache:
                with open(file_path, 'rb') as f:
                    self._cache[file_path] = pickle.load(f)
            
            data = self._cache[file_path]
            
            # Get the specific sentence
            if isinstance(data, dict):
                # Nested structure: {subject_id: [sentences]}
                if subject_id not in data or data[subject_id] is None:
                    continue
                sentences = data[subject_id]
                if sent_idx >= len(sentences):
                    continue
                sentence_data = sentences[sent_idx]
            else:
                # Flat structure
                if sent_idx >= len(data):
                    continue
                sentence_data = data[sent_idx]
            
            if sentence_data is None or 'word' not in sentence_data:
                continue
            
            # Extract words from this sentence
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
                    'word': word
                })
        
        # Clear cache to free memory
        self._cache.clear()
        
        print(f"Loaded {len(self.samples):,} samples for {split} split")
    
    def _extract_eeg_features(self, word_data, eeg_type: str) -> Optional[np.ndarray]:
        """
        Extract EEG features for a word
        
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


def load_sentence_splits(splits_path: str = "data/sentence_splits.pkl"):
    """Load sentence-level splits"""
    with open(splits_path, 'rb') as f:
        splits = pickle.load(f)
    
    # Verify it's sentence-level splits
    if splits['metadata'].get('split_type') != 'sentence_level':
        raise ValueError(f"Expected sentence-level splits, got: {splits['metadata'].get('split_type')}")
    
    return splits


def create_sentence_dataloaders(
    vocabulary,
    batch_size: int = 64,
    num_workers: int = 0,
    splits_path: str = "data/sentence_splits.pkl",
    eeg_type: str = 'GD'
):
    """
    Create dataloaders using sentence-level splits
    
    Returns:
        train_loader, dev_loader, test_loader
    """
    from torch.utils.data import DataLoader
    
    # Load sentence splits
    splits = load_sentence_splits(splits_path)
    
    print("\n" + "="*80)
    print("CREATING SENTENCE-LEVEL DATALOADERS")
    print("="*80)
    
    # Create datasets
    train_dataset = BELTSentenceDataset(
        sentence_list=splits['train'],
        vocabulary=vocabulary,
        split='train',
        eeg_type=eeg_type
    )
    
    dev_dataset = BELTSentenceDataset(
        sentence_list=splits['dev'],
        vocabulary=vocabulary,
        split='dev',
        eeg_type=eeg_type
    )
    
    test_dataset = BELTSentenceDataset(
        sentence_list=splits['test'],
        vocabulary=vocabulary,
        split='test',
        eeg_type=eeg_type
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False
    )
    
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )
    
    print(f"\nDataloaders created:")
    print(f"  Train: {len(train_dataset):,} samples, {len(train_loader):,} batches")
    print(f"  Dev:   {len(dev_dataset):,} samples, {len(dev_loader):,} batches")
    print(f"  Test:  {len(test_dataset):,} samples, {len(test_loader):,} batches")
    
    return train_loader, dev_loader, test_loader
