"""
Sentence-Level Dataset for BELT
Works with sentence-level splits (80/10/10) instead of file-level splits
"""

import hashlib
import torch
from torch.utils.data import Dataset
import pickle
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path


CACHE_VERSION = 2


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
        eeg_type: str = 'GD',
        use_cache: bool = True,
        eeg_types: Optional[List[str]] = None,
        include_n_fixations: bool = False
    ):
        """
        Args:
            sentence_list: List of (file_path, subject_id, sentence_idx) tuples
            vocabulary: Vocabulary object
            split: 'train', 'dev', or 'test'
            eeg_type: 'GD', 'FFD', or 'TRT'
            use_cache: If True, cache processed samples to disk
        """
        self.vocabulary = vocabulary
        self.split = split
        self.eeg_type = eeg_type
        self.eeg_types = list(eeg_types) if eeg_types else [eeg_type]
        self.include_n_fixations = include_n_fixations
        self.samples = []
        
        # Fingerprint caches by split contents and vocabulary so old runs
        # cannot silently reuse stale labels after task/split changes.
        cache_path = self._build_cache_path(sentence_list)
        if use_cache and cache_path.exists():
            print(f"Loading {split} data from cache: {cache_path}")
            with open(cache_path, 'rb') as f:
                self.samples = pickle.load(f)
            print(f"Loaded {len(self.samples):,} samples for {split} split (cached)")
            return
        
        # Cache for loaded pickle files to avoid repeated loading
        self._cache = {}
        
        print(f"Processing {split} data from sentence-level splits...")
        print(f"  Total sentences: {len(sentence_list):,}")
        print(f"  Using EEG types: {self.eeg_types}")
        print(f"  Include nFixations: {self.include_n_fixations}")
        
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
                eeg_features = self._extract_eeg_features(word_data)
                
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
        if self.samples:
            print(f"  Feature dimension: {len(self.samples[0]['eeg'])}")
        
        # Save to cache for next time
        if use_cache:
            print(f"Saving processed samples to cache: {cache_path}")
            with open(cache_path, 'wb') as f:
                pickle.dump(self.samples, f)

    def _build_cache_path(self, sentence_list: List[Tuple[str, Optional[str], int]]) -> Path:
        """Build a cache path tied to the exact split membership and vocabulary."""
        hasher = hashlib.sha1()
        hasher.update(
            (
                f"{CACHE_VERSION}|{self.split}|{self.eeg_type}|{self.eeg_types}|"
                f"{self.include_n_fixations}|{len(sentence_list)}|{self.vocabulary.vocab_size}"
            ).encode('utf-8')
        )
        for word, idx in sorted(self.vocabulary.word2idx.items()):
            hasher.update(f"vocab|{word}|{idx}\n".encode('utf-8'))
        for file_path, subject_id, sent_idx in sentence_list:
            hasher.update(f"sample|{file_path}|{subject_id}|{sent_idx}\n".encode('utf-8'))
        
        cache_dir = Path("data/cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_name = f"processed_{self.split}_{self.eeg_type}_{hasher.hexdigest()[:16]}.pkl"
        return cache_dir / cache_name
    
    def _extract_single_eeg_features(self, word_data, eeg_type: str) -> Optional[np.ndarray]:
        """
        Extract one EEG summary type for a word.
        
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

    def _extract_eeg_features(self, word_data) -> Optional[np.ndarray]:
        """
        Extract the configured EEG feature vector for a word.

        This supports single-type BELT input as well as fused GD/FFD/TRT input.
        """
        features = []

        for eeg_type in self.eeg_types:
            eeg_features = self._extract_single_eeg_features(word_data, eeg_type)
            if eeg_features is None:
                return None
            features.append(eeg_features)

        if self.include_n_fixations:
            features.append(np.asarray([float(word_data.get('nFixations', 0.0))], dtype=np.float32))

        fused = np.concatenate(features, axis=0)
        if np.isnan(fused).any() or np.isinf(fused).any():
            return None

        return fused.astype(np.float32)
    
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
        """Get the label counts for the current split."""
        from collections import Counter

        labels = [sample['label'] for sample in self.samples]
        return dict(Counter(labels))

    def get_sample_weights(self) -> List[float]:
        """Return inverse-frequency sample weights for weighted sampling."""
        label_counts = self.get_label_distribution()
        return [1.0 / label_counts[sample['label']] for sample in self.samples]


def load_sentence_splits(splits_path: str = "data/sentence_splits.pkl"):
    """Load sentence-level splits"""
    with open(splits_path, 'rb') as f:
        splits = pickle.load(f)
    
    # Verify it's sentence-level splits
    split_type = splits['metadata'].get('split_type')
    if split_type not in {'sentence_level', 'sentence_instance', 'unique_sentence_text'}:
        raise ValueError(f"Expected sentence-level splits, got: {split_type}")
    
    return splits


def create_sentence_dataloaders(
    vocabulary,
    batch_size: int = 64,
    num_workers: int = 0,
    splits_path: str = "data/sentence_splits.pkl",
    eeg_type: str = 'GD',
    use_cache: bool = True,
    pin_memory: bool = False,
    eeg_types: Optional[List[str]] = None,
    include_n_fixations: bool = False,
    use_weighted_sampler: bool = False
):
    """
    Create dataloaders using sentence-level splits
    
    Returns:
        train_loader, dev_loader, test_loader
    """
    from torch.utils.data import DataLoader, WeightedRandomSampler
    
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
        eeg_type=eeg_type,
        use_cache=use_cache,
        eeg_types=eeg_types,
        include_n_fixations=include_n_fixations
    )
    
    dev_dataset = BELTSentenceDataset(
        sentence_list=splits['dev'],
        vocabulary=vocabulary,
        split='dev',
        eeg_type=eeg_type,
        use_cache=use_cache,
        eeg_types=eeg_types,
        include_n_fixations=include_n_fixations
    )
    
    test_dataset = BELTSentenceDataset(
        sentence_list=splits['test'],
        vocabulary=vocabulary,
        split='test',
        eeg_type=eeg_type,
        use_cache=use_cache,
        eeg_types=eeg_types,
        include_n_fixations=include_n_fixations
    )
    
    # Create dataloaders
    train_sampler = None
    train_shuffle = True
    if use_weighted_sampler:
        sample_weights = torch.as_tensor(train_dataset.get_sample_weights(), dtype=torch.double)
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        train_shuffle = False
        print("  Train sampler: inverse-frequency weighted sampling enabled")
    else:
        print("  Train sampler: standard shuffled batches")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=train_shuffle,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    print(f"\nDataloaders created:")
    print(f"  Train: {len(train_dataset):,} samples, {len(train_loader):,} batches")
    print(f"  Dev:   {len(dev_dataset):,} samples, {len(dev_loader):,} batches")
    print(f"  Test:  {len(test_dataset):,} samples, {len(test_loader):,} batches")
    
    return train_loader, dev_loader, test_loader
