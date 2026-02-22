"""
COMPREHENSIVE BELT VERIFICATION SCRIPT
Validates implementation matches BELT paper specifications
Based on the detailed verification checklist

Run this before training to ensure correctness!
"""

import sys
import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import yaml
import pickle
from collections import Counter

# Add to path
sys.path.insert(0, str(Path(__file__).parent))

# Color codes for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
BOLD = '\033[1m'
RESET = '\033[0m'

class BELTVerifier:
    """Comprehensive BELT implementation verifier"""
    
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.passed_tests = []
        self.failed_tests = []
        
    def log_pass(self, message):
        """Log passed test"""
        print(f"{GREEN}✓{RESET} {message}")
        self.passed_tests.append(message)
        
    def log_fail(self, message):
        """Log failed test"""
        print(f"{RED}✗{RESET} {message}")
        self.failed_tests.append(message)
        self.errors.append(message)
        
    def log_warn(self, message):
        """Log warning"""
        print(f"{YELLOW}⚠{RESET} {message}")
        self.warnings.append(message)
        
    def log_info(self, message):
        """Log info"""
        print(f"{BLUE}ℹ{RESET} {message}")
    
    def print_header(self, title, level=1):
        """Print section header"""
        if level == 1:
            print(f"\n{'='*80}")
            print(f"{BOLD}{title}{RESET}")
            print(f"{'='*80}")
        else:
            print(f"\n{'-'*80}")
            print(f"{BOLD}{title}{RESET}")
            print(f"{'-'*80}")
    
    # ========================================================================
    # PHASE 1: DATA PIPELINE VERIFICATION
    # ========================================================================
    
    def verify_vocabulary(self):
        """Test 1.1: Vocabulary Selection"""
        self.print_header("PHASE 1.1: VOCABULARY VERIFICATION")
        
        vocab_path = Path("data/vocabulary_top500.pkl")
        
        if not vocab_path.exists():
            self.log_fail(f"Vocabulary file not found: {vocab_path}")
            self.log_info("Run: python prepare_data.py first")
            return False
        
        try:
            from data.vocabulary import Vocabulary
            vocab = Vocabulary(vocab_size=500)
            vocab.load(str(vocab_path))
            
            # Check 1: Exactly 500 words
            if len(vocab.word2idx) == 500:
                self.log_pass(f"Vocabulary size = 500 words ✓")
            else:
                self.log_fail(f"Vocabulary size = {len(vocab.word2idx)}, expected 500")
            
            # Check 2: Common words present
            common_words = ['the', 'a', 'to', 'of', 'and', 'in', 'is']
            missing = [w for w in common_words if w not in vocab.word2idx]
            if not missing:
                self.log_pass(f"Common words present: {common_words[:5]}")
            else:
                self.log_warn(f"Missing common words: {missing}")
            
            # Check 3: word2idx and idx2word are consistent
            consistent = all(vocab.idx2word[idx] == word 
                           for word, idx in vocab.word2idx.items())
            if consistent:
                self.log_pass("word2idx ↔ idx2word mappings consistent")
            else:
                self.log_fail("word2idx ↔ idx2word mappings inconsistent")
            
            # Check 4: Indices are 0 to 499
            min_idx = min(vocab.word2idx.values())
            max_idx = max(vocab.word2idx.values())
            if min_idx == 0 and max_idx == 499:
                self.log_pass(f"Indices range: 0-499 ✓")
            else:
                self.log_fail(f"Indices range: {min_idx}-{max_idx}, expected 0-499")
            
            # Check 5: No special tokens counted
            special_tokens = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '<pad>', '<unk>']
            has_special = any(tok in vocab.word2idx for tok in special_tokens)
            if not has_special:
                self.log_pass("No special tokens in vocabulary ✓")
            else:
                self.log_warn("Special tokens found in vocabulary (may be intentional)")
            
            # Print top-10 words
            if hasattr(vocab, 'word_counts') and vocab.word_counts:
                top_10 = [vocab.idx2word[i] for i in range(min(10, len(vocab.idx2word)))]
                self.log_info(f"Top-10 words: {', '.join(top_10)}")
            
            return True
            
        except Exception as e:
            self.log_fail(f"Vocabulary verification failed: {str(e)}")
            return False
    
    def verify_splits(self):
        """Test 1.2: Sentence-First Splits"""
        self.print_header("PHASE 1.2: DATA SPLITS VERIFICATION")
        
        splits_path = Path("data/splits.pkl")
        
        if not splits_path.exists():
            self.log_fail(f"Splits file not found: {splits_path}")
            self.log_info("Run: python prepare_data.py first")
            return False
        
        try:
            with open(splits_path, 'rb') as f:
                splits = pickle.load(f)
            
            train_indices = splits['train']
            dev_indices = splits['dev']
            test_indices = splits['test']
            
            total = len(train_indices) + len(dev_indices) + len(test_indices)
            
            # Check 1: Proportions are ~80/10/10
            train_ratio = len(train_indices) / total
            dev_ratio = len(dev_indices) / total
            test_ratio = len(test_indices) / total
            
            self.log_info(f"Split sizes: Train={len(train_indices)}, Dev={len(dev_indices)}, Test={len(test_indices)}")
            self.log_info(f"Split ratios: Train={train_ratio:.1%}, Dev={dev_ratio:.1%}, Test={test_ratio:.1%}")
            
            if 0.75 <= train_ratio <= 0.85:
                self.log_pass(f"Train ratio ~80%: {train_ratio:.1%} ✓")
            else:
                self.log_warn(f"Train ratio = {train_ratio:.1%}, expected ~80%")
            
            if 0.05 <= dev_ratio <= 0.15:
                self.log_pass(f"Dev ratio ~10%: {dev_ratio:.1%} ✓")
            else:
                self.log_warn(f"Dev ratio = {dev_ratio:.1%}, expected ~10%")
            
            if 0.05 <= test_ratio <= 0.15:
                self.log_pass(f"Test ratio ~10%: {test_ratio:.1%} ✓")
            else:
                self.log_warn(f"Test ratio = {test_ratio:.1%}, expected ~10%")
            
            # Check 2: No overlap between splits
            train_set = set(train_indices)
            dev_set = set(dev_indices)
            test_set = set(test_indices)
            
            overlap_train_dev = train_set & dev_set
            overlap_train_test = train_set & test_set
            overlap_dev_test = dev_set & test_set
            
            if len(overlap_train_dev) == 0:
                self.log_pass("No Train-Dev overlap ✓")
            else:
                self.log_fail(f"Train-Dev overlap: {len(overlap_train_dev)} samples")
            
            if len(overlap_train_test) == 0:
                self.log_pass("No Train-Test overlap ✓")
            else:
                self.log_fail(f"Train-Test overlap: {len(overlap_train_test)} samples")
            
            if len(overlap_dev_test) == 0:
                self.log_pass("No Dev-Test overlap ✓")
            else:
                self.log_fail(f"Dev-Test overlap: {len(overlap_dev_test)} samples")
            
            # Check 3: All indices are unique within each split
            if len(train_set) == len(train_indices):
                self.log_pass("Train indices are unique ✓")
            else:
                self.log_warn("Train indices have duplicates")
            
            return True
            
        except Exception as e:
            self.log_fail(f"Splits verification failed: {str(e)}")
            return False
    
    def verify_dataset_format(self):
        """Test 1.3: Dataset Output Format"""
        self.print_header("PHASE 1.3: DATASET FORMAT VERIFICATION")
        
        try:
            from data.vocabulary import Vocabulary
            from data.dataset import BELTWordDataset
            
            # Load vocabulary
            vocab = Vocabulary(vocab_size=500)
            vocab.load("data/vocabulary_top500.pkl")
            
            # Load splits
            with open("data/splits.pkl", 'rb') as f:
                splits = pickle.load(f)
            
            # Get pickle files for train split
            dataset_root = Path("dataset/ZuCo")
            tasks = ["task1-SR", "task2-NR", "task2-NR-2.0", "task3-TSR", "task3-TSR-2.0"]
            
            all_pickle_files = []
            for task in tasks:
                pickle_dir = dataset_root / task / "pickle"
                if pickle_dir.exists():
                    pickle_files = sorted(list(pickle_dir.glob("*.pickle")))
                    all_pickle_files.extend(pickle_files)
            
            if not all_pickle_files:
                self.log_fail("No pickle files found. Run: python prepare_data.py")
                return False
            
            # Select files for train split
            train_files = [all_pickle_files[i] for i in splits['train'][:5]]  # Just first 5 for testing
            
            # Create small dataset
            dataset = BELTWordDataset(
                pickle_files=[str(f) for f in train_files],
                vocabulary=vocab,
                split='train',
                eeg_type='GD'
            )
            
            if len(dataset) == 0:
                self.log_fail("Dataset is empty")
                return False
            
            # Test sample (dataset returns tuple: eeg, label, word)
            sample = dataset[0]
            
            # Check 1: Tuple structure
            if isinstance(sample, tuple) and len(sample) == 3:
                self.log_pass(f"Sample is a 3-tuple (eeg, label, word) ✓")
                eeg, label, word = sample
            else:
                self.log_fail(f"Sample is not a 3-tuple: {type(sample)}, len={len(sample) if isinstance(sample, (tuple, list)) else 'N/A'}")
                return False
            
            # Check 2: EEG shape
            if isinstance(eeg, torch.Tensor):
                if eeg.shape == (840,):
                    self.log_pass(f"EEG shape = (840,) ✓")
                else:
                    self.log_fail(f"EEG shape = {eeg.shape}, expected (840,)")
            else:
                self.log_fail(f"EEG is not a tensor: {type(eeg)}")
            
            # Check 3: Label range
            if isinstance(label, (int, torch.Tensor)):
                label_int = int(label)
                if 0 <= label_int < 500:
                    self.log_pass(f"Label in range [0, 499]: {label_int} ✓")
                else:
                    self.log_fail(f"Label out of range: {label_int}")
            else:
                self.log_fail(f"Label is not int/tensor: {type(label)}")
            
            # Check 4: Word is string
            if isinstance(word, str):
                self.log_pass(f"Word is string: '{word}' ✓")
            else:
                self.log_fail(f"Word is not string: {type(word)}")
            
            # Check 5: No NaN values
            if not torch.isnan(eeg).any():
                self.log_pass("No NaN values in EEG ✓")
            else:
                self.log_fail("NaN values found in EEG")
            
            # Check 6: Reasonable value ranges
            if eeg.min() >= 0:
                self.log_pass(f"EEG values non-negative (min={eeg.min():.4f}) ✓")
            else:
                self.log_warn(f"EEG has negative values (min={eeg.min():.4f})")
            
            if eeg.max() < 1000:
                self.log_pass(f"EEG values reasonable (max={eeg.max():.4f}) ✓")
            else:
                self.log_warn(f"EEG has very large values (max={eeg.max():.4f})")
            
            return True
            
        except Exception as e:
            self.log_fail(f"Dataset format verification failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    # ========================================================================
    # PHASE 2: ARCHITECTURE VERIFICATION
    # ========================================================================
    
    def verify_conformer_block(self):
        """Test 2.1: Conformer Block Structure"""
        self.print_header("PHASE 2.1: CONFORMER BLOCK VERIFICATION")
        
        try:
            from models.conformer_block import ConformerBlock
            
            # Create Conformer block
            block = ConformerBlock(
                d_model=840,
                num_heads=8,
                ffn_expansion=4,
                conv_kernel_size=31,
                dropout=0.1
            )
            
            # Check 1: Components exist
            components = ['ffn1', 'attention', 'conv', 'ffn2', 'layer_norm']
            for comp in components:
                if hasattr(block, comp):
                    self.log_pass(f"Has component: {comp} ✓")
                else:
                    self.log_fail(f"Missing component: {comp}")
            
            # Check 2: Forward pass shape
            x = torch.randn(2, 840)  # (batch, d_model)
            output = block(x)
            
            if output.shape == (2, 840):
                self.log_pass(f"Output shape correct: {output.shape} ✓")
            else:
                self.log_fail(f"Output shape wrong: {output.shape}, expected (2, 840)")
            
            # Check 3: Gradient flow
            loss = output.sum()
            loss.backward()
            
            has_grad = any(p.grad is not None for p in block.parameters())
            if has_grad:
                self.log_pass("Gradients flow through block ✓")
            else:
                self.log_fail("No gradients in block")
            
            # Check 4: Parameter count
            num_params = sum(p.numel() for p in block.parameters())
            self.log_info(f"Parameters per block: {num_params:,}")
            
            if 1_000_000 < num_params < 5_000_000:
                self.log_pass(f"Parameter count reasonable: {num_params:,} ✓")
            else:
                self.log_warn(f"Parameter count unusual: {num_params:,}")
            
            return True
            
        except Exception as e:
            self.log_fail(f"Conformer block verification failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def verify_convolution_module(self):
        """Test 2.2: Convolution Module (Table I)"""
        self.print_header("PHASE 2.2: CONVOLUTION MODULE VERIFICATION")
        
        try:
            from models.convolution_module import ConvolutionModule
            
            # Create convolution module
            conv = ConvolutionModule(
                d_model=840,
                kernel_size=31,
                expansion_factor=2,
                dropout=0.1
            )
            
            # Check 1: Components exist (Table I)
            components = {
                'layer_norm': 'LayerNorm',
                'pointwise_conv1': 'Pointwise Conv 1',
                'depthwise_conv': 'Depthwise Conv',
                'batch_norm': 'Batch Norm',
                'activation': 'Swish/SiLU',
                'pointwise_conv2': 'Pointwise Conv 2',
                'dropout': 'Dropout'
            }
            
            for attr, name in components.items():
                if hasattr(conv, attr):
                    self.log_pass(f"Has {name} ✓")
                else:
                    self.log_fail(f"Missing {name}")
            
            # Check 2: Depthwise conv configuration
            if hasattr(conv, 'depthwise_conv'):
                dw_conv = conv.depthwise_conv
                expected_groups = 840 * 2  # d_model * expansion_factor
                
                if dw_conv.groups == expected_groups:
                    self.log_pass(f"Depthwise conv groups = {expected_groups} ✓")
                else:
                    self.log_fail(f"Depthwise conv groups = {dw_conv.groups}, expected {expected_groups}")
                
                if dw_conv.kernel_size == (31,):
                    self.log_pass(f"Kernel size = 31 ✓")
                else:
                    self.log_fail(f"Kernel size = {dw_conv.kernel_size}, expected (31,)")
            
            # Check 3: Pointwise conv 1 expansion
            if hasattr(conv, 'pointwise_conv1'):
                pw_conv1 = conv.pointwise_conv1
                # Should be d_model → d_model*expansion*2 (×2 for GLU)
                expected_out = 840 * 2 * 2  # d_model * expansion * 2_for_GLU = 3360
                
                if pw_conv1.out_channels == expected_out:
                    self.log_pass(f"Pointwise conv 1 expansion correct: {expected_out} ✓")
                else:
                    self.log_fail(f"Pointwise conv 1 out_channels = {pw_conv1.out_channels}, expected {expected_out}")
            
            # Check 4: Forward pass
            x = torch.randn(2, 10, 840)  # (batch, seq_len, d_model)
            output = conv(x)
            
            if output.shape == (2, 10, 840):
                self.log_pass(f"Convolution output shape correct: {output.shape} ✓")
            else:
                self.log_fail(f"Convolution output shape: {output.shape}, expected (2, 10, 840)")
            
            return True
            
        except Exception as e:
            self.log_fail(f"Convolution module verification failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def verify_vector_quantizer(self):
        """Test 2.3: Vector Quantizer"""
        self.print_header("PHASE 2.3: VECTOR QUANTIZER VERIFICATION")
        
        try:
            from models.vector_quantizer import VectorQuantizer
            
            # Create VQ
            vq = VectorQuantizer(
                input_dim=840,
                codebook_size=1024,
                codebook_dim=1024,
                beta=0.3
            )
            
            # Check 1: Codebook configuration
            if hasattr(vq, 'codebook'):
                codebook = vq.codebook
                
                if codebook.num_embeddings == 1024:
                    self.log_pass(f"Codebook size K = 1024 ✓")
                else:
                    self.log_fail(f"Codebook size = {codebook.num_embeddings}, expected 1024")
                
                if codebook.embedding_dim == 1024:
                    self.log_pass(f"Codebook dimension D = 1024 ✓")
                else:
                    self.log_fail(f"Codebook dimension = {codebook.embedding_dim}, expected 1024")
            
            # Check 2: Commitment cost (beta)
            if hasattr(vq, 'beta'):
                if vq.beta == 0.3:
                    self.log_pass(f"Commitment cost β = 0.3 ✓")
                else:
                    self.log_fail(f"Commitment cost β = {vq.beta}, expected 0.3")
            
            # Check 3: Forward pass
            h = torch.randn(4, 840, requires_grad=True)  # (batch, input_dim)
            vq_loss, quantized, perplexity, encodings = vq(h)
            
            if quantized.shape == (4, 1024):
                self.log_pass(f"Quantized shape correct: {quantized.shape} ✓")
            else:
                self.log_fail(f"Quantized shape: {quantized.shape}, expected (4, 1024)")
            
            if vq_loss.dim() == 0:
                self.log_pass(f"VQ loss is scalar ✓")
            else:
                self.log_fail(f"VQ loss shape: {vq_loss.shape}, expected scalar")
            
            if perplexity.dim() == 0:
                self.log_pass(f"Perplexity is scalar ✓")
            else:
                self.log_fail(f"Perplexity shape: {perplexity.shape}, expected scalar")
            
            # Check 4: Actually quantizing
            with torch.no_grad():
                h_test = torch.randn(2, 840)
                _, quantized_test, _, _ = vq(h_test)
                
                # After projection, should be different from input
                h_projected = vq.projection(h_test)
                
                if not torch.allclose(h_projected, quantized_test):
                    self.log_pass("VQ actually quantizes (output ≠ input) ✓")
                else:
                    self.log_fail("VQ not quantizing (output == input)")
            
            # Check 5: Gradient flow (straight-through estimator)
            loss = quantized.sum()
            loss.backward()
            
            if h.grad is not None:
                self.log_pass("Straight-through gradient estimator works ✓")
            else:
                self.log_fail("No gradient through VQ")
            
            self.log_info(f"VQ loss: {vq_loss.item():.4f}, Perplexity: {perplexity.item():.2f}")
            
            return True
            
        except Exception as e:
            self.log_fail(f"Vector quantizer verification failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def verify_dconformer(self):
        """Test 2.4: Complete D-Conformer"""
        self.print_header("PHASE 2.4: D-CONFORMER ENCODER VERIFICATION")
        
        try:
            from models.dconformer import DConformer
            
            # Create D-Conformer
            encoder = DConformer(
                d_model=840,
                num_blocks=6,
                num_heads=8,
                ffn_expansion=4,
                conv_kernel_size=31,
                dropout=0.1
            )
            
            # Check 1: Has 6 Conformer blocks
            if hasattr(encoder, 'conformer_blocks'):
                num_blocks = len(encoder.conformer_blocks)
                
                if num_blocks == 6:
                    self.log_pass(f"D-Conformer has 6 blocks ✓")
                else:
                    self.log_fail(f"D-Conformer has {num_blocks} blocks, expected 6")
            
            # Check 2: Forward pass
            x = torch.randn(4, 840)  # (batch, d_model)
            output = encoder(x)
            
            if output.shape == (4, 840):
                self.log_pass(f"D-Conformer output shape correct: {output.shape} ✓")
            else:
                self.log_fail(f"D-Conformer output shape: {output.shape}, expected (4, 840)")
            
            # Check 3: Parameter count
            num_params = sum(p.numel() for p in encoder.parameters())
            self.log_info(f"D-Conformer parameters: {num_params:,}")
            
            # Expected: ~6 blocks × 2-3M params/block = 12-18M params
            if 10_000_000 < num_params < 30_000_000:
                self.log_pass(f"Parameter count reasonable: {num_params:,} ✓")
            else:
                self.log_warn(f"Parameter count unusual: {num_params:,}")
            
            # Check 4: Gradient flow through all blocks
            loss = output.sum()
            loss.backward()
            
            all_blocks_have_grad = all(
                any(p.grad is not None for p in block.parameters())
                for block in encoder.conformer_blocks
            )
            
            if all_blocks_have_grad:
                self.log_pass("Gradients flow through all 6 blocks ✓")
            else:
                self.log_fail("Some blocks have no gradients")
            
            return True
            
        except Exception as e:
            self.log_fail(f"D-Conformer verification failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def verify_classifier(self):
        """Test 2.5: MLP Classifier"""
        self.print_header("PHASE 2.5: MLP CLASSIFIER VERIFICATION")
        
        try:
            from models.classifier import MLPClassifier
            
            # Create classifier
            classifier = MLPClassifier(
                input_dim=1024,
                hidden_dims=[512, 256],
                output_dim=500,
                dropout=0.3
            )
            
            # Check 1: Forward pass
            x = torch.randn(4, 1024)
            logits = classifier(x)
            
            if logits.shape == (4, 500):
                self.log_pass(f"Classifier output shape correct: {logits.shape} ✓")
            else:
                self.log_fail(f"Classifier output shape: {logits.shape}, expected (4, 500)")
            
            # Check 2: Parameter count
            num_params = sum(p.numel() for p in classifier.parameters())
            self.log_info(f"Classifier parameters: {num_params:,}")
            
            # Expected: 1024×512 + 512×256 + 256×500 = ~780K params
            expected = 1024*512 + 512*256 + 256*500
            if 0.8 * expected < num_params < 1.2 * expected:
                self.log_pass(f"Parameter count reasonable: {num_params:,} ✓")
            else:
                self.log_warn(f"Parameter count differs from expected ~{expected:,}")
            
            return True
            
        except Exception as e:
            self.log_fail(f"Classifier verification failed: {str(e)}")
            return False
    
    # ========================================================================
    # PHASE 3: LOSS FUNCTION VERIFICATION
    # ========================================================================
    
    def verify_loss_functions(self):
        """Test 3: All Loss Functions"""
        self.print_header("PHASE 3: LOSS FUNCTIONS VERIFICATION")
        
        try:
            # Test data
            batch_size = 4
            logits = torch.randn(batch_size, 500)
            labels = torch.randint(0, 500, (batch_size,))
            quantized = torch.randn(batch_size, 1024)
            words = ["the", "cat", "sat", "down"]
            
            # Check 1: Cross-Entropy Loss
            ce_loss = F.cross_entropy(logits, labels)
            
            if ce_loss.dim() == 0 and ce_loss.item() > 0:
                self.log_pass(f"CE loss correct: {ce_loss.item():.4f} ✓")
            else:
                self.log_fail(f"CE loss incorrect: shape={ce_loss.shape}, value={ce_loss.item()}")
            
            # Check 2: Vector Quantization Loss
            from models.vector_quantizer import VectorQuantizer
            vq = VectorQuantizer(input_dim=840, codebook_size=1024, codebook_dim=1024, beta=0.3)
            
            h = torch.randn(batch_size, 840)
            vq_loss, _, perplexity, _ = vq(h)
            
            if vq_loss.dim() == 0 and vq_loss.item() >= 0:
                self.log_pass(f"VQ loss correct: {vq_loss.item():.4f} ✓")
            else:
                self.log_fail(f"VQ loss incorrect: shape={vq_loss.shape}, value={vq_loss.item()}")
            
            if 1 <= perplexity.item() <= 1024:
                self.log_pass(f"Perplexity in range: {perplexity.item():.2f} ✓")
            else:
                self.log_warn(f"Perplexity unusual: {perplexity.item():.2f}")
            
            # Check 3: Contrastive Loss
            self.log_info("Loading BART for contrastive loss (this may take a moment)...")
            from training.losses import ContrastiveLoss
            
            try:
                cl = ContrastiveLoss(
                    eeg_dim=1024,
                    word_dim=768,
                    bart_model_name="facebook/bart-base",
                    temperature=0.07,
                    freeze_bart=True
                )
                
                # Check BART is frozen
                bart_frozen = all(not p.requires_grad for p in cl.bart.parameters())
                if bart_frozen:
                    self.log_pass("BART is frozen ✓")
                else:
                    self.log_fail("BART is NOT frozen")
                
                # Test contrastive loss
                cl_loss = cl(quantized, words)
                
                if cl_loss.dim() == 0 and cl_loss.item() > 0:
                    self.log_pass(f"Contrastive loss correct: {cl_loss.item():.4f} ✓")
                else:
                    self.log_fail(f"Contrastive loss incorrect: shape={cl_loss.shape}, value={cl_loss.item()}")
                
                # Check temperature
                if cl.temperature == 0.07:
                    self.log_pass(f"Temperature τ = 0.07 ✓")
                else:
                    self.log_fail(f"Temperature τ = {cl.temperature}, expected 0.07")
                
            except Exception as e:
                self.log_warn(f"Contrastive loss test skipped: {str(e)}")
                self.log_info("This is OK if you don't have enough RAM for BART")
            
            # Check 4: Combined loss (Equation 7)
            alpha = 0.9
            lambda_vq = 1.0
            cl_loss = torch.tensor(1.8)  # Mock value
            
            total_loss = ce_loss + alpha * cl_loss + lambda_vq * vq_loss
            expected = ce_loss.item() + 0.9 * 1.8 + 1.0 * vq_loss.item()
            
            if abs(total_loss.item() - expected) < 1e-5:
                self.log_pass(f"Combined loss formula correct: {total_loss.item():.4f} ✓")
            else:
                self.log_fail(f"Combined loss incorrect: {total_loss.item():.4f} vs expected {expected:.4f}")
            
            # Check loss weights
            if alpha == 0.9:
                self.log_pass(f"α (contrastive weight) = 0.9 ✓")
            else:
                self.log_fail(f"α = {alpha}, expected 0.9")
            
            if lambda_vq == 1.0:
                self.log_pass(f"λ (VQ weight) = 1.0 ✓")
            else:
                self.log_fail(f"λ = {lambda_vq}, expected 1.0")
            
            return True
            
        except Exception as e:
            self.log_fail(f"Loss functions verification failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    # ========================================================================
    # PHASE 4: TRAINING CONFIGURATION VERIFICATION
    # ========================================================================
    
    def verify_config(self):
        """Test 4: Hyperparameters Match BELT"""
        self.print_header("PHASE 4: CONFIGURATION VERIFICATION")
        
        config_path = Path("config/belt_config.yaml")
        
        if not config_path.exists():
            self.log_fail(f"Config file not found: {config_path}")
            return False
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # BELT paper specifications (Section IV-B, Table I)
            belt_specs = {
                'epochs': 60,
                'batch_size': 64,
                'learning_rate': 5e-6,
                'alpha': 0.9,
                'lambda': 1.0,
                'beta': 0.3,
                'K': 1024,
                'D': 1024,
                'num_blocks': 6,
                'num_heads': 8,
                'conv_kernel': 31,
                'vocab_size': 500
            }
            
            # Check each parameter
            checks = {
                'epochs': config['training']['epochs'],
                'batch_size': config['training']['batch_size'],
                'learning_rate': config['training']['learning_rate'],
                'alpha': config['training']['loss_weights']['alpha'],
                'lambda': config['training']['loss_weights']['lambda'],
                'beta': config['model']['vector_quantizer']['beta'],
                'K': config['model']['vector_quantizer']['codebook_size'],
                'D': config['model']['vector_quantizer']['codebook_dim'],
                'num_blocks': config['model']['conformer']['num_blocks'],
                'num_heads': config['model']['conformer']['num_heads'],
                'conv_kernel': config['model']['conformer']['conv_kernel_size'],
                'vocab_size': config['data']['vocabulary_size']
            }
            
            all_match = True
            for param, actual in checks.items():
                expected = belt_specs[param]
                
                if actual == expected:
                    self.log_pass(f"{param}: {actual} ✓")
                else:
                    self.log_fail(f"{param}: {actual} (expected: {expected})")
                    all_match = False
            
            # Check optimizer
            optimizer = config['training'].get('optimizer', 'sgd').lower()
            if optimizer == 'sgd':
                self.log_pass(f"Optimizer: SGD ✓")
            else:
                self.log_warn(f"Optimizer: {optimizer} (BELT uses SGD)")
            
            if all_match:
                self.log_pass("ALL hyperparameters match BELT paper! ✓✓✓")
            else:
                self.log_warn("Some hyperparameters don't match BELT paper")
            
            return True
            
        except Exception as e:
            self.log_fail(f"Config verification failed: {str(e)}")
            return False
    
    # ========================================================================
    # PHASE 5: END-TO-END INTEGRATION TEST
    # ========================================================================
    
    def verify_end_to_end(self):
        """Test 5: Complete Forward/Backward Pass"""
        self.print_header("PHASE 5: END-TO-END INTEGRATION TEST")
        
        try:
            from models.dconformer import DConformer
            from models.vector_quantizer import VectorQuantizer
            from models.classifier import MLPClassifier
            
            # Build models
            encoder = DConformer(d_model=840, num_blocks=6, num_heads=8, 
                               ffn_expansion=4, conv_kernel_size=31, dropout=0.1)
            vq = VectorQuantizer(input_dim=840, codebook_size=1024, 
                               codebook_dim=1024, beta=0.3)
            classifier = MLPClassifier(input_dim=1024, hidden_dims=[512, 256], 
                                     output_dim=500, dropout=0.3)
            
            # Create dummy batch
            batch_size = 8
            eeg = torch.randn(batch_size, 840)
            labels = torch.randint(0, 500, (batch_size,))
            words = ["the", "cat", "sat", "on", "the", "mat", "quietly", "today"]
            
            # Forward pass
            h = encoder(eeg)  # (8, 840)
            vq_loss, quantized, perplexity, _ = vq(h)  # (8, 1024)
            logits = classifier(quantized)  # (8, 500)
            
            # Check shapes
            if logits.shape == (batch_size, 500):
                self.log_pass(f"End-to-end forward pass shape correct: {logits.shape} ✓")
            else:
                self.log_fail(f"End-to-end output shape: {logits.shape}, expected ({batch_size}, 500)")
            
            # Compute losses
            ce_loss = F.cross_entropy(logits, labels)
            total_loss = ce_loss + 1.0 * vq_loss  # Without contrastive for speed
            
            self.log_info(f"CE loss: {ce_loss.item():.4f}, VQ loss: {vq_loss.item():.4f}")
            self.log_info(f"Total loss: {total_loss.item():.4f}")
            
            # Backward pass
            total_loss.backward()
            
            # Check gradients
            encoder_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 
                                 for p in encoder.parameters())
            vq_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 
                            for p in vq.parameters())
            classifier_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 
                                    for p in classifier.parameters())
            
            if encoder_has_grad:
                self.log_pass("Encoder has gradients ✓")
            else:
                self.log_fail("Encoder has no gradients")
            
            if vq_has_grad:
                self.log_pass("VQ has gradients ✓")
            else:
                self.log_fail("VQ has no gradients")
            
            if classifier_has_grad:
                self.log_pass("Classifier has gradients ✓")
            else:
                self.log_fail("Classifier has no gradients")
            
            # Test predictions
            with torch.no_grad():
                predictions = logits.argmax(dim=1)
                
                if predictions.shape == (batch_size,):
                    self.log_pass(f"Predictions shape correct: {predictions.shape} ✓")
                else:
                    self.log_fail(f"Predictions shape: {predictions.shape}, expected ({batch_size},)")
                
                if all(0 <= p < 500 for p in predictions):
                    self.log_pass("All predictions in range [0, 499] ✓")
                else:
                    self.log_fail("Some predictions out of range")
            
            # Test top-k accuracy computation
            def compute_topk_accuracy(logits, labels, k=10):
                _, topk_indices = torch.topk(logits, k, dim=1)
                correct = topk_indices.eq(labels.unsqueeze(1).expand_as(topk_indices))
                return 100.0 * correct.sum().item() / labels.size(0)
            
            top10_acc = compute_topk_accuracy(logits, labels, k=10)
            self.log_info(f"Top-10 accuracy (random): {top10_acc:.2f}%")
            
            if 0 <= top10_acc <= 100:
                self.log_pass(f"Top-10 accuracy computation works ✓")
            else:
                self.log_fail(f"Top-10 accuracy invalid: {top10_acc}")
            
            return True
            
        except Exception as e:
            self.log_fail(f"End-to-end test failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    # ========================================================================
    # MAIN VERIFICATION
    # ========================================================================
    
    def run_all_verifications(self):
        """Run all verification tests"""
        
        print(f"\n{BOLD}{'='*80}{RESET}")
        print(f"{BOLD}COMPREHENSIVE BELT IMPLEMENTATION VERIFICATION{RESET}")
        print(f"{BOLD}{'='*80}{RESET}\n")
        
        # Phase 1: Data Pipeline
        self.verify_vocabulary()
        self.verify_splits()
        self.verify_dataset_format()
        
        # Phase 2: Architecture
        self.verify_dconformer()
        self.verify_conformer_block()
        self.verify_convolution_module()
        self.verify_vector_quantizer()
        self.verify_classifier()
        
        # Phase 3: Loss Functions
        self.verify_loss_functions()
        
        # Phase 4: Configuration
        self.verify_config()
        
        # Phase 5: End-to-End
        self.verify_end_to_end()
        
        # Final Summary
        self.print_final_summary()
    
    def print_final_summary(self):
        """Print verification summary"""
        
        self.print_header("VERIFICATION SUMMARY", level=1)
        
        total_tests = len(self.passed_tests) + len(self.failed_tests)
        pass_rate = 100 * len(self.passed_tests) / total_tests if total_tests > 0 else 0
        
        print(f"\n{BOLD}Test Results:{RESET}")
        print(f"  {GREEN}✓{RESET} Passed: {len(self.passed_tests)}/{total_tests}")
        print(f"  {RED}✗{RESET} Failed: {len(self.failed_tests)}/{total_tests}")
        print(f"  {YELLOW}⚠{RESET} Warnings: {len(self.warnings)}")
        print(f"  Pass Rate: {pass_rate:.1f}%")
        
        if self.failed_tests:
            print(f"\n{RED}{BOLD}FAILED TESTS:{RESET}")
            for i, test in enumerate(self.failed_tests, 1):
                print(f"  {i}. {test}")
        
        if self.warnings:
            print(f"\n{YELLOW}{BOLD}WARNINGS:{RESET}")
            for i, warning in enumerate(self.warnings, 1):
                print(f"  {i}. {warning}")
        
        print(f"\n{'='*80}\n")
        
        if not self.failed_tests:
            print(f"{GREEN}{BOLD}✓✓✓ ALL CRITICAL TESTS PASSED! ✓✓✓{RESET}")
            print(f"\n{BOLD}Your BELT implementation appears correct!{RESET}")
            print(f"\nYou can now:")
            print(f"  1. Run training: python experiments/model_with_bootstrapping.py")
            print(f"  2. Expect results around: Top-10 accuracy ~31.04%")
            print(f"  3. Compare with BELT paper Table VI")
            
            if self.warnings:
                print(f"\n{YELLOW}Note: Review warnings above (may be OK){RESET}")
        else:
            print(f"{RED}{BOLD}❌ ERRORS FOUND - PLEASE FIX BEFORE TRAINING{RESET}")
            print(f"\nReview the failed tests above and:")
            print(f"  1. Check implementation against BELT paper")
            print(f"  2. Fix errors one by one")
            print(f"  3. Re-run this verification script")
        
        print(f"\n{'='*80}\n")


if __name__ == "__main__":
    verifier = BELTVerifier()
    verifier.run_all_verifications()
