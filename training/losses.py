"""
Loss Functions for BELT
Implements all three losses from Equation 7:
L = L_ce + α*L_cl^w + λ*L_vq
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BartModel, BartTokenizer
from typing import Optional


class ContrastiveLoss(nn.Module):
    """
    Word-level contrastive loss (L_cl^w from Equation 3)
    
    Uses InfoNCE loss with BART word embeddings as guidance
    """
    
    def __init__(
        self,
        eeg_dim: int = 1024,
        word_dim: int = 768,
        bart_model_name: str = "facebook/bart-base",
        temperature: float = 0.07,
        freeze_bart: bool = True
    ):
        """
        Args:
            eeg_dim: Dimension of EEG features (from VQ, default: 1024)
            word_dim: BART embedding dimension (default: 768)
            bart_model_name: BART model to use
            temperature: Temperature τ for InfoNCE (default: 0.07)
            freeze_bart: Whether to freeze BART parameters
        """
        super().__init__()
        
        self.temperature = temperature
        self.eeg_dim = eeg_dim
        self.word_dim = word_dim
        
        # Load BART model and tokenizer
        print(f"Loading BART model: {bart_model_name}")
        self.bart = BartModel.from_pretrained(bart_model_name)
        self.tokenizer = BartTokenizer.from_pretrained(bart_model_name)
        
        # Freeze BART if requested
        if freeze_bart:
            for param in self.bart.parameters():
                param.requires_grad = False
            self.bart.eval()
        
        # Projection layers
        self.eeg_projection = nn.Linear(eeg_dim, word_dim)
        self.word_projection = nn.Linear(word_dim, word_dim)
        
    def get_bart_embeddings(self, words: list) -> torch.Tensor:
        """
        Get BART embeddings for a list of words
        
        Args:
            words: List of word strings
            
        Returns:
            embeddings: (batch, word_dim)
        """
        # Tokenize words (just the word, not full sentences)
        tokens = self.tokenizer(
            words,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=5  # Words are short
        )
        
        # Move to same device as model
        device = next(self.bart.parameters()).device
        tokens = {k: v.to(device) for k, v in tokens.items()}
        
        # Get BART embeddings (use <s> token embedding)
        with torch.no_grad() if next(self.bart.parameters()).requires_grad == False else torch.enable_grad():
            # Get input embeddings directly
            input_embeds = self.bart.get_input_embeddings()
            word_ids = tokens['input_ids']  # (batch, seq_len)
            embeddings = input_embeds(word_ids)  # (batch, seq_len, word_dim)
            
            # FIX: Exclude special tokens <s> (id=0) and </s> (id=2) from averaging
            # BART tokenizer: <s>=0, </s>=2, <pad>=1
            # Only average the actual word tokens (not special tokens)
            special_token_mask = (word_ids == 0) | (word_ids == 2) | (word_ids == 1)
            content_mask = (~special_token_mask).float().unsqueeze(-1)  # (batch, seq_len, 1)
            
            # Average only content tokens
            masked_embeds = embeddings * content_mask
            sum_embeds = masked_embeds.sum(1)  # (batch, word_dim)
            count = content_mask.sum(1).clamp(min=1)  # (batch, 1) avoid div by zero
            embeddings = sum_embeds / count
        
        return embeddings
    
    def forward(
        self,
        eeg_features: torch.Tensor,
        words: list
    ) -> torch.Tensor:
        """
        Compute contrastive loss (Equation 3)
        
        Args:
            eeg_features: (batch, eeg_dim) - quantized EEG features
            words: List of word strings (length = batch)
            
        Returns:
            loss: scalar tensor
        """
        batch_size = eeg_features.size(0)
        
        # Get BART word embeddings
        word_embeddings = self.get_bart_embeddings(words)  # (batch, word_dim)
        
        # Project both to common space
        eeg_proj = self.eeg_projection(eeg_features)  # (batch, word_dim)
        word_proj = self.word_projection(word_embeddings)  # (batch, word_dim)
        
        # L2 normalize
        eeg_proj = F.normalize(eeg_proj, p=2, dim=1)
        word_proj = F.normalize(word_proj, p=2, dim=1)
        
        # Compute similarity matrix: (batch, batch)
        # similarity[i,j] = eeg_proj[i] · word_proj[j]
        similarity = torch.matmul(eeg_proj, word_proj.t()) / self.temperature
        
        # FIX: Handle duplicate words in batch (common for "the", "and", etc.)
        # Create mask where True = same word (should not be treated as negative)
        word_match_mask = torch.zeros((batch_size, batch_size), dtype=torch.bool, device=similarity.device)
        for i in range(batch_size):
            for j in range(batch_size):
                if i != j and words[i] == words[j]:
                    word_match_mask[i, j] = True
        
        # Mask out false negatives: set their logits to very negative value
        # So they don't contribute to the denominator of InfoNCE
        if word_match_mask.any():
            similarity = similarity.masked_fill(word_match_mask, -1e9)
        
        # InfoNCE loss: maximize similarity on diagonal
        # L = -log(exp(sim[i,i]) / sum_j(exp(sim[i,j])))
        labels = torch.arange(batch_size, device=similarity.device)
        loss = F.cross_entropy(similarity, labels)
        
        return loss


class BELTLosses:
    """
    Combined loss computation for BELT
    L = L_ce + α*L_cl^w + λ*L_vq
    """
    
    def __init__(
        self,
        alpha: float = 0.9,
        lambda_vq: float = 1.0,
        use_contrastive: bool = True,
        contrastive_loss: Optional[ContrastiveLoss] = None
    ):
        """
        Args:
            alpha: Weight for contrastive loss (default: 0.9)
            lambda_vq: Weight for VQ loss (default: 1.0)
            use_contrastive: Whether to use contrastive loss (False for ablation)
            contrastive_loss: Contrastive loss module (optional)
        """
        self.alpha = alpha
        self.lambda_vq = lambda_vq
        self.use_contrastive = use_contrastive
        self.contrastive_loss = contrastive_loss
        
        # Cross-entropy loss
        self.ce_loss = nn.CrossEntropyLoss()
    
    def compute_total_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        vq_loss: torch.Tensor,
        eeg_features: Optional[torch.Tensor] = None,
        words: Optional[list] = None
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute total loss (Equation 7)
        
        Args:
            logits: (batch, vocab_size) - classifier outputs
            labels: (batch,) - ground truth labels
            vq_loss: scalar - VQ loss from quantizer
            eeg_features: (batch, eeg_dim) - for contrastive loss
            words: List of word strings - for contrastive loss
            
        Returns:
            total_loss: scalar tensor
            loss_dict: Dictionary with individual loss values
        """
        # L_ce: Cross-entropy loss
        l_ce = self.ce_loss(logits, labels)
        
        # L_vq: Vector quantization loss
        l_vq = vq_loss
        
        # Start with CE + VQ
        total_loss = l_ce + self.lambda_vq * l_vq
        
        loss_dict = {
            'L_ce': l_ce.item(),
            'L_vq': l_vq.item(),
            'L_total': total_loss.item()
        }
        
        # L_cl^w: Contrastive loss (optional)
        if self.use_contrastive and self.contrastive_loss is not None:
            if eeg_features is None or words is None:
                raise ValueError("eeg_features and words required for contrastive loss")
            
            l_cl = self.contrastive_loss(eeg_features, words)
            total_loss = total_loss + self.alpha * l_cl
            
            loss_dict['L_cl'] = l_cl.item()
            loss_dict['L_total'] = total_loss.item()
        else:
            loss_dict['L_cl'] = 0.0
        
        return total_loss, loss_dict


if __name__ == "__main__":
    """Test loss functions"""
    
    print("="*80)
    print("TESTING BELT LOSS FUNCTIONS")
    print("="*80)
    
    # Test Cross-Entropy Loss
    print("\nTest 1: Cross-Entropy Loss")
    ce_loss = nn.CrossEntropyLoss()
    logits = torch.randn(32, 500)
    labels = torch.randint(0, 500, (32,))
    loss = ce_loss(logits, labels)
    print(f"  Logits shape: {logits.shape}")
    print(f"  Labels shape: {labels.shape}")
    print(f"  L_ce: {loss.item():.4f}")
    
    # Test VQ Loss (simulated)
    print("\nTest 2: VQ Loss (simulated)")
    vq_loss = torch.tensor(0.5)
    print(f"  L_vq: {vq_loss.item():.4f}")
    
    # Test Contrastive Loss
    print("\nTest 3: Contrastive Loss")
    print("  (This will download BART model, may take a moment...)")
    contrastive = ContrastiveLoss(
        eeg_dim=1024,
        word_dim=768,
        temperature=0.07,
        freeze_bart=True
    )
    
    eeg_features = torch.randn(8, 1024)
    words = ["the", "brain", "activity", "reading", "word", "neural", "signal", "processing"]
    
    cl_loss = contrastive(eeg_features, words)
    print(f"  EEG features shape: {eeg_features.shape}")
    print(f"  Words: {words}")
    print(f"  L_cl: {cl_loss.item():.4f}")
    
    # Test Combined Loss (with contrastive)
    print("\nTest 4: Combined Loss (Full BELT)")
    belt_losses = BELTLosses(
        alpha=0.9,
        lambda_vq=1.0,
        use_contrastive=True,
        contrastive_loss=contrastive
    )
    
    total_loss, loss_dict = belt_losses.compute_total_loss(
        logits=logits[:8],
        labels=labels[:8],
        vq_loss=vq_loss,
        eeg_features=eeg_features,
        words=words
    )
    
    print(f"  Total loss: {total_loss.item():.4f}")
    print(f"  Loss breakdown:")
    for key, value in loss_dict.items():
        print(f"    {key}: {value:.4f}")
    
    # Test Combined Loss (ablation - no contrastive)
    print("\nTest 5: Combined Loss (Ablation - No Contrastive)")
    belt_losses_ablation = BELTLosses(
        alpha=0.9,
        lambda_vq=1.0,
        use_contrastive=False
    )
    
    total_loss, loss_dict = belt_losses_ablation.compute_total_loss(
        logits=logits[:8],
        labels=labels[:8],
        vq_loss=vq_loss
    )
    
    print(f"  Total loss: {total_loss.item():.4f}")
    print(f"  Loss breakdown:")
    for key, value in loss_dict.items():
        print(f"    {key}: {value:.4f}")
    
    print("\n[OK] Loss functions test passed!")
