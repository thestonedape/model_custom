"""
BELT-Enhanced training/evaluation script.

This script provides a single, explicit train/eval pipeline and avoids duplicate
control-flow paths.
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from data.sentence_dataset import BELTSentenceDataset, load_sentence_splits
from data.vocabulary import Vocabulary
from models.dconformer import DConformer
from models.vector_quantizer import VectorQuantizer
from training.augmentation import MixUp, mixup_criterion
from training.enhanced_losses import FocalLoss, LabelSmoothingCrossEntropy
from training.losses import ContrastiveLoss
from training.metrics import TopKAccuracyTracker
from training.regularization import LinearScheduleDropPath, MultiSampleDropoutClassifier
from training.schedulers import WarmupCosineSchedule


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class BELTEnhancedModel(nn.Module):
    """Enhanced BELT model."""

    def __init__(self, config):
        super().__init__()
        model_cfg = config["model"]

        self.encoder = DConformer(
            d_model=model_cfg["encoder"]["d_model"],
            num_blocks=model_cfg["encoder"]["num_layers"],
            num_heads=model_cfg["encoder"]["num_heads"],
            ffn_expansion=model_cfg["encoder"]["ff_expansion"],
            conv_kernel_size=model_cfg["encoder"]["conv_kernel_size"],
            dropout=model_cfg["encoder"]["dropout"],
        )

        if model_cfg["encoder"].get("use_drop_path", False):
            self._add_drop_path_to_encoder(model_cfg["encoder"]["drop_path_rate"])

        self.vq = VectorQuantizer(
            input_dim=model_cfg["encoder"]["d_model"],
            codebook_size=model_cfg["vq"]["codebook_size"],
            codebook_dim=model_cfg["vq"]["codebook_dim"],
            beta=model_cfg["vq"]["beta"],
        )

        input_dim = model_cfg["vq"]["codebook_dim"]
        hidden_dims = model_cfg["classifier"]["hidden_dims"]
        num_classes = model_cfg["classifier"]["num_classes"]

        dims = [input_dim] + hidden_dims
        layers = []
        for i in range(len(dims) - 1):
            layers.extend([nn.Linear(dims[i], dims[i + 1]), nn.ReLU(), nn.Dropout(0.5)])
        layers.append(nn.Linear(dims[-1], num_classes))
        base_classifier = nn.Sequential(*layers)

        if model_cfg["classifier"].get("use_multi_sample_dropout", False):
            self.classifier = MultiSampleDropoutClassifier(
                base_classifier=base_classifier,
                dropout_p=0.5,
                num_samples=model_cfg["classifier"]["multi_sample_num"],
            )
        else:
            self.classifier = base_classifier

    def _add_drop_path_to_encoder(self, drop_path_rate: float):
        """Attach per-layer scheduled DropPath to Conformer blocks."""
        num_layers = len(self.encoder.conformer_blocks)
        for i, layer in enumerate(self.encoder.conformer_blocks):
            layer.drop_path = LinearScheduleDropPath(
                drop_prob_max=drop_path_rate,
                layer_idx=i,
                num_layers=num_layers,
            )

    def forward(self, x, return_vq_loss=False, use_vq=True):
        """Forward pass."""
        encoded = self.encoder(x)
        pooled = encoded.mean(dim=1) if encoded.dim() == 3 else encoded

        if use_vq:
            vq_loss, quantized, _, _ = self.vq(pooled)
        else:
            vq_loss = torch.tensor(0.0, device=x.device)
            quantized = pooled

        logits = self.classifier(quantized)
        if return_vq_loss:
            return logits, vq_loss, quantized
        return logits


def build_device(config):
    """Select and print device information."""
    device = torch.device("cuda" if torch.cuda.is_available() and config["device"]["use_cuda"] else "cpu")
    print(f"\n[Device] Using: {device}")
    if device.type == "cuda":
        print(f"[CUDA] Device count: {torch.cuda.device_count()}")
        print(f"[CUDA] Current device: {torch.cuda.current_device()}")
        print(f"[CUDA] Device name: {torch.cuda.get_device_name(device)}")
    return device


def build_dataloaders(config, vocab, splits):
    """Create train/val/test dataloaders from sentence-level splits."""
    print("\n[Data] Creating datasets from sentence-level splits...")

    train_dataset = BELTSentenceDataset(
        sentence_list=splits["train"],
        vocabulary=vocab,
        split="train",
        eeg_type="GD",
    )
    val_dataset = BELTSentenceDataset(
        sentence_list=splits["dev"],
        vocabulary=vocab,
        split="dev",
        eeg_type="GD",
    )
    test_dataset = BELTSentenceDataset(
        sentence_list=splits["test"],
        vocabulary=vocab,
        split="test",
        eeg_type="GD",
    )

    train_batch_size = config["training"].get("batch_size", config["data"]["batch_size"])
    eval_batch_size = config["evaluation"]["batch_size"]
    num_workers = config["data"]["num_workers"]
    pin_memory = config["data"]["pin_memory"]

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    print(
        f"[Data] Samples - train: {len(train_dataset):,}, val: {len(val_dataset):,}, "
        f"test: {len(test_dataset):,}"
    )
    return train_loader, val_loader, test_loader


def build_optimizer_and_scheduler(model, config):
    """Build AdamW optimizer and warmup-cosine scheduler."""
    optimizer_cfg = config["training"]["optimizer"]
    scheduler_cfg = config["training"]["scheduler"]

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(optimizer_cfg["lr"]),
        weight_decay=float(optimizer_cfg["weight_decay"]),
        betas=tuple(float(b) for b in optimizer_cfg["betas"]),
        eps=float(optimizer_cfg["eps"]),
    )
    scheduler = WarmupCosineSchedule(
        optimizer=optimizer,
        warmup_epochs=scheduler_cfg["warmup_epochs"],
        total_epochs=config["training"]["num_epochs"],
        min_lr=float(scheduler_cfg["min_lr"]),
    )
    return optimizer, scheduler


def build_loss_functions(config, device):
    """Build CE/focal/label-smoothing and optional contrastive losses."""
    loss_cfg = config["training"]["loss"]

    if loss_cfg.get("use_label_smoothing", False):
        criterion_ce = LabelSmoothingCrossEntropy(epsilon=loss_cfg["label_smoothing"])
        print(f"[Loss] Label smoothing enabled (eps={loss_cfg['label_smoothing']})")
    elif loss_cfg.get("use_focal_loss", False):
        criterion_ce = FocalLoss(alpha=loss_cfg["focal_alpha"], gamma=loss_cfg["focal_gamma"])
        print(f"[Loss] Focal loss enabled (alpha={loss_cfg['focal_alpha']}, gamma={loss_cfg['focal_gamma']})")
    else:
        criterion_ce = nn.CrossEntropyLoss()
        print("[Loss] Standard CrossEntropy enabled")

    alpha = float(loss_cfg["alpha"])
    if alpha > 0:
        emb_cfg = config.get("data", {}).get("word_embeddings", {})
        bart_model_name = emb_cfg.get("model_name", "facebook/bart-base")
        word_dim = int(emb_cfg.get("embedding_dim", 768))
        eeg_dim = int(config["model"]["vq"]["codebook_dim"])
        freeze_bart = bool(emb_cfg.get("freeze_bart", True))
        criterion_cl = ContrastiveLoss(
            eeg_dim=eeg_dim,
            word_dim=word_dim,
            bart_model_name=bart_model_name,
            temperature=float(loss_cfg["temperature"]),
            freeze_bart=freeze_bart,
        ).to(device)
        print(f"[Loss] Contrastive enabled (alpha={alpha}, bart={bart_model_name})")
    else:
        criterion_cl = None
        print("[Loss] Contrastive disabled (alpha=0.0)")

    return {"ce": criterion_ce, "cl": criterion_cl, "vq": None}


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None, device="cpu"):
    """Load checkpoint and optionally optimizer/scheduler state."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    epoch = int(checkpoint.get("epoch", 0))
    best_val_acc = float(checkpoint.get("best_val_acc", checkpoint.get("val_results", {}).get("top10_acc", 0.0)))
    return epoch, best_val_acc


def save_checkpoint(path, epoch, model, optimizer, scheduler, val_results, best_val_acc):
    """Save a checkpoint with full training state."""
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_results": val_results,
            "best_val_acc": best_val_acc,
        },
        path,
    )


def train_epoch(model, dataloader, optimizer, scheduler, criterion_dict, device, config, epoch):
    """Train for one epoch."""
    model.train()

    criterion_ce = criterion_dict["ce"]
    criterion_cl = criterion_dict["cl"]
    alpha = float(config["training"]["loss"]["alpha"])
    lambda_vq = float(config["training"]["loss"]["lambda_vq"])

    use_mixup = config["data"].get("use_mixup", False)
    mixup = MixUp(alpha=config["data"]["mixup_alpha"], prob=config["data"]["mixup_prob"]) if use_mixup else None

    total_loss = 0.0
    num_batches = 0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for eeg_data, labels, words in pbar:
        eeg_data = eeg_data.to(device)
        labels = labels.to(device)

        if use_mixup:
            eeg_data, labels_a, labels_b, lam = mixup(eeg_data, labels)

        logits, vq_loss, quantized = model(eeg_data, return_vq_loss=True, use_vq=True)

        if use_mixup:
            loss_ce = mixup_criterion(criterion_ce, logits, labels_a, labels_b, lam)
        else:
            loss_ce = criterion_ce(logits, labels)

        if alpha > 0:
            if criterion_cl is None:
                raise RuntimeError("alpha > 0 but contrastive loss is not initialized")
            loss_cl = criterion_cl(quantized.to(device), list(words))
        else:
            loss_cl = torch.tensor(0.0, device=device)

        loss = loss_ce + alpha * loss_cl + lambda_vq * vq_loss

        optimizer.zero_grad()
        loss.backward()

        if config["training"]["gradient_clipping"]["enabled"]:
            max_norm = config["training"]["gradient_clipping"]["max_norm"]
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        optimizer.step()

        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix(
            {
                "loss": f"{loss.item():.4f}",
                "ce": f"{loss_ce.item():.4f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
            }
        )

    scheduler.step()
    return total_loss / max(num_batches, 1)


@torch.no_grad()
def evaluate(model, dataloader, criterion_dict, device):
    """Evaluate on a dataloader."""
    model.eval()
    criterion_ce = criterion_dict["ce"]
    metrics = TopKAccuracyTracker(k_values=[1, 3, 5, 10])

    total_loss = 0.0
    num_batches = 0

    for eeg_data, labels, _words in tqdm(dataloader, desc="Evaluating"):
        eeg_data = eeg_data.to(device)
        labels = labels.to(device)

        logits = model(eeg_data, return_vq_loss=False, use_vq=True)
        loss_ce = criterion_ce(logits, labels)

        total_loss += loss_ce.item()
        num_batches += 1
        metrics.update(logits, labels)

    return total_loss / max(num_batches, 1), metrics.compute()


def train_model(
    model,
    optimizer,
    scheduler,
    criterion_dict,
    train_loader,
    val_loader,
    test_loader,
    config,
    device,
    start_epoch=1,
    best_val_acc=0.0,
):
    """Run the main training loop and final test evaluation."""
    save_dir = Path(config["training"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    Path(config["training"]["log_dir"]).mkdir(parents=True, exist_ok=True)

    num_epochs = int(config["training"]["num_epochs"])
    save_every = int(config["training"]["save_every"])
    save_best = bool(config["training"]["save_best"])
    warm_start_epochs = 3

    lambda_vq_target = float(config["training"]["loss"]["lambda_vq"])
    best_epoch = max(start_epoch - 1, 0)

    print("\n[Training] Starting training...")
    for epoch in range(start_epoch, num_epochs + 1):
        print(f"\n{'=' * 80}")
        print(f"EPOCH {epoch}/{num_epochs}")
        print(f"{'=' * 80}")

        config["training"]["loss"]["lambda_vq"] = 0.0 if epoch <= warm_start_epochs else lambda_vq_target
        print(f"[VQ] lambda_vq for this epoch: {config['training']['loss']['lambda_vq']}")

        train_loss = train_epoch(model, train_loader, optimizer, scheduler, criterion_dict, device, config, epoch)
        val_loss, val_results = evaluate(model, val_loader, criterion_dict, device)

        print(f"\nEpoch {epoch} Results:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Val Top-1:  {val_results['top1_acc'] * 100:.2f}%")
        print(f"  Val Top-3:  {val_results['top3_acc'] * 100:.2f}%")
        print(f"  Val Top-5:  {val_results['top5_acc'] * 100:.2f}%")
        print(f"  Val Top-10: {val_results['top10_acc'] * 100:.2f}%")

        if val_results["top10_acc"] > best_val_acc:
            best_val_acc = val_results["top10_acc"]
            best_epoch = epoch
            best_path = save_dir / "best_model.pt"
            save_checkpoint(best_path, epoch, model, optimizer, scheduler, val_results, best_val_acc)
            print(f"[Save] Best model saved: {best_path}")

        if epoch % save_every == 0:
            ckpt_path = save_dir / f"checkpoint_epoch_{epoch}.pt"
            save_checkpoint(ckpt_path, epoch, model, optimizer, scheduler, val_results, best_val_acc)
            print(f"[Save] Checkpoint saved: {ckpt_path}")

    config["training"]["loss"]["lambda_vq"] = lambda_vq_target

    best_path = save_dir / "best_model.pt"
    if best_path.exists():
        print(f"\n[Eval] Loading best checkpoint: {best_path}")
        load_checkpoint(best_path, model, device=device)
    else:
        print("\n[Eval] Best checkpoint not found; evaluating current in-memory model.")

    test_loss, test_results = evaluate(model, test_loader, criterion_dict, device)
    print("\nFINAL TEST RESULTS")
    print(f"  Test Loss: {test_loss:.4f}")
    print(f"  Test Top-1:  {test_results['top1_acc'] * 100:.2f}%")
    print(f"  Test Top-3:  {test_results['top3_acc'] * 100:.2f}%")
    print(f"  Test Top-5:  {test_results['top5_acc'] * 100:.2f}%")
    print(f"  Test Top-10: {test_results['top10_acc'] * 100:.2f}%")

    results = {
        "best_val_top10_acc": best_val_acc,
        "best_epoch": best_epoch,
        "test_metrics": {
            "loss": test_loss,
            "top1_acc": test_results["top1_acc"],
            "top3_acc": test_results["top3_acc"],
            "top5_acc": test_results["top5_acc"],
            "top10_acc": test_results["top10_acc"],
        },
    }
    results_path = save_dir / "final_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[Save] Final results saved: {results_path}")


def eval_model(model, criterion_dict, val_loader, test_loader, config, device, checkpoint_path):
    """Load a checkpoint and run evaluation on val/test sets."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"[Eval] Loading checkpoint: {checkpoint_path}")
    epoch, best_val_acc = load_checkpoint(checkpoint_path, model, device=device)
    print(f"[Eval] Loaded epoch={epoch}, best_val_acc={best_val_acc:.4f}")

    val_loss, val_results = evaluate(model, val_loader, criterion_dict, device)
    test_loss, test_results = evaluate(model, test_loader, criterion_dict, device)

    print("\nEVALUATION RESULTS")
    print(f"  Val Loss: {val_loss:.4f}")
    print(f"  Val Top-10: {val_results['top10_acc'] * 100:.2f}%")
    print(f"  Test Loss: {test_loss:.4f}")
    print(f"  Test Top-10: {test_results['top10_acc'] * 100:.2f}%")

    out = {
        "checkpoint": str(checkpoint_path),
        "val_metrics": {"loss": val_loss, **val_results},
        "test_metrics": {"loss": test_loss, **test_results},
    }
    out_path = Path(config["training"]["save_dir"]) / "eval_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"[Save] Eval results saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/enhanced_config.yaml")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval"])
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume training")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path for --mode eval")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    set_seed(int(config["seed"]))
    device = build_device(config)

    print("\n[Data] Loading vocabulary...")
    vocab = Vocabulary(vocab_size=500)
    vocab.load("data/vocabulary_top500.pkl")
    print(f"[Data] Vocabulary size: {len(vocab.word2idx)}")

    splits_path = Path("data/sentence_splits.pkl")
    if not splits_path.exists():
        print(f"\nError: Sentence splits not found at {splits_path}")
        print("Please run: python prepare_sentence_splits.py")
        return

    print("\n[Data] Loading sentence-level splits (80/10/10)...")
    splits = load_sentence_splits(str(splits_path))
    train_loader, val_loader, test_loader = build_dataloaders(config, vocab, splits)

    print("\n[Model] Building BELT-Enhanced model...")
    model = BELTEnhancedModel(config).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] Trainable parameters: {num_params:,}")

    optimizer, scheduler = build_optimizer_and_scheduler(model, config)
    criterion_dict = build_loss_functions(config, device)

    optimizer_cfg = config["training"]["optimizer"]
    scheduler_cfg = config["training"]["scheduler"]
    print("\n" + "=" * 80)
    print("BELT-ENHANCED CONFIGURATION SUMMARY")
    print("=" * 80)
    print(f"Model: {config['model']['name']}")
    print(f"Optimizer: {optimizer_cfg['name'].upper()}")
    print(f"Learning Rate: {optimizer_cfg['lr']}")
    print(f"Scheduler: {scheduler_cfg['name']}")
    print(f"Warmup Epochs: {scheduler_cfg['warmup_epochs']}")
    print(f"Gradient Clipping: {config['training']['gradient_clipping']['enabled']}")
    print(f"MixUp: {config['data'].get('use_mixup', False)}")
    print(f"DropPath: {config['model']['encoder'].get('use_drop_path', False)}")
    print(f"Multi-Sample Dropout: {config['model']['classifier'].get('use_multi_sample_dropout', False)}")
    print("=" * 80)

    start_epoch = 1
    best_val_acc = 0.0
    if args.mode == "train" and args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            print(f"[Resume] Loading checkpoint: {resume_path}")
            epoch, best_val_acc = load_checkpoint(
                resume_path,
                model,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
            )
            start_epoch = epoch + 1
            print(f"[Resume] Resuming from epoch {start_epoch}, best_val_acc={best_val_acc:.4f}")
        else:
            print(f"[Resume] WARNING: checkpoint not found: {resume_path}. Starting from scratch.")

    if args.mode == "train":
        train_model(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion_dict=criterion_dict,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            config=config,
            device=device,
            start_epoch=start_epoch,
            best_val_acc=best_val_acc,
        )
    else:
        checkpoint = args.checkpoint or str(Path(config["training"]["save_dir"]) / "best_model.pt")
        eval_model(
            model=model,
            criterion_dict=criterion_dict,
            val_loader=val_loader,
            test_loader=test_loader,
            config=config,
            device=device,
            checkpoint_path=checkpoint,
        )


if __name__ == "__main__":
    main()
