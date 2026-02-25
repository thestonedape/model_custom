#!/usr/bin/env python3
"""Run alpha=0 ablation for BELT-Enhanced and compute confusion/per-class reports.

This script does two things:
1) Evaluates the existing Enhanced best checkpoint on test split.
2) Trains an Enhanced alpha=0.0 ablation run, then evaluates its best checkpoint.

Outputs are written under `results/analysis/`.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from data.sentence_dataset import BELTSentenceDataset, load_sentence_splits
from data.vocabulary import Vocabulary
from experiments.model_enhanced import (
    BELTEnhancedModel,
    evaluate,
    set_seed,
    train_epoch,
)
from training.enhanced_losses import LabelSmoothingCrossEntropy
from training.schedulers import WarmupCosineSchedule


def pct(x: float) -> float:
    return 100.0 * float(x)


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_loaders(config: Dict, workers: int) -> Tuple[DataLoader, DataLoader, DataLoader, Vocabulary]:
    vocab = Vocabulary(vocab_size=500)
    vocab.load("data/vocabulary_top500.pkl")
    splits = load_sentence_splits("data/sentence_splits.pkl")

    train_dataset = BELTSentenceDataset(
        sentence_list=splits["train"], vocabulary=vocab, split="train", eeg_type="GD"
    )
    val_dataset = BELTSentenceDataset(
        sentence_list=splits["dev"], vocabulary=vocab, split="dev", eeg_type="GD"
    )
    test_dataset = BELTSentenceDataset(
        sentence_list=splits["test"], vocabulary=vocab, split="test", eeg_type="GD"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=True,
        num_workers=workers,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["evaluation"]["batch_size"],
        shuffle=False,
        num_workers=workers,
        pin_memory=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["evaluation"]["batch_size"],
        shuffle=False,
        num_workers=workers,
        pin_memory=False,
    )
    return train_loader, val_loader, test_loader, vocab


def setup_train_components(model: BELTEnhancedModel, config: Dict):
    optimizer_cfg = config["training"]["optimizer"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(optimizer_cfg["lr"]),
        weight_decay=float(optimizer_cfg["weight_decay"]),
        betas=tuple(float(b) for b in optimizer_cfg["betas"]),
        eps=float(optimizer_cfg["eps"]),
    )
    scheduler_cfg = config["training"]["scheduler"]
    scheduler = WarmupCosineSchedule(
        optimizer=optimizer,
        warmup_epochs=scheduler_cfg["warmup_epochs"],
        total_epochs=config["training"]["num_epochs"],
        min_lr=float(scheduler_cfg["min_lr"]),
    )
    criterion_ce = LabelSmoothingCrossEntropy(
        epsilon=float(config["training"]["loss"]["label_smoothing"])
    )
    criterion_dict = {
        "ce": criterion_ce,
        "cl": None,
        "vq": None,
    }
    return optimizer, scheduler, criterion_dict


def evaluate_with_confusion(
    model: BELTEnhancedModel,
    dataloader: DataLoader,
    criterion_ce: LabelSmoothingCrossEntropy,
    device: torch.device,
    vocab: Vocabulary,
) -> Dict:
    model.eval()
    n_classes = len(vocab.word2idx)

    def word_for(idx: int) -> str:
        return str(vocab.idx2word.get(int(idx), "<UNK>"))

    conf = torch.zeros((n_classes, n_classes), dtype=torch.long)
    total_loss = 0.0
    n_batches = 0
    n_samples = 0
    top1 = 0
    top5 = 0
    top10 = 0

    with torch.no_grad():
        for eeg_data, labels, _words in dataloader:
            eeg_data = eeg_data.to(device)
            labels = labels.to(device)
            logits = model(eeg_data, return_vq_loss=False, use_vq=True)
            loss = criterion_ce(logits, labels)

            total_loss += float(loss.item())
            n_batches += 1
            n_samples += int(labels.size(0))

            preds = torch.argmax(logits, dim=1)
            for t, p in zip(labels.cpu(), preds.cpu()):
                conf[int(t), int(p)] += 1

            tk = logits.topk(10, dim=1).indices
            corr = tk.eq(labels.view(-1, 1))
            top1 += int(corr[:, :1].any(dim=1).sum().item())
            top5 += int(corr[:, :5].any(dim=1).sum().item())
            top10 += int(corr[:, :10].any(dim=1).sum().item())

    support = conf.sum(dim=1)
    correct = conf.diag()
    per_class_acc = torch.where(support > 0, correct.float() / support.float(), torch.zeros_like(correct, dtype=torch.float))

    offdiag = conf.clone()
    idx = torch.arange(n_classes)
    offdiag[idx, idx] = 0
    flat_vals, flat_idx = torch.topk(offdiag.view(-1), k=min(30, offdiag.numel()))
    top_confusions = []
    for v, fi in zip(flat_vals.tolist(), flat_idx.tolist()):
        if v <= 0:
            continue
        true_idx = fi // n_classes
        pred_idx = fi % n_classes
        top_confusions.append(
            {
                "true_idx": int(true_idx),
                "true_word": word_for(int(true_idx)),
                "pred_idx": int(pred_idx),
                "pred_word": word_for(int(pred_idx)),
                "count": int(v),
            }
        )

    hardest = sorted(
        [
            {
                "class_idx": i,
                "word": word_for(i),
                "support": int(support[i].item()),
                "correct": int(correct[i].item()),
                "acc": float(per_class_acc[i].item()),
            }
            for i in range(n_classes)
            if int(support[i].item()) > 0
        ],
        key=lambda x: x["acc"],
    )
    easiest = list(reversed(hardest))

    return {
        "test_loss": total_loss / max(1, n_batches),
        "top1_acc": top1 / max(1, n_samples),
        "top5_acc": top5 / max(1, n_samples),
        "top10_acc": top10 / max(1, n_samples),
        "samples": n_samples,
        "per_class": hardest,
        "hardest_20": hardest[:20],
        "easiest_20": easiest[:20],
        "top_confusions": top_confusions,
    }


def write_reports(prefix: str, result: Dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / f"{prefix}_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "test_loss": result["test_loss"],
                "top1_acc": result["top1_acc"],
                "top5_acc": result["top5_acc"],
                "top10_acc": result["top10_acc"],
                "samples": result["samples"],
            },
            f,
            indent=2,
        )

    class_path = out_dir / f"{prefix}_per_class_accuracy.csv"
    with open(class_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class_idx", "word", "support", "correct", "acc"])
        for row in result["per_class"]:
            w.writerow([row["class_idx"], row["word"], row["support"], row["correct"], f"{row['acc']:.6f}"])

    txt_path = out_dir / f"{prefix}_confusion_summary.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"{prefix} test metrics\n")
        f.write(f"loss={result['test_loss']:.4f}\n")
        f.write(f"top1={pct(result['top1_acc']):.2f}%\n")
        f.write(f"top5={pct(result['top5_acc']):.2f}%\n")
        f.write(f"top10={pct(result['top10_acc']):.2f}%\n\n")
        f.write("Top confusion pairs (true -> pred):\n")
        for c in result["top_confusions"][:20]:
            f.write(
                f"  {c['true_word']} ({c['true_idx']}) -> {c['pred_word']} ({c['pred_idx']}): {c['count']}\n"
            )
        f.write("\nHardest 20 classes:\n")
        for r in result["hardest_20"]:
            f.write(
                f"  {r['word']} ({r['class_idx']}): acc={pct(r['acc']):.2f}% support={r['support']} correct={r['correct']}\n"
            )


def train_alpha0_ablation(
    base_config: Dict,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    out_dir: Path,
) -> Tuple[BELTEnhancedModel, Dict]:
    config = copy.deepcopy(base_config)
    config["training"]["loss"]["alpha"] = 0.0
    config["training"]["num_epochs"] = epochs
    config["training"]["save_dir"] = str(out_dir / "checkpoints")
    os.makedirs(config["training"]["save_dir"], exist_ok=True)

    model = BELTEnhancedModel(config).to(device)
    optimizer, scheduler, criterion_dict = setup_train_components(model, config)

    best_val_top10 = -1.0
    best_path = out_dir / "checkpoints" / "best_model.pt"
    history: List[Dict] = []

    warm_start_epochs = 3
    lambda_vq_cfg = float(config["training"]["loss"]["lambda_vq"])

    for epoch in range(1, epochs + 1):
        if epoch <= warm_start_epochs:
            config["training"]["loss"]["lambda_vq"] = 0.0
        else:
            config["training"]["loss"]["lambda_vq"] = lambda_vq_cfg

        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, criterion_dict, device, config, epoch
        )
        val_loss, val_res = evaluate(model, val_loader, criterion_dict, device, config)
        row = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "val_top1": float(val_res["top1_acc"]),
            "val_top5": float(val_res["top5_acc"]),
            "val_top10": float(val_res["top10_acc"]),
        }
        history.append(row)
        print(
            f"[alpha0] epoch={epoch} train_loss={train_loss:.4f} "
            f"val_top10={pct(val_res['top10_acc']):.2f}% val_top1={pct(val_res['top1_acc']):.2f}%"
        )

        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_results": val_res,
            "config_overrides": {"alpha": 0.0, "epochs": epochs},
        }
        if val_res["top10_acc"] > best_val_top10:
            best_val_top10 = float(val_res["top10_acc"])
            torch.save(ckpt, best_path)

        if epoch % 5 == 0:
            torch.save(ckpt, out_dir / "checkpoints" / f"epoch_{epoch}.pt")

    with open(out_dir / "alpha0_train_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    best_ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    return model, {"best_val_top10": best_val_top10, "best_path": str(best_path)}


def evaluate_checkpoint(
    config: Dict,
    ckpt_path: str,
    test_loader: DataLoader,
    device: torch.device,
    vocab: Vocabulary,
) -> Dict:
    model = BELTEnhancedModel(config).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    criterion_ce = LabelSmoothingCrossEntropy(
        epsilon=float(config["training"]["loss"]["label_smoothing"])
    )
    return evaluate_with_confusion(model, test_loader, criterion_ce, device, vocab)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/enhanced_config.yaml")
    parser.add_argument("--alpha0-epochs", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    base_config = load_config(args.config)
    set_seed(args.seed)
    device = torch.device(
        "cuda" if torch.cuda.is_available() and base_config["device"]["use_cuda"] else "cpu"
    )
    print(f"[device] {device}")

    out_dir = Path("results/analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader, vocab = build_loaders(base_config, workers=args.workers)

    # 1) Evaluate current Enhanced best checkpoint.
    enhanced_best_ckpt = "results/enhanced_checkpoints/best_model.pt"
    enhanced_res = evaluate_checkpoint(base_config, enhanced_best_ckpt, test_loader, device, vocab)
    write_reports("enhanced_best", enhanced_res, out_dir)
    print(
        f"[enhanced_best:test] top1={pct(enhanced_res['top1_acc']):.2f}% "
        f"top5={pct(enhanced_res['top5_acc']):.2f}% top10={pct(enhanced_res['top10_acc']):.2f}% "
        f"loss={enhanced_res['test_loss']:.4f}"
    )

    # 2) Run alpha=0 ablation training and evaluate best checkpoint.
    alpha0_dir = out_dir / f"alpha0_run_{args.alpha0_epochs}ep"
    alpha0_model, alpha0_meta = train_alpha0_ablation(
        base_config, train_loader, val_loader, device, args.alpha0_epochs, alpha0_dir
    )
    criterion_ce = LabelSmoothingCrossEntropy(
        epsilon=float(base_config["training"]["loss"]["label_smoothing"])
    )
    alpha0_res = evaluate_with_confusion(alpha0_model, test_loader, criterion_ce, device, vocab)
    write_reports("alpha0_best", alpha0_res, alpha0_dir)
    print(
        f"[alpha0:test] top1={pct(alpha0_res['top1_acc']):.2f}% "
        f"top5={pct(alpha0_res['top5_acc']):.2f}% top10={pct(alpha0_res['top10_acc']):.2f}% "
        f"loss={alpha0_res['test_loss']:.4f}"
    )

    summary = {
        "enhanced_best_test": {
            "top1": enhanced_res["top1_acc"],
            "top5": enhanced_res["top5_acc"],
            "top10": enhanced_res["top10_acc"],
            "loss": enhanced_res["test_loss"],
        },
        "alpha0_test": {
            "top1": alpha0_res["top1_acc"],
            "top5": alpha0_res["top5_acc"],
            "top10": alpha0_res["top10_acc"],
            "loss": alpha0_res["test_loss"],
        },
        "delta_alpha0_minus_enhanced": {
            "top1": alpha0_res["top1_acc"] - enhanced_res["top1_acc"],
            "top5": alpha0_res["top5_acc"] - enhanced_res["top5_acc"],
            "top10": alpha0_res["top10_acc"] - enhanced_res["top10_acc"],
            "loss": alpha0_res["test_loss"] - enhanced_res["test_loss"],
        },
        "alpha0_meta": alpha0_meta,
    }
    with open(out_dir / "alpha0_vs_enhanced_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("[done] wrote results/analysis/alpha0_vs_enhanced_summary.json")


if __name__ == "__main__":
    main()
