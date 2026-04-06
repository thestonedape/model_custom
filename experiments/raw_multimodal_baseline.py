"""
Raw fixation-level EEG + eye-tracking baseline.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.raw_fixation_torch_dataset import RawFixationCacheDataset, collate_raw_fixation_batch
from models import RawMultimodalWordClassifier
from training.metrics import MetricsTracker


def clear_cuda_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def parse_args():
    parser = argparse.ArgumentParser(description="Train a raw multimodal fixation-level baseline")
    parser.add_argument("--config", type=str, default="config/raw_multimodal_mixed.yaml")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    return parser.parse_args()


def create_loader(
    cache_path,
    vocab_path,
    splits_path,
    split_name,
    tasks,
    batch_size,
    num_workers,
    shuffle,
    normalize_eeg=False,
    normalize_et=False,
    normalization_eps=1e-5,
):
    dataset = RawFixationCacheDataset(
        cache_path=cache_path,
        vocab_path=vocab_path,
        splits_path=splits_path,
        split_name=split_name,
        tasks=tasks,
        normalize_eeg=normalize_eeg,
        normalize_et=normalize_et,
        normalization_eps=normalization_eps,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_raw_fixation_batch,
    )
    return dataset, loader


def move_batch_to_device(batch, device):
    moved = dict(batch)
    for key in [
        "eeg",
        "eeg_mask",
        "et",
        "et_mask",
        "labels",
        "metrics",
        "mean_pupil_size",
        "n_fixations",
        "lengths",
        "sentence_idx",
        "word_idx",
        "fixation_idx",
    ]:
        if moved.get(key) is not None:
            moved[key] = moved[key].to(device)
    if moved.get("sentence_raw_eeg") is not None:
        moved["sentence_raw_eeg"] = moved["sentence_raw_eeg"].to(device)
    if moved.get("sentence_raw_mask") is not None:
        moved["sentence_raw_mask"] = moved["sentence_raw_mask"].to(device)
    return moved


def evaluate_loader(model, loader, device, tracker, max_batches=None, split_name="Dev", shard_name=None):
    model.eval()
    with torch.inference_mode():
        for batch_idx, batch in enumerate(loader):
            batch = move_batch_to_device(batch, device)
            logits = model(
                eeg=batch["eeg"],
                eeg_mask=batch["eeg_mask"],
                et=batch["et"],
                et_mask=batch["et_mask"],
                metrics=batch["metrics"],
                mean_pupil_size=batch["mean_pupil_size"],
                n_fixations=batch["n_fixations"],
            )
            labels = batch["labels"]
            loss = F.cross_entropy(logits, labels)
            tracker.update(
                logits,
                labels,
                {
                    "L_total": float(loss.item()),
                    "L_ce": float(loss.item()),
                    "L_vq": 0.0,
                    "L_cl": 0.0,
                },
            )
            if max_batches is not None and batch_idx + 1 >= max_batches:
                break
            del batch, logits, labels, loss

        if shard_name is not None and len(loader) > 0:
            print(f"  {split_name} shard '{shard_name}': consumed {min(len(loader), max_batches or len(loader))} batches")


def evaluate_split(
    model,
    cache_paths,
    vocab_path,
    splits_path,
    split_name,
    tasks,
    batch_size,
    num_workers,
    device,
    max_batches=None,
    min_batch_size=8,
    normalize_eeg=False,
    normalize_et=False,
    normalization_eps=1e-5,
):
    tracker = MetricsTracker(k_values=[1, 5, 10])
    total_samples = 0
    total_batches = 0

    for cache_path in cache_paths:
        current_batch_size = batch_size
        shard_name = Path(cache_path).name

        while True:
            dataset, loader = create_loader(
                cache_path=cache_path,
                vocab_path=vocab_path,
                splits_path=splits_path,
                split_name=split_name,
                tasks=tasks,
                batch_size=current_batch_size,
                num_workers=num_workers,
                shuffle=False,
                normalize_eeg=normalize_eeg,
                normalize_et=normalize_et,
                normalization_eps=normalization_eps,
            )
            granularity = dataset.metadata[0].get("sample_granularity", "sample") if dataset.metadata else "sample"
            print(
                f"  {split_name} shard '{shard_name}': {len(dataset)} {granularity} samples "
                f"(batch_size={current_batch_size})"
            )
            if len(dataset) == 0:
                break

            try:
                evaluate_loader(
                    model=model,
                    loader=loader,
                    device=device,
                    tracker=tracker,
                    max_batches=max_batches,
                    split_name=split_name,
                    shard_name=shard_name,
                )
                total_samples += len(dataset)
                total_batches += len(loader)
                break
            except torch.OutOfMemoryError:
                del loader, dataset
                clear_cuda_memory()
                if current_batch_size <= min_batch_size:
                    raise
                next_batch_size = max(min_batch_size, current_batch_size // 2)
                print(
                    f"  OOM during {split_name} on shard '{shard_name}'. "
                    f"Retrying with batch_size={next_batch_size}."
                )
                current_batch_size = next_batch_size
                continue
            finally:
                clear_cuda_memory()

    print(f"\n{split_name} aggregate: {total_samples} samples across {total_batches} shard batches")
    tracker.print_summary(prefix=split_name)
    return tracker.compute()


def main():
    args = parse_args()
    config_path = Path(args.config)
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    print("=" * 80)
    print("RAW MULTIMODAL BASELINE")
    print("=" * 80)
    print(f"Configuration loaded from: {config_path}")

    cache_path = config["data"].get("cache_path")
    cache_paths = config["data"].get("cache_paths")
    vocab_path = config["data"]["vocab_path"]
    splits_path = config["data"]["splits_path"]
    tasks = list(config["data"]["tasks"])
    normalize_eeg = config["data"].get("normalize_eeg", False)
    normalize_et = config["data"].get("normalize_et", False)
    normalization_eps = config["data"].get("normalization_eps", 1e-5)
    if cache_paths is None:
        if cache_path is None:
            raise ValueError("Config must define data.cache_path or data.cache_paths")
        cache_paths = [cache_path]
    elif cache_path is not None:
        cache_paths = list(cache_paths) + [cache_path]

    missing_paths = [path for path in cache_paths if not Path(path).exists()]
    if missing_paths:
        raise FileNotFoundError(
            "Raw caches not found: "
            + ", ".join(missing_paths)
            + ". Build them first with scripts/build_fixation_cache.py."
        )

    if not cache_paths:
        raise ValueError("Config must define data.cache_path or data.cache_paths")

    print("\nRaw cache shards:")
    for shard_path in cache_paths:
        print(f"  - {shard_path}")

    device = torch.device(config["hardware"].get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    model = RawMultimodalWordClassifier(
        eeg_dim=config["model"].get("eeg_dim", 105),
        et_dim=config["model"].get("et_dim", 4),
        metrics_dim=config["model"].get("metrics_dim", 5),
        model_dim=config["model"].get("model_dim", 256),
        et_hidden_dim=config["model"].get("et_hidden_dim", 64),
        aux_hidden_dim=config["model"].get("aux_hidden_dim", 64),
        num_heads=config["model"].get("num_heads", 8),
        num_layers=config["model"].get("num_layers", 3),
        ff_dim=config["model"].get("ff_dim", 512),
        dropout=config["model"].get("dropout", 0.1),
        num_classes=config["model"].get("num_classes", 500),
        norm_first=config["model"].get("norm_first", False),
        patch_kernel_size=config["model"].get("patch_kernel_size", 7),
        patch_stride=config["model"].get("patch_stride", 4),
    ).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"].get("learning_rate", 1e-4),
        weight_decay=config["training"].get("weight_decay", 1e-2),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config["training"].get("epochs", 10),
    )

    save_dir = Path(config["experiments"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    best_dev = 0.0
    start_epoch = 1
    total_epochs = config["training"].get("epochs", 10)

    resume_path = args.resume or config["training"].get("resume_from")
    if resume_path is not None:
        resume_file = Path(resume_path)
        if not resume_file.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_file}")
        checkpoint = torch.load(resume_file, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        best_dev = float(checkpoint.get("best_dev_top10", 0.0))
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        print(
            f"Resumed from checkpoint: {resume_file}\n"
            f"  Last completed epoch: {start_epoch - 1}\n"
            f"  Best dev Top-10 so far: {best_dev * 100:.2f}%"
        )

    if start_epoch > total_epochs:
        print(
            f"Checkpoint is already at epoch {start_epoch - 1}, "
            f"which is beyond configured total epochs={total_epochs}. Nothing to run."
        )
        return

    for epoch in range(start_epoch, total_epochs + 1):
        print("\n" + "=" * 80)
        print(f"EPOCH {epoch}/{total_epochs}")
        print("=" * 80)
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.2e}")

        model.train()
        train_tracker = MetricsTracker(k_values=[1, 5, 10])
        start_time = time.time()
        global_batch_idx = 0
        max_train_batches = config["training"].get("max_train_batches")
        stop_training = False

        for cache_path_item in cache_paths:
            train_dataset, train_loader = create_loader(
                cache_path=cache_path_item,
                vocab_path=vocab_path,
                splits_path=splits_path,
                split_name="train",
                tasks=tasks,
                batch_size=config["training"]["batch_size"],
                num_workers=config["hardware"].get("num_workers", 0),
                shuffle=True,
                normalize_eeg=normalize_eeg,
                normalize_et=normalize_et,
                normalization_eps=normalization_eps,
            )
            shard_name = Path(cache_path_item).name
            granularity = train_dataset.metadata[0].get("sample_granularity", "sample") if train_dataset.metadata else "sample"
            print(f"  Train shard '{shard_name}': {len(train_dataset)} {granularity} samples")
            if len(train_dataset) == 0:
                continue

            for batch_idx, batch in enumerate(train_loader):
                batch = move_batch_to_device(batch, device)
                optimizer.zero_grad()
                logits = model(
                    eeg=batch["eeg"],
                    eeg_mask=batch["eeg_mask"],
                    et=batch["et"],
                    et_mask=batch["et_mask"],
                    metrics=batch["metrics"],
                    mean_pupil_size=batch["mean_pupil_size"],
                    n_fixations=batch["n_fixations"],
                )
                labels = batch["labels"]
                loss = F.cross_entropy(logits, labels)
                loss.backward()
                if config["training"].get("grad_clip", 0) > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["grad_clip"])
                optimizer.step()

                train_tracker.update(
                    logits.detach(),
                    labels,
                    {
                        "L_total": float(loss.item()),
                        "L_ce": float(loss.item()),
                        "L_vq": 0.0,
                        "L_cl": 0.0,
                    },
                )

                if global_batch_idx % config["logging"].get("log_interval", 10) == 0:
                    print(
                        f"  Shard {shard_name} | Batch {batch_idx}/{len(train_loader)} | "
                        f"Global {global_batch_idx} | Loss: {loss.item():.4f} | "
                        f"Time: {time.time() - start_time:.1f}s"
                    )

                global_batch_idx += 1
                if max_train_batches is not None and global_batch_idx >= max_train_batches:
                    print(f"\n  Early stop: reached max_train_batches={max_train_batches}")
                    stop_training = True
                    break

                del batch, logits, labels, loss

            del train_loader
            del train_dataset
            clear_cuda_memory()
            if stop_training:
                break

        train_tracker.print_summary(prefix=f"Epoch {epoch} Train")
        print("\nEvaluating on dev set...")
        clear_cuda_memory()
        dev_metrics = evaluate_split(
            model,
            cache_paths=cache_paths,
            vocab_path=vocab_path,
            splits_path=splits_path,
            split_name="dev",
            tasks=tasks,
            batch_size=config["evaluation"]["batch_size"],
            num_workers=config["hardware"].get("num_workers", 0),
            device=device,
            max_batches=config["evaluation"].get("max_eval_batches"),
            min_batch_size=config["evaluation"].get("min_batch_size", 8),
            normalize_eeg=normalize_eeg,
            normalize_et=normalize_et,
            normalization_eps=normalization_eps,
        )

        dev_top10 = dev_metrics.get("top10_acc", 0.0)
        print(
            f"\nEPOCH ACCURACY SUMMARY | "
            f"Train Top-1: {train_tracker.compute().get('top1_acc', 0.0)*100:.2f}% | "
            f"Train Top-5: {train_tracker.compute().get('top5_acc', 0.0)*100:.2f}% | "
            f"Train Top-10: {train_tracker.compute().get('top10_acc', 0.0)*100:.2f}% | "
            f"Dev Top-1: {dev_metrics.get('top1_acc', 0.0)*100:.2f}% | "
            f"Dev Top-5: {dev_metrics.get('top5_acc', 0.0)*100:.2f}% | "
            f"Dev Top-10: {dev_top10*100:.2f}%"
        )

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "config": config,
            "best_dev_top10": best_dev,
        }
        checkpoint_path = save_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")

        if dev_top10 > best_dev:
            best_dev = dev_top10
            best_path = save_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"Saved best model: {best_path}")

        scheduler.step()


if __name__ == "__main__":
    main()
