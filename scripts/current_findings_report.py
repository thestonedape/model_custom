#!/usr/bin/env python3
"""Generate a reliable summary of current replica/enhanced findings.

Usage:
  python scripts/current_findings_report.py
  python scripts/current_findings_report.py --json-out results/current_findings_summary.json
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch


def torch_load_cpu(path: str) -> Dict[str, Any]:
    """Load checkpoint safely across torch versions."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def pct(x: Optional[float]) -> Optional[float]:
    return None if x is None else round(float(x) * 100.0, 4)


def parse_scalar(raw: str) -> Any:
    raw = raw.strip()
    if not raw:
        return None

    if raw.startswith('"') and raw.endswith('"') and len(raw) >= 2:
        return raw[1:-1]
    if raw.startswith("'") and raw.endswith("'") and len(raw) >= 2:
        return raw[1:-1]

    low = raw.lower()
    if low == "true":
        return True
    if low == "false":
        return False

    # List [a, b, c]
    if raw.startswith("[") and raw.endswith("]"):
        items = [x.strip() for x in raw[1:-1].split(",") if x.strip()]
        return [parse_scalar(x) for x in items]

    # Numeric
    try:
        if any(ch in raw for ch in [".", "e", "E"]):
            return float(raw)
        return int(raw)
    except ValueError:
        return raw


def get_yaml_path_value(yaml_text: str, path: List[str]) -> Any:
    """Minimal indentation-based YAML path reader for simple key/value configs."""
    stack: List[Dict[str, Any]] = []

    for raw_line in yaml_text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip() or ":" not in line:
            continue

        indent = len(line) - len(line.lstrip(" "))
        key, value = line.strip().split(":", 1)
        key = key.strip()
        value = value.strip()

        while stack and stack[-1]["indent"] >= indent:
            stack.pop()

        stack.append({"key": key, "indent": indent})
        current_path = [x["key"] for x in stack]

        if current_path == path and value != "":
            return parse_scalar(value)

    return None


def load_replica_metrics() -> Dict[str, Any]:
    data = json.loads(Path("results/main_results/final_results.json").read_text(encoding="utf-8"))
    test = data.get("test_metrics", {})
    return {
        "best_dev_acc_pct": pct(data.get("best_dev_acc")),
        "final_epoch": data.get("final_epoch"),
        "test": {
            "top1_pct": pct(test.get("top1_acc")),
            "top5_pct": pct(test.get("top5_acc")),
            "top10_pct": pct(test.get("top10_acc")),
            "loss": test.get("loss"),
            "L_ce": test.get("L_ce"),
            "L_vq": test.get("L_vq"),
            "L_cl": test.get("L_cl"),
        },
    }


def load_enhanced_checkpoints() -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for ckpt_path in sorted(glob.glob("results/enhanced_checkpoints/*.pt")):
        ckpt = torch_load_cpu(ckpt_path)
        if not isinstance(ckpt, dict):
            continue
        val = ckpt.get("val_results", {})
        if not isinstance(val, dict):
            continue
        if "top10_acc" not in val:
            continue
        rows.append(
            {
                "file": Path(ckpt_path).name,
                "epoch": ckpt.get("epoch"),
                "top1_pct": pct(val.get("top1_acc")),
                "top3_pct": pct(val.get("top3_acc")),
                "top5_pct": pct(val.get("top5_acc")),
                "top10_pct": pct(val.get("top10_acc")),
            }
        )

    if not rows:
        return {"count": 0, "best": None, "latest": None, "rows": []}

    rows_by_epoch = sorted(rows, key=lambda r: (r.get("epoch") is None, r.get("epoch"), r["file"]))
    best = max(rows_by_epoch, key=lambda r: r.get("top10_pct") if r.get("top10_pct") is not None else -1.0)
    latest = max(rows_by_epoch, key=lambda r: r.get("epoch") if isinstance(r.get("epoch"), int) else -1)

    return {
        "count": len(rows_by_epoch),
        "best": best,
        "latest": latest,
        "rows": rows_by_epoch,
    }


def parse_enhanced_log() -> Dict[str, Any]:
    log_path = Path("results/enhanced_training_log.txt")
    text = log_path.read_text(encoding="utf-8", errors="ignore")

    summary_pattern = re.compile(
        r"Epoch\s+(\d+)\s+Results:\s*\n"
        r"\s*Train Loss:\s*([0-9.]+)\s*\n"
        r"\s*Val Loss:\s*([0-9.]+)\s*\n"
        r"\s*Val Top-1:\s*([0-9.]+)%\s*\n"
        r"\s*Val Top-3:\s*([0-9.]+)%\s*\n"
        r"\s*Val Top-5:\s*([0-9.]+)%\s*\n"
        r"\s*Val Top-10:\s*([0-9.]+)%",
        re.M,
    )

    summaries: List[Dict[str, Any]] = []
    for m in summary_pattern.finditer(text):
        summaries.append(
            {
                "epoch": int(m.group(1)),
                "train_loss": float(m.group(2)),
                "val_loss": float(m.group(3)),
                "top1_pct": float(m.group(4)),
                "top3_pct": float(m.group(5)),
                "top5_pct": float(m.group(6)),
                "top10_pct": float(m.group(7)),
            }
        )

    progress_epochs = [int(x) for x in re.findall(r"Epoch\s+(\d+):\s+\d+%", text)]

    out: Dict[str, Any] = {
        "completed_count": len(summaries),
        "first_completed_epoch": summaries[0]["epoch"] if summaries else None,
        "last_completed_epoch": summaries[-1]["epoch"] if summaries else None,
        "in_progress_epoch": max(progress_epochs) if progress_epochs else None,
        "best_completed": None,
        "top1_unique_values": [],
        "top10_min": None,
        "top10_max": None,
        "top10_spread": None,
    }

    if summaries:
        best = max(summaries, key=lambda r: r["top10_pct"])
        t1_values = sorted({r["top1_pct"] for r in summaries})
        t10_values = [r["top10_pct"] for r in summaries]
        out.update(
            {
                "best_completed": best,
                "top1_unique_values": t1_values,
                "top10_min": min(t10_values),
                "top10_max": max(t10_values),
                "top10_spread": max(t10_values) - min(t10_values),
            }
        )

    return out


def load_training_setup() -> Dict[str, Any]:
    cfg_text = Path("config/enhanced_config.yaml").read_text(encoding="utf-8", errors="ignore")
    code = Path("experiments/model_enhanced.py").read_text(encoding="utf-8", errors="ignore")

    return {
        "optimizer": get_yaml_path_value(cfg_text, ["training", "optimizer", "name"]),
        "lr": get_yaml_path_value(cfg_text, ["training", "optimizer", "lr"]),
        "weight_decay": get_yaml_path_value(cfg_text, ["training", "optimizer", "weight_decay"]),
        "alpha": get_yaml_path_value(cfg_text, ["training", "loss", "alpha"]),
        "lambda_vq": get_yaml_path_value(cfg_text, ["training", "loss", "lambda_vq"]),
        "label_smoothing": get_yaml_path_value(cfg_text, ["training", "loss", "label_smoothing"]),
        "mixup_enabled": get_yaml_path_value(cfg_text, ["data", "use_mixup"]),
        "drop_path_enabled": get_yaml_path_value(cfg_text, ["model", "encoder", "use_drop_path"]),
        "multi_sample_dropout_enabled": get_yaml_path_value(cfg_text, ["model", "classifier", "use_multi_sample_dropout"]),
        "train_calls_use_vq_true": code.count("model(eeg_data, return_vq_loss=True, use_vq=True)"),
        "eval_calls_use_vq_true": code.count("model(eeg_data, return_vq_loss=False, use_vq=True)"),
        "train_loop_block_count": code.count("if args.mode == 'train':"),
    }


def print_human(summary: Dict[str, Any]) -> None:
    replica = summary["replica"]
    enhanced = summary["enhanced"]
    log = summary["enhanced_log"]
    setup = summary["training_setup"]

    print("=" * 90)
    print("CURRENT FINDINGS SUMMARY")
    print("=" * 90)

    print("\nReplica (final_results.json)")
    print(f"  Best Dev Acc: {replica['best_dev_acc_pct']}%")
    print(
        f"  Test Top-1/5/10: {replica['test']['top1_pct']}% / "
        f"{replica['test']['top5_pct']}% / {replica['test']['top10_pct']}%"
    )

    print("\nEnhanced (checkpoints)")
    if enhanced["best"]:
        b = enhanced["best"]
        print(f"  Best checkpoint: {b['file']} (epoch {b['epoch']})")
        print(f"  Best Val Top-1/10: {b['top1_pct']}% / {b['top10_pct']}%")
    if enhanced["latest"]:
        l = enhanced["latest"]
        print(f"  Latest saved epoch: {l['epoch']} ({l['file']}), Val Top-10={l['top10_pct']}%")

    print("\nEnhanced log")
    print(
        f"  Completed summaries: {log['completed_count']} "
        f"(epochs {log['first_completed_epoch']}..{log['last_completed_epoch']})"
    )
    print(f"  In-progress epoch seen in log tail: {log['in_progress_epoch']}")
    if log["best_completed"]:
        print(
            f"  Best completed summary: epoch {log['best_completed']['epoch']}, "
            f"Top-10={log['best_completed']['top10_pct']}%"
        )
    print(f"  Top-1 unique values: {log['top1_unique_values']}")
    print(
        f"  Top-10 min/max/spread: {log['top10_min']} / "
        f"{log['top10_max']} / {log['top10_spread']}"
    )

    print("\nObjective/config")
    print(f"  Optimizer: {setup['optimizer']} (lr={setup['lr']}, wd={setup['weight_decay']})")
    print(f"  Loss weights: alpha={setup['alpha']}, lambda_vq={setup['lambda_vq']}")
    print(
        f"  Enhancements: label_smoothing={setup['label_smoothing']}, mixup={setup['mixup_enabled']}, "
        f"drop_path={setup['drop_path_enabled']}, multi_sample_dropout={setup['multi_sample_dropout_enabled']}"
    )
    print(
        f"  Code checks: train use_vq=True calls={setup['train_calls_use_vq_true']}, "
        f"eval use_vq=True calls={setup['eval_calls_use_vq_true']}, "
        f"train-loop blocks={setup['train_loop_block_count']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize verified current findings from run artifacts.")
    parser.add_argument("--json-out", type=str, default=None, help="Optional path to write JSON summary")
    args = parser.parse_args()

    summary = {
        "replica": load_replica_metrics(),
        "enhanced": load_enhanced_checkpoints(),
        "enhanced_log": parse_enhanced_log(),
        "training_setup": load_training_setup(),
    }

    print_human(summary)

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\nWrote JSON summary to: {out_path}")


if __name__ == "__main__":
    main()
