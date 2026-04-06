"""
Build a raw fixation-level EEG + eye-tracking cache from ZuCo MATLAB files.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.raw_fixation_dataset import build_fixation_cache


def parse_args():
    parser = argparse.ArgumentParser(description="Build a raw fixation-level multimodal ZuCo cache")
    parser.add_argument("--dataset-root", default="dataset/ZuCo")
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--file-start", type=int, default=0)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--max-sentences", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--include-sentence-raw", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 80)
    print("BUILDING RAW FIXATION-LEVEL ZUCO CACHE")
    print("=" * 80)
    print(f"Dataset root: {args.dataset_root}")
    print(f"Tasks: {args.tasks}")
    print(f"Output: {args.output}")
    print(f"File start: {args.file_start}")
    print(f"Max files: {args.max_files}")
    print(f"Max sentences: {args.max_sentences}")
    print(f"Max samples: {args.max_samples}")
    print(f"Include sentence raw EEG: {bool(args.include_sentence_raw)}")

    payload = build_fixation_cache(
        dataset_root=args.dataset_root,
        tasks=args.tasks,
        output_path=args.output,
        include_sentence_raw=args.include_sentence_raw,
        file_start=args.file_start,
        max_files=args.max_files,
        max_sentences=args.max_sentences,
        max_samples=args.max_samples,
    )

    print("\n" + "=" * 80)
    print("BUILD COMPLETE")
    print("=" * 80)
    print(payload["summary"])
    print(f"\nSaved cache to: {Path(args.output)}")


if __name__ == "__main__":
    main()
