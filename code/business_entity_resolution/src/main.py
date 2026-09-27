"""
Main entry point for the Business Entity Resolution Challenge.
Supports one-command execution from raw data to submission outputs.
"""

import argparse
import sys
from pathlib import Path

# Ensure code and src directories are in sys.path
_src_dir = Path(__file__).resolve().parent
_code_dir = _src_dir.parents[1]
for _p in [str(_src_dir), str(_code_dir)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from business_entity_resolution.src.config import (
    DEFAULT_HELD_OUT_COUNTRY,
    DEFAULT_SEED,
    OUTPUT_DIR,
    TEST_DATA_DIR,
    TRAIN_DATA_DIR,
    detect_system_resources,
)
from business_entity_resolution.src.validation import generate_entity_splits
from business_entity_resolution.src.train_and_evaluate import run_training_and_validation
from business_entity_resolution.src.pipeline import run_pipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="Amazon ML Challenge 2026 — Business Entity Resolution Pipeline"
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        default=TEST_DATA_DIR,
        help="Path to directory containing test_source1/2/3.tsv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Path to directory where matching_results.tsv and candidate_pairs.tsv will be saved",
    )
    parser.add_argument(
        "--mode",
        choices=["all", "train-val", "predict"],
        default="all",
        help="Mode of operation: train-val, predict, or all (end-to-end)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--held-out-country",
        type=str,
        default=DEFAULT_HELD_OUT_COUNTRY,
        help="Country to hold out for leave-one-country-out validation (default: India)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.75,
        help="Decision probability threshold (default: 0.75)",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=40,
        help="Maximum candidate pairs per Source-1 entity (default: 40)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2048,
        help="Batch size for vectorized pairwise scoring (default: 2048)",
    )
    parser.add_argument(
        "--ann-k",
        type=int,
        default=30,
        help="Top-k nearest candidates for ANN / inverted index retrieval (default: 30)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 60)
    print("AMAZON ML CHALLENGE 2026 — BUSINESS ENTITY RESOLUTION")
    print(f"Mode: {args.mode}")
    print(f"Seed: {args.seed}")
    print("=" * 60)

    detect_system_resources(verbose=True)

    if args.mode in ("all", "train-val"):
        print("\n>>> Phase 1: Validation Split Generation...")
        generate_entity_splits(seed=args.seed, held_out_country=args.held_out_country)

        print("\n>>> Phase 2: Model Training and Progressive Ladder Evaluation...")
        val_results = run_training_and_validation(seed=args.seed)
        best_t = val_results.get("best_threshold", args.threshold)
        args.threshold = best_t

    if args.mode in ("all", "predict"):
        print(f"\n>>> Phase 3: Generating Test Output Files with Threshold {args.threshold}...")
        matching_file, candidate_file = run_pipeline(
            test_dir=args.test_dir,
            output_dir=args.output_dir,
            threshold=args.threshold,
            max_cands_per_s1=args.max_candidates,
            batch_size=args.batch_size,
        )
        print("\nGenerated Output Files:")
        print(f"  Matching Results: {matching_file}")
        print(f"  Candidate Pairs:  {candidate_file}")

    print("\nExecution complete.")


if __name__ == "__main__":
    main()
