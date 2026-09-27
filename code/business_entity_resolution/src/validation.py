"""
Validation harness for Business Entity Resolution.
Splits data strictly by SOURCE-1 ENTITY (never by pair), stratified by country (~80/20).
Constructs a Leave-One-Country-Out (LOCO) split to simulate unseen-country testing.
Enforces strict anti-leakage verification.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import numpy as np

# Ensure src directory is in sys.path
_src_dir = Path(__file__).resolve().parent
_code_dir = _src_dir.parents[1]
for _p in [str(_src_dir), str(_code_dir)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from business_entity_resolution.src.config import (
        CACHE_DIR,
        DEFAULT_HELD_OUT_COUNTRY,
        DEFAULT_SEED,
        DEFAULT_VAL_RATIO,
        TRAIN_SOURCE1,
        detect_system_resources,
    )
    from business_entity_resolution.src.io_utils import load_s1_metadata
except ImportError:
    from config import (
        CACHE_DIR,
        DEFAULT_HELD_OUT_COUNTRY,
        DEFAULT_SEED,
        DEFAULT_VAL_RATIO,
        TRAIN_SOURCE1,
        detect_system_resources,
    )
    from io_utils import load_s1_metadata


def generate_entity_splits(
    source1_file: Path = TRAIN_SOURCE1,
    val_ratio: float = DEFAULT_VAL_RATIO,
    seed: int = DEFAULT_SEED,
    held_out_country: str = DEFAULT_HELD_OUT_COUNTRY,
    output_dir: Optional[Path] = None,
) -> Dict[str, any]:
    """
    Split Source 1 entities ~80/20, stratified by country.
    Also generate a Leave-One-Country-Out (LOCO) split holding out `held_out_country`.

    Parameters:
    -----------
    source1_file: path to train_source1.tsv
    val_ratio: fraction for validation (default 0.20)
    seed: random seed for reproducibility
    held_out_country: country name to hold out for LOCO testing
    output_dir: directory to save split files (defaults to CACHE_DIR / 'splits')

    Returns:
    --------
    dict with split statistics and entity ID sets:
        'train_s1_ids', 'val_s1_ids', 'loco_train_s1_ids', 'loco_val_s1_ids', 'stats'
    """
    if output_dir is None:
        output_dir = CACHE_DIR / "splits"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading Source 1 metadata from {source1_file}...")
    country_map, _ = load_s1_metadata(source1_file)
    total_entities = len(country_map)
    print(f"Loaded {total_entities:,} Source 1 entities.")

    # Group entity IDs by country
    country_groups: Dict[str, List[str]] = {}
    for eid, country in country_map.items():
        country_groups.setdefault(country, []).append(eid)

    rng = np.random.RandomState(seed)

    train_ids_list: List[str] = []
    val_ids_list: List[str] = []
    split_stats: Dict[str, dict] = {}

    for country, eids in country_groups.items():
        n = len(eids)
        eids_arr = np.array(eids)
        rng.shuffle(eids_arr)

        n_val = int(round(n * val_ratio))
        val_eids = eids_arr[:n_val].tolist()
        train_eids = eids_arr[n_val:].tolist()

        train_ids_list.extend(train_eids)
        val_ids_list.extend(val_eids)

        split_stats[country] = {
            "total": n,
            "train": len(train_eids),
            "val": len(val_eids),
            "val_percentage": round(len(val_eids) / n * 100, 2),
        }

    train_s1_set = set(train_ids_list)
    val_s1_set = set(val_ids_list)

    # Leave-One-Country-Out (LOCO) split
    # Held-out country entities: LOCO validation set
    # All other countries: LOCO training set
    loco_train_ids = [
        eid for eid, c in country_map.items() if c != held_out_country
    ]
    loco_val_ids = [
        eid for eid, c in country_map.items() if c == held_out_country
    ]

    # --- Strict Anti-Leakage Verification ---
    assert_no_leakage(train_s1_set, val_s1_set, total_entities)

    print("\n--- Split Summary ---")
    print(f"Total entities: {total_entities:,}")
    print(f"Train set: {len(train_s1_set):,} ({len(train_s1_set)/total_entities*100:.2f}%)")
    print(f"Validation set: {len(val_s1_set):,} ({len(val_s1_set)/total_entities*100:.2f}%)")
    for country, s in split_stats.items():
        print(f"  Country '{country}': Total={s['total']:,} | Train={s['train']:,} | Val={s['val']:,} ({s['val_percentage']}%)")

    print(f"\nLOCO Split (held-out country: '{held_out_country}'):")
    print(f"  LOCO Train (other countries): {len(loco_train_ids):,}")
    print(f"  LOCO Eval ('{held_out_country}'): {len(loco_val_ids):,}")

    # Save split files to disk in lightweight format
    manifest = {
        "seed": seed,
        "val_ratio": val_ratio,
        "held_out_country": held_out_country,
        "total_entities": total_entities,
        "n_train": len(train_ids_list),
        "n_val": len(val_ids_list),
        "stats": split_stats,
        "loco_stats": {
            "held_out_country": held_out_country,
            "loco_train_count": len(loco_train_ids),
            "loco_val_count": len(loco_val_ids),
        },
    }

    manifest_path = output_dir / "split_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # Save ID lists as compact plain text files (one ID per line)
    val_ids_path = output_dir / "val_s1_ids.txt"
    with open(val_ids_path, "w", encoding="utf-8") as f:
        for eid in val_ids_list:
            f.write(f"{eid}\n")

    train_ids_path = output_dir / "train_s1_ids.txt"
    with open(train_ids_path, "w", encoding="utf-8") as f:
        for eid in train_ids_list:
            f.write(f"{eid}\n")

    print(f"\nSaved split manifest to: {manifest_path}")
    print(f"Saved validation IDs to: {val_ids_path} ({os.path.getsize(val_ids_path)/1024/1024:.2f} MB)")
    print(f"Saved training IDs to: {train_ids_path} ({os.path.getsize(train_ids_path)/1024/1024:.2f} MB)")

    return {
        "train_s1_ids": train_s1_set,
        "val_s1_ids": val_s1_set,
        "loco_train_s1_ids": set(loco_train_ids),
        "loco_val_s1_ids": set(loco_val_ids),
        "manifest": manifest,
        "country_map": country_map,
    }


def assert_no_leakage(
    train_ids: Set[str], val_ids: Set[str], expected_total: int
) -> None:
    """
    Assert that train and validation sets have zero overlap and partition all entities.
    """
    overlap = train_ids.intersection(val_ids)
    if len(overlap) > 0:
        raise ValueError(
            f"DATA LEAKAGE DETECTED! {len(overlap)} Source-1 entities appear in both train and validation: "
            f"{list(overlap)[:5]}"
        )
    if len(train_ids) + len(val_ids) != expected_total:
        raise ValueError(
            f"Partition mismatch: train ({len(train_ids)}) + val ({len(val_ids)}) "
            f"= {len(train_ids) + len(val_ids)} != expected {expected_total}"
        )
    print("[LEAKAGE CHECK PASSED] Zero overlap between train and validation entity IDs.")


def load_split_ids(
    split_name: str = "val", split_dir: Optional[Path] = None
) -> Set[str]:
    """
    Load saved split IDs from disk.
    """
    if isinstance(split_name, Path):
        split_dir, split_name = split_name, "val"
    if split_dir is None:
        split_dir = CACHE_DIR / "splits"
    path = split_dir / f"{split_name}_s1_ids.txt"
    if not path.is_file():
        raise FileNotFoundError(f"Split file not found: {path}. Run generate_entity_splits() first.")
    with open(path, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


if __name__ == "__main__":
    detect_system_resources()
    generate_entity_splits()
