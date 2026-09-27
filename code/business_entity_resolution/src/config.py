"""
Configuration and hardware detection for the Business Entity Resolution pipeline.
Detects available RAM, CPU count, and disk space dynamically.
"""

import os
import sys
from pathlib import Path
import psutil

# Base project directories (relative to student_resource repo root)
SRC_DIR = Path(__file__).resolve().parent
REPO_ROOT = SRC_DIR.parents[2]
CODE_ROOT = REPO_ROOT / "code"

# Ensure both CODE_ROOT and SRC_DIR are in sys.path to avoid conflict with stdlib 'code'
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

DATASET_DIR = REPO_ROOT / "dataset"
TRAIN_DATA_DIR = DATASET_DIR / "train"
TEST_DATA_DIR = DATASET_DIR / "test"
OUTPUT_DIR = REPO_ROOT / "output"
CACHE_DIR = REPO_ROOT / ".cache_er"

# Input raw file paths
TRAIN_SOURCE1 = TRAIN_DATA_DIR / "train_source1.tsv"
TRAIN_SOURCE2 = TRAIN_DATA_DIR / "train_source2.tsv"
TRAIN_SOURCE3 = TRAIN_DATA_DIR / "train_source3.tsv"
TRAIN_GROUND_TRUTH = TRAIN_DATA_DIR / "train_ground_truth.tsv"

TEST_SOURCE1 = TEST_DATA_DIR / "test_source1.tsv"
TEST_SOURCE2 = TEST_DATA_DIR / "test_source2.tsv"
TEST_SOURCE3 = TEST_DATA_DIR / "test_source3.tsv"

# Output submission paths
MATCHING_RESULTS_FILE = OUTPUT_DIR / "matching_results.tsv"
CANDIDATE_PAIRS_FILE = OUTPUT_DIR / "candidate_pairs.tsv"

# General configuration parameters
DEFAULT_SEED = 42
DEFAULT_VAL_RATIO = 0.20
DEFAULT_HELD_OUT_COUNTRY = "India"  # Configurable leave-one-country-out target


def detect_system_resources(verbose: bool = True) -> dict:
    """
    Detect available RAM, CPU count, and disk space dynamically via psutil.
    Never hardcode hardware assumptions.
    """
    mem = psutil.virtual_memory()
    disk = psutil.disk_usage(str(REPO_ROOT))
    cpu_count = psutil.cpu_count(logical=True)
    cpu_count_phys = psutil.cpu_count(logical=False)

    info = {
        "cpu_count_logical": cpu_count,
        "cpu_count_physical": cpu_count_phys,
        "ram_total_gb": round(mem.total / (1024 ** 3), 2),
        "ram_available_gb": round(mem.available / (1024 ** 3), 2),
        "ram_percent_used": mem.percent,
        "disk_total_gb": round(disk.total / (1024 ** 3), 2),
        "disk_free_gb": round(disk.free / (1024 ** 3), 2),
        "disk_percent_used": disk.percent,
    }

    if verbose:
        print("=" * 60)
        print("SYSTEM RESOURCE DETECTION (psutil)")
        print(f"  CPU Cores: {cpu_count} logical ({cpu_count_phys} physical)")
        print(f"  RAM: {info['ram_available_gb']} GB available / {info['ram_total_gb']} GB total ({info['ram_percent_used']}% used)")
        print(f"  Disk: {info['disk_free_gb']} GB free / {info['disk_total_gb']} GB total ({info['disk_percent_used']}% used)")
        if info["ram_available_gb"] < 2.0:
            print("  [ALERT] Low available RAM (< 2 GB)! Chunking and memory-mapping required.")
        if info["disk_free_gb"] < 5.0:
            print("  [ALERT] Low disk space (< 5 GB)! Avoid persisting uncompressed duplicates.")
        print("=" * 60)

    return info


def get_process_memory_mb() -> float:
    """Return the current process RSS memory in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)
