"""
I/O utilities for streaming and reading TSV data with minimal memory footprint.
Ensures zero data corruption, strict tab separation, and streaming where possible.
"""

from pathlib import Path
from typing import Dict, Generator, Iterable, List, Optional, Set, Tuple
import psutil
import os


DELIM = "\t"


def stream_tsv_rows(file_path: Path) -> Generator[Tuple[str, ...], None, None]:
    """
    Stream rows from a TSV file without loading the whole file into memory.
    Yields tuple of string fields for each line (excluding header).
    """
    with open(file_path, "r", encoding="utf-8") as f:
        header = f.readline()
        if not header:
            return
        for line in f:
            line = line.rstrip("\n\r")
            if not line:
                continue
            yield tuple(line.split(DELIM))


def load_s1_metadata(
    file_path: Path,
    include_fields: bool = False,
) -> Tuple[Dict[str, str], Optional[Dict[str, Tuple[str, str]]]]:
    """
    Load S1 entity country mapping and optionally (business_name, business_address).
    
    Returns:
    --------
    country_map: dict of s1_id -> country
    records_map: dict of s1_id -> (business_name, business_address) if include_fields else None
    """
    country_map = {}
    records_map = {} if include_fields else None

    with open(file_path, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\n\r").split(DELIM)
        # Expected: entity_id, business_name, business_address, country
        for line in f:
            parts = line.rstrip("\n\r").split(DELIM)
            if len(parts) >= 4:
                eid, name, addr, country = parts[0], parts[1], parts[2], parts[3]
                country_map[eid] = country
                if include_fields:
                    records_map[eid] = (name, addr)
            elif len(parts) >= 1:
                country_map[parts[0]] = "UNKNOWN"

    return country_map, records_map


def load_ground_truth(
    file_path: Path,
    s1_id_filter: Optional[Set[str]] = None,
) -> Dict[str, Set[str]]:
    """
    Load ground truth mapping from TSV: source1_entity_id -> set of matched_entity_ids.
    If s1_id_filter is provided, only loads matching S1 IDs to save memory.
    """
    gt = {}
    with open(file_path, "r", encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            parts = line.rstrip("\n\r").split(DELIM)
            s1_id = parts[0]
            if s1_id_filter is not None and s1_id not in s1_id_filter:
                continue
            if len(parts) > 1 and parts[1].strip():
                gt[s1_id] = set(parts[1].strip().split(","))
            else:
                gt[s1_id] = set()
    return gt


def write_id_list_tsv(
    file_path: Path,
    id_mapping: Dict[str, Iterable[str]],
    header: Tuple[str, str] = ("source1_entity_id", "matched_entity_ids"),
    s1_order: Optional[List[str]] = None,
) -> None:
    """
    Write a tab-separated file mapping source1_entity_id -> comma-separated ID list.
    Ensures strict formatting compliant with utils/validate_submission.py:
    - Tab separator between the two columns
    - Comma separator between IDs in the list
    - No duplicate IDs in the list
    - Empty string for singletons / no candidates
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    order = s1_order if s1_order is not None else sorted(id_mapping.keys())

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(f"{header[0]}{DELIM}{header[1]}\n")
        for s1_id in order:
            matched = id_mapping.get(s1_id, [])
            if matched:
                # Remove intra-duplicates while preserving order
                seen = set()
                unique_ids = []
                for mid in matched:
                    if mid not in seen:
                        seen.add(mid)
                        unique_ids.append(mid)
                id_str = ",".join(unique_ids)
            else:
                id_str = ""
            f.write(f"{s1_id}{DELIM}{id_str}\n")
