"""
Stage R1, R2, R3 Diagnostic Script.
Runs comprehensive segmentation analysis on the local validation split.
"""

import time
import os
import sys
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple
import numpy as np
import pandas as pd

# Ensure code and src directories are in sys.path
_src_dir = Path(__file__).resolve().parent
_code_dir = _src_dir.parents[1]
for _p in [str(_src_dir), str(_code_dir)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from business_entity_resolution.src.config import (
    CACHE_DIR,
    DEFAULT_SEED,
    TRAIN_GROUND_TRUTH,
    TRAIN_SOURCE1,
    TRAIN_SOURCE2,
    TRAIN_SOURCE3,
)
from business_entity_resolution.src.io_utils import (
    load_ground_truth,
    load_s1_metadata,
)
from business_entity_resolution.src.validation import load_split_ids
from business_entity_resolution.src.normalization import (
    normalize_business_name,
    normalize_address,
)
from business_entity_resolution.src.blocking import (
    NAME_STOPWORDS,
    ADDR_STOPWORDS,
)
from business_entity_resolution.src.features import (
    FEATURE_NAMES,
    extract_precomputed_pairwise_features,
)
from business_entity_resolution.src.model import EntityMatchingModel
from business_entity_resolution.src.postprocess import apply_many_to_one_consistency
from business_entity_resolution.src.scorer import evaluate_predictions, compute_entity_metrics


def run_diagnostics(n_val_entities: int = 2000, seed: int = DEFAULT_SEED, threshold: float = 0.75):
    print("=" * 60)
    print("RUNNING DIAGNOSTICS FOR STAGES R1, R2, R3")
    print(f"Validation entities: {n_val_entities:,}, Seed: {seed}, Threshold: {threshold}")
    print("=" * 60)

    val_s1_ids = list(load_split_ids("val"))
    country_map, s1_records = load_s1_metadata(TRAIN_SOURCE1, include_fields=True)
    gt = load_ground_truth(TRAIN_GROUND_TRUTH)

    rng = np.random.RandomState(seed)
    val_sample_eids = rng.choice(val_s1_ids, size=min(n_val_entities, len(val_s1_ids)), replace=False).tolist()
    val_sample_set = set(val_sample_eids)
    val_country_map = {eid: country_map[eid] for eid in val_sample_eids}
    val_gt = {eid: gt.get(eid, set()) for eid in val_sample_eids}

    # Load trained model
    model_path = CACHE_DIR / "models" / "lgbm_matcher.txt"
    matching_model = EntityMatchingModel()
    matching_model.load(model_path)
    print(f"Loaded ensemble model from {model_path}")

    # Build inverted index for validation entities
    core_clean_re = re.compile(r"\.(com|org|net|in|co\.in|fr|io|biz|info|edu|gov)\b", re.IGNORECASE)
    non_alnum_re = re.compile(r"[^a-z0-9]+")
    digits_re = re.compile(r"\b\d{2,6}\b")

    s1_parsed = {}
    core_index = {}
    compact_index = {}
    toksort_index = {}
    pfx5_index = {}
    num_addr_index = {}
    postal_addr_index = {}
    name_num_index = {}

    for eid in val_sample_eids:
        raw_name, raw_addr = s1_records[eid]
        c = country_map.get(eid, "").lower().strip()
        c_pfx = f"{c}_" if c else ""

        norm_name, core, phon = normalize_business_name(raw_name)
        norm_addr, postal, unit, nums = normalize_address(raw_addr)

        cc = non_alnum_re.sub(" ", core_clean_re.sub("", core)).strip()
        comp = cc.replace(" ", "")
        words = [w for w in cc.split() if len(w) >= 3 and w not in NAME_STOPWORDS]
        tok_sort = " ".join(sorted(words))

        addr_l = norm_addr.lower()
        addr_nums = digits_re.findall(addr_l)
        addr_words = [w for w in non_alnum_re.split(addr_l) if len(w) >= 4 and not w.isdigit() and w not in ADDR_STOPWORDS]

        s1_parsed[eid] = (norm_name, core, phon or "", norm_addr, postal or "", unit or "", nums, c)

        item = eid
        if len(cc) >= 3:
            core_index.setdefault(f"{c_pfx}{cc}", []).append(item)
        if len(comp) >= 4:
            compact_index.setdefault(f"{c_pfx}{comp}", []).append(item)
        if len(comp) >= 5:
            pfx5_index.setdefault(f"{c_pfx}{comp[:5]}", []).append(item)
        if len(words) >= 2:
            toksort_index.setdefault(f"{c_pfx}{tok_sort}", []).append(item)
        if addr_nums and addr_words:
            num_addr_index.setdefault(f"{c_pfx}{addr_nums[0]}_{addr_words[0]}", []).append(item)
        if postal and addr_words:
            postal_addr_index.setdefault(f"{c_pfx}{postal}_{addr_words[0]}", []).append(item)
        if words and addr_nums:
            name_num_index.setdefault(f"{c_pfx}{words[0]}_{addr_nums[0]}", []).append(item)

    # Stream S2 and S3 to collect candidates
    print("Streaming S2 and S3 to collect candidates...")
    val_candidates: Dict[str, Set[str]] = {eid: set() for eid in val_sample_eids}
    cand_info = {}

    for src_path in [TRAIN_SOURCE2, TRAIN_SOURCE3]:
        with open(src_path, "r", encoding="utf-8") as f:
            next(f)
            for line in f:
                parts = line.rstrip("\n\r").split("\t")
                if len(parts) < 2:
                    continue
                cid = parts[0]
                cname = parts[1]
                caddr = parts[2] if len(parts) > 2 else ""
                cc = parts[3].lower().strip() if len(parts) > 3 else ""
                c_pfx = f"{cc}_" if cc else ""

                c_clean = non_alnum_re.sub(" ", core_clean_re.sub("", cname.lower())).strip()
                c_comp = c_clean.replace(" ", "")
                cwords = [w for w in c_clean.split() if len(w) >= 3 and w not in NAME_STOPWORDS]

                matched_eids = set()
                if len(c_clean) >= 3 and f"{c_pfx}{c_clean}" in core_index:
                    matched_eids.update(core_index[f"{c_pfx}{c_clean}"])
                if len(c_comp) >= 4 and f"{c_pfx}{c_comp}" in compact_index:
                    matched_eids.update(compact_index[f"{c_pfx}{c_comp}"])
                if len(cwords) >= 2 and f"{c_pfx}{' '.join(sorted(cwords))}" in toksort_index:
                    matched_eids.update(toksort_index[f"{c_pfx}{' '.join(sorted(cwords))}"])
                
                caddr_l = caddr.lower()
                c_nums = digits_re.findall(caddr_l)
                c_addr_words = [w for w in non_alnum_re.split(caddr_l) if len(w) >= 4 and not w.isdigit() and w not in ADDR_STOPWORDS]
                if c_nums and c_addr_words and f"{c_pfx}{c_nums[0]}_{c_addr_words[0]}" in num_addr_index:
                    matched_eids.update(num_addr_index[f"{c_pfx}{c_nums[0]}_{c_addr_words[0]}"])
                if c_words_num_key := (f"{c_pfx}{cwords[0]}_{c_nums[0]}" if cwords and c_nums else None):
                    if c_words_num_key in name_num_index:
                        matched_eids.update(name_num_index[c_words_num_key])
                if len(c_comp) >= 5 and f"{c_pfx}{c_comp[:5]}" in pfx5_index:
                    matched_eids.update(pfx5_index[f"{c_pfx}{c_comp[:5]}"])

                for eid in matched_eids:
                    if len(val_candidates[eid]) < 40:
                        val_candidates[eid].add(cid)
                        cand_info[cid] = (cname, caddr, cc)

    # Feature extraction & scoring
    print("Scoring candidate pairs...")
    pair_rows = []
    X_val_list = []
    y_val_list = []

    for s1_id in val_sample_eids:
        s1_n, s1_c, s1_ph, s1_a, s1_post, s1_u, s1_num_set, s1_country = s1_parsed[s1_id]
        true_set = val_gt.get(s1_id, set())

        for cid in val_candidates.get(s1_id, set()):
            if cid in cand_info:
                cname, caddr, cc = cand_info[cid]
                cand_n, cand_c, cand_ph = normalize_business_name(cname)
                cand_a, cand_post, cand_u, cand_num_set = normalize_address(caddr)
                feat = extract_precomputed_pairwise_features(
                    s1_n, s1_c, s1_ph, s1_a, s1_post, s1_u, s1_num_set, s1_country,
                    cid, cand_n, cand_c, cand_ph or "", cand_a, cand_post or "", cand_u or "", cand_num_set, cc
                )
                X_val_list.append(feat)
                pair_rows.append((s1_id, cid))
                y_val_list.append(1 if cid in true_set else 0)

    X_val = np.array(X_val_list, dtype=np.float32)
    y_val = np.array(y_val_list, dtype=np.int32)
    val_probs = matching_model.predict_proba(X_val)

    # Cache validation candidate features and scores for ultra-fast R4, R5, R6 experiments!
    cache_path = CACHE_DIR / "val_diag_cache.npz"
    np.savez_compressed(
        cache_path,
        val_sample_eids=np.array(val_sample_eids),
        pair_s1=np.array([p[0] for p in pair_rows]),
        pair_cand=np.array([p[1] for p in pair_rows]),
        X_val=X_val,
        y_val=y_val,
        val_probs=val_probs,
    )
    print(f"Cached validation pairs to {cache_path}")

    # Evaluate at current threshold
    thresholded_pairs = [
        (pair_rows[i][0], pair_rows[i][1], float(val_probs[i]))
        for i in range(len(pair_rows))
        if val_probs[i] >= threshold
    ]
    preds = apply_many_to_one_consistency(thresholded_pairs, val_sample_set)
    metrics = evaluate_predictions(val_gt, preds, val_sample_set, val_country_map)

    # Detailed Stage R1 & R3 breakdown
    print("\n" + "=" * 60)
    print("STAGE R1: SINGLETON RATE ANALYSIS")
    print("=" * 60)
    true_empty_count = sum(1 for eid in val_sample_eids if len(val_gt[eid]) == 0)
    pred_empty_count = sum(1 for eid in val_sample_eids if len(preds[eid]) == 0)
    print(f"True Singleton Count:      {true_empty_count} / {len(val_sample_eids)} ({true_empty_count/len(val_sample_eids)*100:.2f}%)")
    print(f"Predicted Empty Count:     {pred_empty_count} / {len(val_sample_eids)} ({pred_empty_count/len(val_sample_eids)*100:.2f}%)")
    print(f"Over-prediction ratio:     {pred_empty_count / max(1, true_empty_count):.2f}x")

    for c in ["US", "India"]:
        c_eids = [eid for eid in val_sample_eids if val_country_map.get(eid) == c]
        c_true_empty = sum(1 for eid in c_eids if len(val_gt[eid]) == 0)
        c_pred_empty = sum(1 for eid in c_eids if len(preds[eid]) == 0)
        print(f"  Country '{c}': True={c_true_empty/len(c_eids)*100:.2f}%, Pred={c_pred_empty/len(c_eids)*100:.2f}% (ratio={c_pred_empty/max(1, c_true_empty):.2f}x)")

    print("\n" + "=" * 60)
    print("STAGE R3: ERROR LOCALIZATION BY SEGMENT")
    print("=" * 60)
    print(f"Overall Validation Macro F_0.5: {metrics['macro_f05']:.4f}")
    print(f"Macro Precision:                {metrics['macro_precision']:.4f}")
    print(f"Macro Recall:                   {metrics['macro_recall']:.4f}")
    print(f"Singleton F_0.5:                {metrics['singleton_f05']:.4f}")
    print(f"Non-Singleton F_0.5:            {metrics['non_singleton_f05']:.4f}")

    # By Country
    print("\n[Metrics by Country]:")
    for c, sc in metrics["by_country"].items():
        print(f"  {c:10s}: F_0.5={sc['macro_f05']:.4f}, P={sc['macro_precision']:.4f}, R={sc['macro_recall']:.4f} (count={sc['count']})")

    # By Match Count Bucket
    print("\n[Metrics by Predicted Match-Count Bucket]:")
    buckets = {
        "0 (empty)": lambda cnt: cnt == 0,
        "1": lambda cnt: cnt == 1,
        "2-3": lambda cnt: 2 <= cnt <= 3,
        "4-10": lambda cnt: 4 <= cnt <= 10,
        ">10": lambda cnt: cnt > 10,
    }
    for bname, bfn in buckets.items():
        b_eids = [eid for eid in val_sample_eids if bfn(len(preds[eid]))]
        if not b_eids:
            continue
        b_f05_scores = [compute_entity_metrics(val_gt[eid], preds[eid])[0] for eid in b_eids]
        b_true_empty = sum(1 for eid in b_eids if len(val_gt[eid]) == 0)
        print(f"  Bucket {bname:10s}: Count={len(b_eids):4d} ({len(b_eids)/len(val_sample_eids)*100:5.1f}%), Mean F_0.5={np.mean(b_f05_scores):.4f}, True Singletons={b_true_empty}")

    # By Candidate Source (S2 vs S3)
    print("\n[Metrics by Candidate Source (S2 vs S3)]:")
    s2_true_cnt = 0
    s2_pred_cnt = 0
    s2_tp = 0
    s3_true_cnt = 0
    s3_pred_cnt = 0
    s3_tp = 0

    for eid in val_sample_eids:
        for cid in val_gt[eid]:
            if cid.startswith("S2-"):
                s2_true_cnt += 1
            elif cid.startswith("S3-"):
                s3_true_cnt += 1
        for cid in preds[eid]:
            if cid.startswith("S2-"):
                s2_pred_cnt += 1
                if cid in val_gt[eid]:
                    s2_tp += 1
            elif cid.startswith("S3-"):
                s3_pred_cnt += 1
                if cid in val_gt[eid]:
                    s3_tp += 1

    s2_p = s2_tp / max(1, s2_pred_cnt)
    s2_r = s2_tp / max(1, s2_true_cnt)
    s2_f05 = (1.25 * s2_p * s2_r) / (0.25 * s2_p + s2_r) if (0.25 * s2_p + s2_r) > 0 else 0.0

    s3_p = s3_tp / max(1, s3_pred_cnt)
    s3_r = s3_tp / max(1, s3_true_cnt)
    s3_f05 = (1.25 * s3_p * s3_r) / (0.25 * s3_p + s3_r) if (0.25 * s3_p + s3_r) > 0 else 0.0

    print(f"  Source S2: True={s2_true_cnt:,}, Pred={s2_pred_cnt:,}, TP={s2_tp:,}, Precision={s2_p:.4f}, Recall={s2_r:.4f}, Micro F_0.5={s2_f05:.4f}")
    print(f"  Source S3: True={s3_true_cnt:,}, Pred={s3_pred_cnt:,}, TP={s3_tp:,}, Precision={s3_p:.4f}, Recall={s3_r:.4f}, Micro F_0.5={s3_f05:.4f}")


if __name__ == "__main__":
    run_diagnostics()
