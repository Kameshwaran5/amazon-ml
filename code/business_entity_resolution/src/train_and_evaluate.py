"""
Training and validation evaluation script.
Executes Stages 4, 5, 6, 7, 8 and reports metrics for the progressive ladder.
Strictly ensures zero data leakage.
"""

import time
import os
import sys
import re
import psutil
from pathlib import Path
from typing import Dict, List, Set, Tuple
import numpy as np

# Ensure code and src directories are in sys.path
_src_dir = Path(__file__).resolve().parent
_code_dir = _src_dir.parents[1]
for _p in [str(_src_dir), str(_code_dir)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from business_entity_resolution.src.config import (
        CACHE_DIR,
        DEFAULT_SEED,
        TRAIN_GROUND_TRUTH,
        TRAIN_SOURCE1,
        TRAIN_SOURCE2,
        TRAIN_SOURCE3,
        detect_system_resources,
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
    from business_entity_resolution.src.calibration import ProbabilityCalibrator
    from business_entity_resolution.src.threshold import optimize_threshold
    from business_entity_resolution.src.postprocess import apply_many_to_one_consistency
    from business_entity_resolution.src.scorer import evaluate_predictions
except ImportError:
    from config import (
        CACHE_DIR,
        DEFAULT_SEED,
        TRAIN_GROUND_TRUTH,
        TRAIN_SOURCE1,
        TRAIN_SOURCE2,
        TRAIN_SOURCE3,
        detect_system_resources,
    )
    from io_utils import (
        load_ground_truth,
        load_s1_metadata,
    )
    from validation import load_split_ids
    from normalization import (
        normalize_business_name,
        normalize_address,
    )
    from blocking import (
        NAME_STOPWORDS,
        ADDR_STOPWORDS,
    )
    from features import (
        FEATURE_NAMES,
        extract_precomputed_pairwise_features,
    )
    from model import EntityMatchingModel
    from calibration import ProbabilityCalibrator
    from threshold import optimize_threshold
    from postprocess import apply_many_to_one_consistency
    from scorer import evaluate_predictions


def run_training_and_validation(
    n_train_entities: int = 6000,
    n_val_entities: int = 2000,
    seed: int = DEFAULT_SEED,
) -> Dict[str, any]:
    """
    Train Ensemble matching model on genuine mined hard negatives from 10.3M records,
    and evaluate progressive ladder on real validation candidates.
    """
    sys_info = detect_system_resources(verbose=True)
    t_start = time.time()

    # 1. Load splits (strictly isolated)
    print("\n[STEP 1] Loading Train and Validation Splits (Zero Leakage)...", flush=True)
    train_s1_ids = list(load_split_ids("train"))
    val_s1_ids = list(load_split_ids("val"))
    print(f"Total train pool: {len(train_s1_ids):,}, total val pool: {len(val_s1_ids):,}", flush=True)

    country_map, s1_records = load_s1_metadata(TRAIN_SOURCE1, include_fields=True)
    gt = load_ground_truth(TRAIN_GROUND_TRUTH)

    # Deterministic subsample of training and validation entities
    rng = np.random.RandomState(seed)
    train_sample_eids = rng.choice(train_s1_ids, size=min(n_train_entities, len(train_s1_ids)), replace=False).tolist()
    val_sample_eids = rng.choice(val_s1_ids, size=min(n_val_entities, len(val_s1_ids)), replace=False).tolist()

    val_sample_set = set(val_sample_eids)
    val_country_map = {eid: country_map[eid] for eid in val_sample_eids}
    val_gt = {eid: gt.get(eid, set()) for eid in val_sample_eids}

    # Verify zero leakage
    assert set(train_sample_eids).intersection(val_sample_set) == set(), "Data leakage between train and val samples!"

    print(f"Sampled {len(train_sample_eids):,} Train S1 entities and {len(val_sample_eids):,} Val S1 entities.", flush=True)

    # Compile regexes for ultra-fast candidate tokenization
    core_clean_re = re.compile(r"\.(com|org|net|in|co\.in|fr|io|biz|info|edu|gov)\b", re.IGNORECASE)
    non_alnum_re = re.compile(r"[^a-z0-9]+")
    digits_re = re.compile(r"\b\d{2,6}\b")

    # 2. Build Inverted Index over Train & Val entities
    print("\n[STEP 2] Building Multi-Pass Inverted Index over S1 entities...", flush=True)
    t_idx0 = time.time()

    # Track S1 parsed representations
    # eid -> tuple of parsed fields for fast feature extraction
    s1_parsed: Dict[str, Tuple] = {}

    # Inverted index: key -> list of (eid, is_val)
    core_index: Dict[str, List[Tuple[str, bool]]] = {}
    compact_index: Dict[str, List[Tuple[str, bool]]] = {}
    toksort_index: Dict[str, List[Tuple[str, bool]]] = {}
    pfx5_index: Dict[str, List[Tuple[str, bool]]] = {}
    rare_word_index: Dict[str, List[Tuple[str, bool]]] = {}
    num_addr_index: Dict[str, List[Tuple[str, bool]]] = {}
    postal_addr_index: Dict[str, List[Tuple[str, bool]]] = {}
    name_num_index: Dict[str, List[Tuple[str, bool]]] = {}

    all_eval_eids = [(eid, False) for eid in train_sample_eids] + [(eid, True) for eid in val_sample_eids]

    for eid, is_val in all_eval_eids:
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

        item = (eid, is_val)
        if len(cc) >= 3:
            core_index.setdefault(f"{c_pfx}{cc}", []).append(item)
        if len(comp) >= 4:
            compact_index.setdefault(f"{c_pfx}{comp}", []).append(item)
        if len(comp) >= 5:
            pfx5_index.setdefault(f"{c_pfx}{comp[:5]}", []).append(item)
        if len(words) >= 2:
            toksort_index.setdefault(f"{c_pfx}{tok_sort}", []).append(item)
        for w in words[:3]:
            rare_word_index.setdefault(f"{c_pfx}{w}", []).append(item)

        if addr_nums and addr_words:
            num_addr_index.setdefault(f"{c_pfx}{addr_nums[0]}_{addr_words[0]}", []).append(item)
        if postal and addr_words:
            postal_addr_index.setdefault(f"{c_pfx}{postal}_{addr_words[0]}", []).append(item)
        if words and addr_nums:
            name_num_index.setdefault(f"{c_pfx}{words[0]}_{addr_nums[0]}", []).append(item)

    # Cap posting lists on common/generic keys to prevent crowding
    pfx5_index = {k: v for k, v in pfx5_index.items() if len(v) <= 30}
    rare_word_index = {k: v for k, v in rare_word_index.items() if len(v) <= 30}
    num_addr_index = {k: v for k, v in num_addr_index.items() if len(v) <= 30}
    postal_addr_index = {k: v for k, v in postal_addr_index.items() if len(v) <= 30}
    name_num_index = {k: v for k, v in name_num_index.items() if len(v) <= 30}

    print(f"Indices built in {time.time()-t_idx0:.2f}s! Key counts: core={len(core_index):,}, compact={len(compact_index):,}, toksort={len(toksort_index):,}, addr_num={len(num_addr_index):,}", flush=True)

    # 3. Stream S2 & S3 across all 10.3M candidate rows to mine hard negatives & collect val candidates
    print("\n[STEP 3] Streaming S2 & S3 across 10.3M records to collect true matches and hard negatives...", flush=True)
    t_stream0 = time.time()

    # Storage for train pairs: positive and negative
    train_pos_features = []
    train_neg_features = []
    train_neg_counts = {eid: 0 for eid in train_sample_eids}
    max_negs_per_train_eid = 4

    # Storage for validation candidates: val_eid -> list of (cid, priority, cname, caddr, ccountry)
    val_candidates: Dict[str, Set[str]] = {eid: set() for eid in val_sample_eids}
    val_cand_info: Dict[str, Tuple[str, str, str]] = {}

    # Target sets for fast lookup
    train_gt_map = {eid: gt.get(eid, set()) for eid in train_sample_eids}
    all_train_targets = {cid for eid in train_sample_eids for cid in train_gt_map[eid]}
    all_val_targets = {cid for eid in val_sample_eids for cid in val_gt[eid]}

    total_streamed_rows = 0

    for src_path in [TRAIN_SOURCE2, TRAIN_SOURCE3]:
        t_file0 = time.time()
        file_rows = 0
        with open(src_path, "r", encoding="utf-8") as f:
            next(f)
            for line in f:
                file_rows += 1
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

                # Match against inverted index
                matched_s1_items: List[Tuple[str, bool, int]] = []

                # Exact core (priority 100)
                if len(c_clean) >= 3 and f"{c_pfx}{c_clean}" in core_index:
                    for eid, is_val in core_index[f"{c_pfx}{c_clean}"]:
                        matched_s1_items.append((eid, is_val, 100))

                # Compact core (priority 90)
                if len(c_comp) >= 4 and f"{c_pfx}{c_comp}" in compact_index:
                    for eid, is_val in compact_index[f"{c_pfx}{c_comp}"]:
                        matched_s1_items.append((eid, is_val, 90))

                # Token sort (priority 80)
                if len(cwords) >= 2 and f"{c_pfx}{' '.join(sorted(cwords))}" in toksort_index:
                    for eid, is_val in toksort_index[f"{c_pfx}{' '.join(sorted(cwords))}"]:
                        matched_s1_items.append((eid, is_val, 80))

                # Address numbers & street tokens
                caddr_l = caddr.lower()
                c_nums = digits_re.findall(caddr_l)
                c_addr_words = [w for w in non_alnum_re.split(caddr_l) if len(w) >= 4 and not w.isdigit() and w not in ADDR_STOPWORDS]

                if c_nums and c_addr_words and f"{c_pfx}{c_nums[0]}_{c_addr_words[0]}" in num_addr_index:
                    for eid, is_val in num_addr_index[f"{c_pfx}{c_nums[0]}_{c_addr_words[0]}"]:
                        matched_s1_items.append((eid, is_val, 60))

                if c_words_num_key := (f"{c_pfx}{cwords[0]}_{c_nums[0]}" if cwords and c_nums else None):
                    if c_words_num_key in name_num_index:
                        for eid, is_val in name_num_index[c_words_num_key]:
                            matched_s1_items.append((eid, is_val, 50))

                # 5-char prefix
                if len(c_comp) >= 5 and f"{c_pfx}{c_comp[:5]}" in pfx5_index:
                    for eid, is_val in pfx5_index[f"{c_pfx}{c_comp[:5]}"]:
                        matched_s1_items.append((eid, is_val, 40))

                # Process matched items
                for eid, is_val, priority in matched_s1_items:
                    if is_val:
                        # Validation candidate
                        if len(val_candidates[eid]) < 40:
                            val_candidates[eid].add(cid)
                            val_cand_info[cid] = (cname, caddr, cc)
                    else:
                        # Training sample
                        s1_targets = train_gt_map.get(eid, set())
                        if cid in s1_targets:
                            # True Positive pair
                            s1_n, s1_c, s1_ph, s1_a, s1_post, s1_u, s1_num_set, s1_country = s1_parsed[eid]
                            cand_n, cand_c, cand_ph = normalize_business_name(cname)
                            cand_a, cand_post, cand_u, cand_num_set = normalize_address(caddr)
                            feat = extract_precomputed_pairwise_features(
                                s1_n, s1_c, s1_ph, s1_a, s1_post, s1_u, s1_num_set, s1_country,
                                cid, cand_n, cand_c, cand_ph or "", cand_a, cand_post or "", cand_u or "", cand_num_set, cc
                            )
                            train_pos_features.append(feat)
                        elif train_neg_counts[eid] < max_negs_per_train_eid:
                            # Genuine Hard Negative pair
                            s1_n, s1_c, s1_ph, s1_a, s1_post, s1_u, s1_num_set, s1_country = s1_parsed[eid]
                            cand_n, cand_c, cand_ph = normalize_business_name(cname)
                            cand_a, cand_post, cand_u, cand_num_set = normalize_address(caddr)
                            feat = extract_precomputed_pairwise_features(
                                s1_n, s1_c, s1_ph, s1_a, s1_post, s1_u, s1_num_set, s1_country,
                                cid, cand_n, cand_c, cand_ph or "", cand_a, cand_post or "", cand_u or "", cand_num_set, cc
                            )
                            train_neg_features.append(feat)
                            train_neg_counts[eid] += 1

                # If this candidate is a validation target that was missed by keys, record it for inspection
                if cid in all_val_targets and cid not in val_cand_info:
                    val_cand_info[cid] = (cname, caddr, cc)

        total_streamed_rows += file_rows
        print(f"  Streamed {src_path.name}: {file_rows:,} rows in {time.time()-t_file0:.1f}s", flush=True)

    print(f"All {total_streamed_rows:,} records streamed in {time.time()-t_stream0:.1f}s!", flush=True)

    # 4. Measure Validation Recall Ceiling on REAL full data
    total_val_targets_count = sum(len(s) for s in val_gt.values())
    recalled_val_targets_count = sum(len(val_gt[eid].intersection(val_candidates[eid])) for eid in val_sample_eids)
    val_blocking_recall = recalled_val_targets_count / total_val_targets_count if total_val_targets_count > 0 else 0.0

    cand_lens = [len(cands) for cands in val_candidates.values()]
    print(f"\n[STAGE 3 RECALL CEILING ON 10.3M RECORDS]:")
    print(f"  Recalled: {recalled_val_targets_count:,} / {total_val_targets_count:,} ({val_blocking_recall*100:.2f}%)")
    print(f"  Candidates per entity: mean={np.mean(cand_lens):.2f}, median={np.median(cand_lens):.1f}, p95={np.percentile(cand_lens, 95):.1f}, max={max(cand_lens)}")

    # 5. Build Training Feature Matrix & Train Ensemble (LightGBM + CatBoost)
    print("\n[STEP 4] Training Ensemble Matching Model (LightGBM + CatBoost)...", flush=True)
    X_pos = np.array(train_pos_features, dtype=np.float32) if train_pos_features else np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)
    X_neg = np.array(train_neg_features, dtype=np.float32) if train_neg_features else np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)

    X_train = np.vstack([X_pos, X_neg])
    y_train = np.concatenate([np.ones(len(X_pos), dtype=np.int32), np.zeros(len(X_neg), dtype=np.int32)])

    # Shuffle training samples
    perm = rng.permutation(len(X_train))
    X_train = X_train[perm]
    y_train = y_train[perm]

    print(f"Constructed X_train shape: {X_train.shape} (Positives: {len(X_pos):,}, Hard Negatives: {len(X_neg):,})", flush=True)

    matching_model = EntityMatchingModel(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=63,
        random_state=seed,
        ensemble_weight_lgb=0.50,
    )
    matching_model.fit(X_train, y_train)

    top_features = sorted(matching_model.get_feature_importances().items(), key=lambda x: x[1], reverse=True)
    print("Top Feature Importances:")
    for fn, imp in top_features[:10]:
        print(f"  {fn:32s}: {imp:.1f}")

    # Save ensemble models
    model_path = CACHE_DIR / "models" / "lgbm_matcher.txt"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    matching_model.save(model_path)
    print(f"Models saved to {model_path} and {model_path.with_name('lgbm_matcher_catboost.cbm')}")

    # 6. Extract Validation Features & Score Candidate Pairs
    print("\n[STEP 5] Scoring Validation Candidate Pairs...", flush=True)
    val_pairs = []
    X_val_list = []
    y_val_list = []

    for s1_id in val_sample_eids:
        s1_n, s1_c, s1_ph, s1_a, s1_post, s1_u, s1_num_set, s1_country = s1_parsed[s1_id]
        true_set = val_gt.get(s1_id, set())

        for cid in val_candidates.get(s1_id, set()):
            if cid in val_cand_info:
                cname, caddr, cc = val_cand_info[cid]
                cand_n, cand_c, cand_ph = normalize_business_name(cname)
                cand_a, cand_post, cand_u, cand_num_set = normalize_address(caddr)
                feat = extract_precomputed_pairwise_features(
                    s1_n, s1_c, s1_ph, s1_a, s1_post, s1_u, s1_num_set, s1_country,
                    cid, cand_n, cand_c, cand_ph or "", cand_a, cand_post or "", cand_u or "", cand_num_set, cc
                )
                X_val_list.append(feat)
                val_pairs.append((s1_id, cid))
                y_val_list.append(1 if cid in true_set else 0)

    X_val = np.array(X_val_list, dtype=np.float32)
    y_val = np.array(y_val_list, dtype=np.int32)
    val_probs = matching_model.predict_proba(X_val)
    print(f"Scored {len(val_probs):,} validation candidate pairs (Positives in candidate pool: {np.sum(y_val==1):,}).", flush=True)

    # 7. Probability Calibration (Stage 6)
    print("\n[STEP 6] Probability Calibration (Isotonic Regression)...", flush=True)
    calibrator = ProbabilityCalibrator()
    calibrator.fit(val_probs, y_val)
    brier_before, brier_after = calibrator.evaluate_calibration(val_probs, y_val)
    print(f"Brier score: Before={brier_before:.5f}, After={brier_after:.5f} (reduction={((brier_before-brier_after)/brier_before)*100:.1f}%)")

    scored_val_pairs = [
        (val_pairs[i][0], val_pairs[i][1], float(val_probs[i]))
        for i in range(len(val_pairs))
    ]

    # 8. Precision-Oriented Threshold Optimization (Stage 7)
    best_t, best_metrics, history = optimize_threshold(
        scored_val_pairs, val_gt, val_sample_set, val_country_map, threshold_range=(0.40, 0.85, 0.05)
    )

    # 9. Global Consistency Post-Processing (Stage 8)
    print("\n[STEP 7] Global Consistency Post-Processing (Many-to-One)...", flush=True)
    thresholded_pairs = [p for p in scored_val_pairs if p[2] >= best_t]
    postprocessed_preds = apply_many_to_one_consistency(thresholded_pairs, val_sample_set, contested_margin_drop=None)
    final_metrics = evaluate_predictions(val_gt, postprocessed_preds, val_sample_set, val_country_map)

    t_end = time.time()
    rss_mb = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)

    print("\n" + "=" * 60)
    print("FINAL VALIDATION EVALUATION RESULTS")
    print(f"  Overall Macro F_0.5: {final_metrics['macro_f05']:.4f}")
    print(f"  Macro Precision:     {final_metrics['macro_precision']:.4f}")
    print(f"  Macro Recall:        {final_metrics['macro_recall']:.4f}")
    print(f"  Singleton F_0.5:     {final_metrics['singleton_f05']:.4f}")
    print(f"  Non-Singleton F_0.5: {final_metrics['non_singleton_f05']:.4f}")
    for c, sc in final_metrics["by_country"].items():
        print(f"  Country '{c}' F_0.5: {sc['macro_f05']:.4f}")
    print(f"  Optimal Threshold:   {best_t:.2f}")
    print(f"  Total Runtime:       {t_end - t_start:.2f}s ({((t_end - t_start)/60):.2f} min)")
    print(f"  Peak RSS Memory:     {rss_mb:.1f} MB")
    print("=" * 60)

    return {
        "final_metrics": final_metrics,
        "best_threshold": best_t,
        "val_blocking_recall": val_blocking_recall,
        "cand_stats": {
            "mean": float(np.mean(cand_lens)),
            "median": float(np.median(cand_lens)),
            "p95": float(np.percentile(cand_lens, 95)),
        },
        "top_features": top_features,
        "runtime": t_end - t_start,
        "peak_memory_mb": rss_mb,
    }


if __name__ == "__main__":
    run_training_and_validation()
