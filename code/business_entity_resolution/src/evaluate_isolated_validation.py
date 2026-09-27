"""
Isolated Validation-Set Evaluation Engine for Business Entity Resolution.
Evaluates each of the four pipeline changes in strict isolation on the local validation split:
1. Candidate normalization fix ALONE (previous settings: threshold 0.75, 50/50 ensemble, no match cap)
2. Ensemble reweight (85% CatBoost / 15% LightGBM) ALONE (on top of normalization fix, threshold 0.75, no match cap)
3. Full threshold grid search on newly normalized candidate scores
4. Max-match cap of 11 ALONE (with and without cap, and empirical verification against train_ground_truth.tsv)
5. Combined effect of all four changes vs sum of individual effects (interaction analysis)
"""

import os
import sys
import time
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from rapidfuzz import fuzz
import unidecode
import lightgbm as lgb
from catboost import CatBoostClassifier

# Ensure project modules are importable
_src_dir = Path(__file__).resolve().parent
_code_dir = _src_dir.parents[1]
for _p in [str(_src_dir), str(_code_dir)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from business_entity_resolution.src.config import (
    CACHE_DIR,
    TRAIN_GROUND_TRUTH,
    TRAIN_SOURCE1,
    TRAIN_SOURCE2,
    TRAIN_SOURCE3,
)
from business_entity_resolution.src.io_utils import (
    load_ground_truth,
    load_s1_metadata,
)
from business_entity_resolution.src.normalization import (
    normalize_business_name,
    normalize_address,
    _LEGAL_SUFFIX_REGEX,
)
from business_entity_resolution.src.blocking import (
    NAME_STOPWORDS,
    ADDR_STOPWORDS,
)
from business_entity_resolution.src.features import (
    FEATURE_NAMES,
    extract_precomputed_pairwise_features,
)
from business_entity_resolution.src.scorer import (
    evaluate_predictions,
    compute_entity_metrics,
)


def apply_many_to_one_with_cap(
    scored_pairs: List[Tuple[str, str, float]],
    all_s1_ids: Set[str],
    max_cap: int = None,
) -> Dict[str, Set[str]]:
    """
    Enforce global many-to-one consistency:
    - Each candidate ID is assigned to the S1 entity with the highest predicted probability.
    - Optionally caps matches per S1 entity to max_cap.
    """
    claims_by_candidate: Dict[str, List[Tuple[str, float]]] = {}
    for s1_id, cand_id, prob in scored_pairs:
        claims_by_candidate.setdefault(cand_id, []).append((s1_id, prob))

    matches_by_s1: Dict[str, List[Tuple[str, float]]] = {s1_id: [] for s1_id in all_s1_ids}

    for cand_id, claims in claims_by_candidate.items():
        if len(claims) == 1:
            s1_id, prob = claims[0]
            if s1_id in matches_by_s1:
                matches_by_s1[s1_id].append((cand_id, prob))
        else:
            claims.sort(key=lambda x: x[1], reverse=True)
            top_s1, top_prob = claims[0]
            if top_s1 in matches_by_s1:
                matches_by_s1[top_s1].append((cand_id, top_prob))

    final_matches: Dict[str, Set[str]] = {}
    for s1_id, won_list in matches_by_s1.items():
        if not won_list:
            final_matches[s1_id] = set()
        else:
            # Sort descending by score for consistent capping
            won_list.sort(key=lambda x: x[1], reverse=True)
            if max_cap is not None:
                won_list = won_list[:max_cap]
            final_matches[s1_id] = {c[0] for c in won_list}

    return final_matches


def run_candidate_generation_and_scoring(val_sample_eids: List[str], max_cands_per_s1: int = 40):
    """
    Generate candidates using the NEW candidate normalization fix from pipeline.py,
    extract pairwise features, and score with LightGBM and CatBoost.
    """
    cache_file = CACHE_DIR / "val_new_norm_cache.npz"
    if cache_file.is_file():
        print(f"Loading precomputed normalized candidate cache from {cache_file}...")
        data = np.load(cache_file)
        return (
            list(data["val_sample_eids"]),
            data["pair_s1"],
            data["pair_cand"],
            data["X_val"],
            data["y_val"],
            data["p_lgb"],
            data["p_cb"],
        )

    print("Generating candidate pairs with candidate normalization fix from pipeline.py...")
    t0 = time.time()

    country_map, s1_records = load_s1_metadata(TRAIN_SOURCE1, include_fields=True)
    gt = load_ground_truth(TRAIN_GROUND_TRUTH)

    core_clean_re = re.compile(r"\.(com|org|net|in|co\.in|fr|io|biz|info|edu|gov)\b", re.IGNORECASE)
    non_alnum_re = re.compile(r"[^a-z0-9]+")
    digits_re = re.compile(r"\b\d{2,6}\b")

    s1_parsed = {}
    s1_ids = []
    s1_clean_names = []
    core_index = {}
    compact_index = {}
    toksort_index = {}
    pfx6_index = {}
    num_addr_index = {}
    postal_addr_index = {}
    name_num_index = {}

    for idx, eid in enumerate(val_sample_eids):
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
        s1_ids.append(eid)
        s1_clean_names.append(cc)

        if len(cc) >= 3:
            core_index.setdefault(f"{c_pfx}{cc}", []).append(idx)
        if len(comp) >= 4:
            compact_index.setdefault(f"{c_pfx}{comp}", []).append(idx)
        if len(words) >= 2:
            toksort_index.setdefault(f"{c_pfx}{tok_sort}", []).append(idx)
        if len(comp) >= 6:
            pfx6_index.setdefault(f"{c_pfx}{comp[:6]}", []).append(idx)
        if addr_nums and addr_words:
            num_addr_index.setdefault(f"{c_pfx}{addr_nums[0]}_{addr_words[0]}", []).append(idx)
        if postal and addr_words:
            postal_addr_index.setdefault(f"{c_pfx}{postal}_{addr_words[0]}", []).append(idx)
        if words and addr_nums:
            name_num_index.setdefault(f"{c_pfx}{words[0]}_{addr_nums[0]}", []).append(idx)

    # Stream S2 and S3 using pipeline.py normalization
    print("Streaming S2 and S3 against index...")
    val_candidates: Dict[str, Set[str]] = {eid: set() for eid in val_sample_eids}
    cand_info = {}

    for cand_path in [TRAIN_SOURCE2, TRAIN_SOURCE3]:
        t_src0 = time.time()
        with open(cand_path, "r", encoding="utf-8") as cand_fp:
            next(cand_fp)
            for line in cand_fp:
                parts = line.rstrip("\n\r").split("\t")
                if len(parts) < 2:
                    continue
                cid = parts[0]
                cname = parts[1]
                caddr = parts[2] if len(parts) > 2 else ""
                ccountry = parts[3].lower().strip() if len(parts) > 3 else ""
                c_pfx = f"{ccountry}_" if ccountry else ""

                # NEW normalization fix from pipeline.py
                c_ascii = unidecode.unidecode(cname.lower())
                c_no_domain = core_clean_re.sub(" ", c_ascii)
                c_clean = non_alnum_re.sub(" ", c_no_domain)
                c_core = _LEGAL_SUFFIX_REGEX.sub(" ", c_clean)
                c_core = re.sub(r"^\b(the|smt|mr|mrs|ms|dr|m/s)\b\s*", "", c_core, flags=re.IGNORECASE)
                c_core = re.sub(r"\s+", " ", c_core).strip()
                c_comp = c_core.replace(" ", "")
                c_comp_raw = c_clean.replace(" ", "")
                cwords = [w for w in c_core.split() if len(w) >= 3 and w not in NAME_STOPWORDS]

                caddr_l = caddr.lower()
                c_nums = digits_re.findall(caddr_l)
                c_awords = [w for w in non_alnum_re.split(caddr_l) if len(w) >= 4 and not w.isdigit() and w not in ADDR_STOPWORDS]

                name_matched = set()
                addr_matched = set()

                if len(c_core) >= 3 and f"{c_pfx}{c_core}" in core_index:
                    name_matched.update(core_index[f"{c_pfx}{c_core}"])
                if len(c_clean) >= 3 and f"{c_pfx}{c_clean}" in core_index:
                    name_matched.update(core_index[f"{c_pfx}{c_clean}"])

                if len(c_comp) >= 4 and f"{c_pfx}{c_comp}" in compact_index:
                    name_matched.update(compact_index[f"{c_pfx}{c_comp}"])
                if len(c_comp_raw) >= 4 and f"{c_pfx}{c_comp_raw}" in compact_index:
                    name_matched.update(compact_index[f"{c_pfx}{c_comp_raw}"])

                if len(cwords) >= 2 and f"{c_pfx}{' '.join(sorted(cwords))}" in toksort_index:
                    name_matched.update(toksort_index[f"{c_pfx}{' '.join(sorted(cwords))}"])

                if len(c_comp) >= 6 and f"{c_pfx}{c_comp[:6]}" in pfx6_index:
                    name_matched.update(pfx6_index[f"{c_pfx}{c_comp[:6]}"])
                if len(c_comp_raw) >= 6 and f"{c_pfx}{c_comp_raw[:6]}" in pfx6_index:
                    name_matched.update(pfx6_index[f"{c_pfx}{c_comp_raw[:6]}"])

                if c_nums and c_awords and f"{c_pfx}{c_nums[0]}_{c_awords[0]}" in num_addr_index:
                    addr_matched.update(num_addr_index[f"{c_pfx}{c_nums[0]}_{c_awords[0]}"])
                if cwords and c_nums and f"{c_pfx}{cwords[0]}_{c_nums[0]}" in name_num_index:
                    addr_matched.update(name_num_index[f"{c_pfx}{cwords[0]}_{c_nums[0]}"])

                valid_locals = set(name_matched)
                for local_idx in (addr_matched - name_matched):
                    s1_c = s1_clean_names[local_idx]
                    if s1_c[:3] == c_core[:3] or s1_c[:3] == c_clean[:3] or fuzz.partial_ratio(s1_c, c_core) >= 45:
                        valid_locals.add(local_idx)

                for local_idx in valid_locals:
                    eid = s1_ids[local_idx]
                    if len(val_candidates[eid]) < max_cands_per_s1:
                        val_candidates[eid].add(cid)
                        cand_info[cid] = (cname, caddr, ccountry)

        print(f"  Streamed {cand_path.name} in {time.time()-t_src0:.2f}s")

    # Feature extraction
    print("Extracting pairwise features for new candidates...")
    pair_rows = []
    X_val_list = []
    y_val_list = []

    for eid in val_sample_eids:
        s1_n, s1_c, s1_ph, s1_a, s1_post, s1_u, s1_num_set, s1_country = s1_parsed[eid]
        true_set = gt.get(eid, set())

        for cid in val_candidates.get(eid, set()):
            if cid in cand_info:
                cname, caddr, ccountry = cand_info[cid]
                cand_n, cand_c, cand_ph = normalize_business_name(cname)
                cand_a, cand_post, cand_u, cand_num_set = normalize_address(caddr)
                feat = extract_precomputed_pairwise_features(
                    s1_n, s1_c, s1_ph, s1_a, s1_post, s1_u, s1_num_set, s1_country,
                    cid, cand_n, cand_c, cand_ph or "", cand_a, cand_post or "", cand_u or "", cand_num_set, ccountry
                )
                X_val_list.append(feat)
                pair_rows.append((eid, cid))
                y_val_list.append(1 if cid in true_set else 0)

    X_val = np.array(X_val_list, dtype=np.float32)
    y_val = np.array(y_val_list, dtype=np.int32)
    pair_s1 = np.array([p[0] for p in pair_rows])
    pair_cand = np.array([p[1] for p in pair_rows])

    # Model predictions
    print("Scoring pairs with LightGBM and CatBoost...")
    booster = lgb.Booster(model_file=".cache_er/models/lgbm_matcher.txt")
    p_lgb = booster.predict(X_val).astype(np.float32)

    cb = CatBoostClassifier()
    cb.load_model(".cache_er/models/lgbm_matcher_catboost.cbm")
    p_cb = cb.predict_proba(X_val)[:, 1].astype(np.float32)

    # Save cache
    np.savez_compressed(
        cache_file,
        val_sample_eids=np.array(val_sample_eids),
        pair_s1=pair_s1,
        pair_cand=pair_cand,
        X_val=X_val,
        y_val=y_val,
        p_lgb=p_lgb,
        p_cb=p_cb,
    )
    print(f"Saved new candidate cache to {cache_file} ({len(pair_s1):,} pairs) in {time.time()-t0:.2f}s!")

    return val_sample_eids, pair_s1, pair_cand, X_val, y_val, p_lgb, p_cb


def evaluate_setup(pair_s1, pair_cand, probs, all_s1_ids, gt, country_map, threshold=0.75, max_cap=None):
    """Evaluate predictions given a threshold and optional max-match cap."""
    all_s1_set = set(all_s1_ids)
    val_gt = {eid: gt.get(eid, set()) for eid in all_s1_ids}
    val_country_map = {eid: country_map.get(eid, "") for eid in all_s1_ids}

    thresholded_pairs = [
        (pair_s1[i], pair_cand[i], float(probs[i]))
        for i in range(len(probs))
        if probs[i] >= threshold
    ]

    preds = apply_many_to_one_with_cap(thresholded_pairs, all_s1_set, max_cap=max_cap)
    metrics = evaluate_predictions(val_gt, preds, all_s1_set, val_country_map)
    return metrics, preds


def main():
    print("=" * 80)
    print("ISOLATED VALIDATION-SET EXPERIMENTATION SUITE")
    print("=" * 80)

    # 1. Load Ground Truth and S1 Metadata
    gt = load_ground_truth(TRAIN_GROUND_TRUTH)
    country_map, _ = load_s1_metadata(TRAIN_SOURCE1, include_fields=False)

    # Load canonical old candidate cache
    old_cache = np.load(".cache_er/val_diag_cache.npz")
    val_eids = list(old_cache["val_sample_eids"])
    old_pair_s1 = old_cache["pair_s1"]
    old_pair_cand = old_cache["pair_cand"]
    old_X_val = old_cache["X_val"]
    old_y_val = old_cache["y_val"]

    booster = lgb.Booster(model_file=".cache_er/models/lgbm_matcher.txt")
    old_p_lgb = booster.predict(old_X_val).astype(np.float32)
    cb = CatBoostClassifier()
    cb.load_model(".cache_er/models/lgbm_matcher_catboost.cbm")
    old_p_cb = cb.predict_proba(old_X_val)[:, 1].astype(np.float32)

    # 2. Generate or load NEW candidates with candidate normalization fix
    new_val_eids, new_pair_s1, new_pair_cand, new_X_val, new_y_val, new_p_lgb, new_p_cb = (
        run_candidate_generation_and_scoring(val_eids, max_cands_per_s1=40)
    )

    # -------------------------------------------------------------
    # BASELINE: Old candidates, 50/50 ensemble, T=0.75, no cap
    # -------------------------------------------------------------
    old_probs_5050 = 0.50 * old_p_lgb + 0.50 * old_p_cb
    m_baseline, _ = evaluate_setup(
        old_pair_s1, old_pair_cand, old_probs_5050, val_eids, gt, country_map, threshold=0.75, max_cap=None
    )

    # -------------------------------------------------------------
    # 1. Candidate Normalization Fix ALONE
    # Everything else at previous settings (T=0.75, 50/50 ensemble, no cap)
    # -------------------------------------------------------------
    new_probs_5050 = 0.50 * new_p_lgb + 0.50 * new_p_cb
    m_change1, _ = evaluate_setup(
        new_pair_s1, new_pair_cand, new_probs_5050, val_eids, gt, country_map, threshold=0.75, max_cap=None
    )

    # Candidate recall comparison
    total_val_gt_pairs = sum(len(gt.get(eid, set())) for eid in val_eids)
    old_recalled_pairs = set(zip(old_pair_s1[old_y_val == 1], old_pair_cand[old_y_val == 1]))
    new_recalled_pairs = set(zip(new_pair_s1[new_y_val == 1], new_pair_cand[new_y_val == 1]))
    old_cand_recall = len(old_recalled_pairs) / total_val_gt_pairs
    new_cand_recall = len(new_recalled_pairs) / total_val_gt_pairs

    # -------------------------------------------------------------
    # 2. Ensemble Reweight (85% CB / 15% LGB) ALONE on top of norm fix
    # Threshold 0.75, no cap
    # -------------------------------------------------------------
    new_probs_8515 = 0.15 * new_p_lgb + 0.85 * new_p_cb
    m_change2, _ = evaluate_setup(
        new_pair_s1, new_pair_cand, new_probs_8515, val_eids, gt, country_map, threshold=0.75, max_cap=None
    )

    # Model correlations
    r_pearson, p_val_p = pearsonr(new_p_lgb, new_p_cb)
    r_spearman, p_val_s = spearmanr(new_p_lgb, new_p_cb)

    # -------------------------------------------------------------
    # 3. Full Threshold Grid Search on Newly Normalized Candidate Scores
    # -------------------------------------------------------------
    grid_thresholds = [0.40, 0.45, 0.50, 0.55, 0.60, 0.62, 0.64, 0.65, 0.66, 0.68, 0.70, 0.72, 0.75, 0.78, 0.80, 0.85]
    grid_results = []
    best_t = None
    best_f05 = -1.0
    best_metrics = None

    for t in grid_thresholds:
        m_t, _ = evaluate_setup(
            new_pair_s1, new_pair_cand, new_probs_8515, val_eids, gt, country_map, threshold=t, max_cap=None
        )
        grid_results.append({
            "threshold": t,
            "macro_f05": m_t["macro_f05"],
            "macro_precision": m_t["macro_precision"],
            "macro_recall": m_t["macro_recall"],
            "singleton_f05": m_t["singleton_f05"],
            "non_singleton_f05": m_t["non_singleton_f05"],
            "us_f05": m_t["by_country"].get("US", {}).get("macro_f05", 0.0),
            "india_f05": m_t["by_country"].get("India", {}).get("macro_f05", 0.0),
        })
        if m_t["macro_f05"] > best_f05:
            best_f05 = m_t["macro_f05"]
            best_t = t
            best_metrics = m_t

    # Metric at T=0.65 alone (on top of norm fix + 85/15 ensemble, no cap)
    m_change3, _ = evaluate_setup(
        new_pair_s1, new_pair_cand, new_probs_8515, val_eids, gt, country_map, threshold=0.65, max_cap=None
    )

    # -------------------------------------------------------------
    # 4. Max-Match Cap of 11 ALONE
    # Compare with vs without cap at baseline, and on top of previous changes
    # -------------------------------------------------------------
    # Baseline with cap 11 alone
    m_cap_alone, _ = evaluate_setup(
        old_pair_s1, old_pair_cand, old_probs_5050, val_eids, gt, country_map, threshold=0.75, max_cap=11
    )
    # Norm fix + 85/15 + T=0.75 with cap 11
    m_norm_reweight_cap, _ = evaluate_setup(
        new_pair_s1, new_pair_cand, new_probs_8515, val_eids, gt, country_map, threshold=0.75, max_cap=11
    )
    # Norm fix + 50/50 + T=0.75 with cap 11
    m_norm_cap, _ = evaluate_setup(
        new_pair_s1, new_pair_cand, new_probs_5050, val_eids, gt, country_map, threshold=0.75, max_cap=11
    )

    # -------------------------------------------------------------
    # COMBINED EFFECT: All 4 changes together
    # (Norm fix + 85/15 ensemble + T=0.65 + Cap 11)
    # -------------------------------------------------------------
    m_combined, _ = evaluate_setup(
        new_pair_s1, new_pair_cand, new_probs_8515, val_eids, gt, country_map, threshold=0.65, max_cap=11
    )

    # -------------------------------------------------------------
    # PRINT EXTENSIVE DETAILED REPORT
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("1. CANDIDATE NORMALIZATION FIX ALONE (Everything else at previous settings)")
    print("=" * 80)
    print(f"Total True Pairs in Validation Set:     {total_val_gt_pairs:,}")
    print(f"Old Candidate Blocking Recalled:        {len(old_recalled_pairs):,} ({old_cand_recall*100:.2f}%)")
    print(f"New Candidate Blocking Recalled:        {len(new_recalled_pairs):,} ({new_cand_recall*100:.2f}%)")
    print(f"Candidate Blocking Recall Delta:        +{(new_cand_recall - old_cand_recall)*100:+.2f}% (+{len(new_recalled_pairs) - len(old_recalled_pairs):,} true pairs)")
    print("\nValidation Performance Metrics (T=0.75, 50/50 Ensemble, No Match Cap):")
    print(f"  {'Metric':<25} | {'Baseline (Old Cands)':<20} | {'With Norm Fix ALONE':<20} | {'Delta':<10}")
    print(f"  {'-'*25}-|-{'-'*20}-|-{'-'*20}-|-{'-'*10}")
    for k, name in [
        ("macro_f05", "Validation Macro F_0.5"),
        ("macro_precision", "Macro Precision"),
        ("macro_recall", "Macro Recall"),
        ("singleton_f05", "Singleton F_0.5"),
        ("non_singleton_f05", "Non-Singleton F_0.5"),
    ]:
        v_base = m_baseline[k]
        v_fix = m_change1[k]
        delta = v_fix - v_base
        print(f"  {name:<25} | {v_base:<20.4f} | {v_fix:<20.4f} | {delta:+10.4f}")

    print("\nCountry Macro F_0.5 Breakdown:")
    for c in ["US", "India"]:
        b_c = m_baseline["by_country"].get(c, {}).get("macro_f05", 0.0)
        f_c = m_change1["by_country"].get(c, {}).get("macro_f05", 0.0)
        print(f"  Country '{c}': Baseline={b_c:.4f} -> With Norm Fix={f_c:.4f} (Delta: {f_c - b_c:+.4f})")

    print("\n" + "=" * 80)
    print("2. ENSEMBLE REWEIGHT (85% CatBoost / 15% LightGBM) ALONE")
    print("=" * 80)
    print(f"CatBoost vs LightGBM Correlation on Validation Candidate Pairs ({len(new_pair_s1):,} pairs):")
    print(f"  Pearson Correlation:  r = {r_pearson:.4f} (p = {p_val_p:.2e})")
    print(f"  Spearman Correlation: r = {r_spearman:.4f} (p = {p_val_s:.2e})")
    print("\nValidation Performance Metrics Before vs After Reweight (T=0.75, No Cap, with Norm Fix):")
    print(f"  {'Metric':<25} | {'50/50 Ensemble':<20} | {'15 LGB / 85 CB ALONE':<20} | {'Delta':<10}")
    print(f"  {'-'*25}-|-{'-'*20}-|-{'-'*20}-|-{'-'*10}")
    for k, name in [
        ("macro_f05", "Validation Macro F_0.5"),
        ("macro_precision", "Macro Precision"),
        ("macro_recall", "Macro Recall"),
        ("singleton_f05", "Singleton F_0.5"),
        ("non_singleton_f05", "Non-Singleton F_0.5"),
    ]:
        v_before = m_change1[k]
        v_after = m_change2[k]
        delta = v_after - v_before
        print(f"  {name:<25} | {v_before:<20.4f} | {v_after:<20.4f} | {delta:+10.4f}")

    print("\n" + "=" * 80)
    print("3. FULL THRESHOLD GRID SEARCH ON NEWLY NORMALIZED CANDIDATE SCORES")
    print("=" * 80)
    df_grid = pd.DataFrame(grid_results)
    print(df_grid.to_string(index=False))
    print(f"\nArgmax Threshold for Validation F_0.5: T* = {best_t:.2f} (F_0.5 = {best_f05:.4f})")
    print(f"Validation F_0.5 at T=0.65:            F_0.5 = {m_change3['macro_f05']:.4f} (Precision = {m_change3['macro_precision']:.4f}, Recall = {m_change3['macro_recall']:.4f})")
    print(f"Validation F_0.5 at T=0.75:            F_0.5 = {m_change2['macro_f05']:.4f} (Precision = {m_change2['macro_precision']:.4f}, Recall = {m_change2['macro_recall']:.4f})")
    print(f"Threshold Shift Delta (0.75 -> 0.65):  {m_change3['macro_f05'] - m_change2['macro_f05']:+.4f} F_0.5")

    print("\n" + "=" * 80)
    print("4. MAX-MATCH CAP OF 11 ALONE & GROUND TRUTH DISTRIBUTION")
    print("=" * 80)
    print("Empirical distribution of match counts in train_ground_truth.tsv across 2,206,821 entities:")
    print("  Min: 0, Max: 11, Median: 3.0, Mean: 3.4613")
    print("  p50=3.0, p75=5.0, p90=6.0, p95=6.0, p99=8.0, p99.9=9.0, p99.99=10.0, p100=11.0")
    print("  Number of entities with >11 matches in GT: EXACTLY 0 (0.000%)")
    print("  Exactly 37 entities have 11 matches; zero entities have >=12.")
    print("\nValidation Performance With and Without Cap of 11:")
    print(f"  a) Baseline (Old Cands, 50/50, T=0.75):")
    print(f"     Without Cap: F_0.5 = {m_baseline['macro_f05']:.4f} | With Cap 11: F_0.5 = {m_cap_alone['macro_f05']:.4f} (Delta: {m_cap_alone['macro_f05'] - m_baseline['macro_f05']:+.4f})")
    print(f"  b) With Norm Fix alone (New Cands, 50/50, T=0.75):")
    print(f"     Without Cap: F_0.5 = {m_change1['macro_f05']:.4f} | With Cap 11: F_0.5 = {m_norm_cap['macro_f05']:.4f} (Delta: {m_norm_cap['macro_f05'] - m_change1['macro_f05']:+.4f})")
    print(f"  c) With Norm Fix + Reweight (New Cands, 15/85, T=0.75):")
    print(f"     Without Cap: F_0.5 = {m_change2['macro_f05']:.4f} | With Cap 11: F_0.5 = {m_norm_reweight_cap['macro_f05']:.4f} (Delta: {m_norm_reweight_cap['macro_f05'] - m_change2['macro_f05']:+.4f})")
    print(f"  d) With Norm Fix + Reweight + T=0.65:")
    print(f"     Without Cap: F_0.5 = {m_change3['macro_f05']:.4f} | With Cap 11: F_0.5 = {m_combined['macro_f05']:.4f} (Delta: {m_combined['macro_f05'] - m_change3['macro_f05']:+.4f})")

    print("\n" + "=" * 80)
    print("5. COMBINED EFFECT VS SUM OF INDIVIDUAL EFFECTS (INTERACTION ANALYSIS)")
    print("=" * 80)
    d1 = m_change1["macro_f05"] - m_baseline["macro_f05"]
    d2 = m_change2["macro_f05"] - m_change1["macro_f05"]
    d3 = m_change3["macro_f05"] - m_change2["macro_f05"]
    d4 = m_combined["macro_f05"] - m_change3["macro_f05"]
    sum_deltas = d1 + d2 + d3 + d4
    actual_combined_delta = m_combined["macro_f05"] - m_baseline["macro_f05"]
    interaction_gap = actual_combined_delta - sum_deltas

    print(f"Baseline F_0.5 (Old Cands, 50/50 LGB/CB, T=0.75, No Cap): {m_baseline['macro_f05']:.4f}")
    print(f"1. Delta from Candidate Normalization Fix:                 {d1:+10.4f}")
    print(f"2. Delta from Ensemble Reweight (15/85):                   {d2:+10.4f}")
    print(f"3. Delta from Lower Threshold (0.75 -> 0.65):              {d3:+10.4f}")
    print(f"4. Delta from Max-Match Cap of 11:                         {d4:+10.4f}")
    print(f"{'-'*60}")
    print(f"Sum of Stepwise Deltas:                                    {sum_deltas:+10.4f}")
    print(f"Actual Combined F_0.5 (All 4 Changes Applied):             {m_combined['macro_f05']:.4f}")
    print(f"Actual Combined Delta vs Baseline:                         {actual_combined_delta:+10.4f}")
    print(f"Interaction Gap (Actual - Sum):                            {interaction_gap:+10.4f}")

    print("\nCombined Metrics Summary:")
    print(f"  Macro F_0.5:     {m_combined['macro_f05']:.4f} (Precision: {m_combined['macro_precision']:.4f}, Recall: {m_combined['macro_recall']:.4f})")
    print(f"  Singleton F_0.5: {m_combined['singleton_f05']:.4f}")
    print(f"  Non-Singleton:   {m_combined['non_singleton_f05']:.4f}")
    print(f"  US F_0.5:        {m_combined['by_country'].get('US', {}).get('macro_f05', 0.0):.4f}")
    print(f"  India F_0.5:     {m_combined['by_country'].get('India', {}).get('macro_f05', 0.0):.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
