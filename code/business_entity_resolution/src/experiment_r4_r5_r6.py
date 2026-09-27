"""
Experiment Harness for Stages R4, R5, R6.
Evaluates entity-level aggregate features, correlation-checked ensemble reweighting,
and precision-oriented threshold optimization.
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
import lightgbm as lgb
from catboost import CatBoostClassifier

# Ensure code and src directories are in sys.path
_src_dir = Path(__file__).resolve().parent
_code_dir = _src_dir.parents[1]
for _p in [str(_src_dir), str(_code_dir)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from business_entity_resolution.src.config import (
    CACHE_DIR,
    TRAIN_GROUND_TRUTH,
    TRAIN_SOURCE1,
)
from business_entity_resolution.src.io_utils import (
    load_ground_truth,
    load_s1_metadata,
)
from business_entity_resolution.src.features import FEATURE_NAMES
from business_entity_resolution.src.postprocess import apply_many_to_one_consistency
from business_entity_resolution.src.scorer import evaluate_predictions, compute_entity_metrics


def compute_entity_aggregates(pair_s1: np.ndarray, pair_cand: np.ndarray, X: np.ndarray, feature_names: list) -> np.ndarray:
    """
    Compute 11 entity-level aggregate features for each candidate pair:
    1. s1_cand_count
    2. cand_frac_s2
    3. cand_frac_s3
    4. s1_has_postal (derived from postal_code_match != 0 or postal features)
    5. s1_has_building_num
    6. cand_has_postal
    7. cand_has_building_num
    8. s1_cand_max_score (using overall_composite_score)
    9. s1_cand_mean_score
    10. s1_cand_std_score
    11. s1_score_gap_1_2
    """
    comp_idx = feature_names.index("overall_composite_score")
    post_idx = feature_names.index("postal_code_match")
    unit_idx = feature_names.index("unit_number_match")
    is_s2_idx = feature_names.index("candidate_source_is_s2")

    # Group by s1
    s1_indices = {}
    for idx, s1 in enumerate(pair_s1):
        s1_indices.setdefault(s1, []).append(idx)

    n_pairs = len(pair_s1)
    extra_features = np.zeros((n_pairs, 11), dtype=np.float32)

    for s1, idx_list in s1_indices.items():
        k = len(idx_list)
        scores = X[idx_list, comp_idx]
        sorted_scores = np.sort(scores)[::-1]

        max_s = float(sorted_scores[0])
        mean_s = float(np.mean(sorted_scores))
        std_s = float(np.std(sorted_scores)) if k > 1 else 0.0
        gap_1_2 = float(sorted_scores[0] - sorted_scores[1]) if k > 1 else max_s

        s2_count = sum(1 for i in idx_list if X[i, is_s2_idx] == 1.0)
        frac_s2 = s2_count / k
        frac_s3 = 1.0 - frac_s2

        # Check if s1 has postal or unit (any non-zero in matching features across candidates)
        has_post = 1.0 if np.any(X[idx_list, post_idx] != 0.0) else 0.0
        has_unit = 1.0 if np.any(X[idx_list, unit_idx] != 0.0) else 0.0

        for i in idx_list:
            c_has_post = 1.0 if X[i, post_idx] != 0.0 else 0.0
            c_has_unit = 1.0 if X[i, unit_idx] != 0.0 else 0.0
            extra_features[i] = [
                float(k),
                frac_s2,
                frac_s3,
                has_post,
                has_unit,
                c_has_post,
                c_has_unit,
                max_s,
                mean_s,
                std_s,
                gap_1_2,
            ]

    return extra_features


def run_experiments():
    print("=" * 60)
    print("RUNNING STAGES R4, R5, R6 EXPERIMENTS")
    print("=" * 60)

    data = np.load(".cache_er/val_diag_cache.npz")
    val_eids = list(data["val_sample_eids"])
    pair_s1 = data["pair_s1"]
    pair_cand = data["pair_cand"]
    X_val = data["X_val"]
    y_val = data["y_val"]

    gt = load_ground_truth(TRAIN_GROUND_TRUTH)
    country_map, _ = load_s1_metadata(TRAIN_SOURCE1, include_fields=False)

    val_sample_set = set(val_eids)
    val_gt = {eid: gt.get(eid, set()) for eid in val_eids}
    val_country_map = {eid: country_map.get(eid, "") for eid in val_eids}

    # Load existing models
    booster = lgb.Booster(model_file=".cache_er/models/lgbm_matcher.txt")
    p_lgb = booster.predict(X_val)
    cb = CatBoostClassifier()
    cb.load_model(".cache_er/models/lgbm_matcher_catboost.cbm")
    p_cb = cb.predict_proba(X_val)[:, 1]

    # STAGE R5: Correlation check
    print("\n--- STAGE R5: ENSEMBLE CORRELATION & REWEIGHTING ---")
    r_pearson, _ = pearsonr(p_lgb, p_cb)
    r_spearman, _ = spearmanr(p_lgb, p_cb)
    print(f"LGBM vs CatBoost Correlation: Pearson={r_pearson:.4f}, Spearman={r_spearman:.4f}")

    results_table = []

    # Exp 1: Baseline 50/50 blend at T=0.75
    p_base = 0.5 * p_lgb + 0.5 * p_cb
    t_pairs = [(pair_s1[i], pair_cand[i], float(p_base[i])) for i in range(len(p_base)) if p_base[i] >= 0.75]
    preds = apply_many_to_one_consistency(t_pairs, val_sample_set)
    m = evaluate_predictions(val_gt, preds, val_sample_set, val_country_map)
    results_table.append({
        "Experiment": "Exp 1: Baseline 50/50 Ensemble (T=0.75)",
        "F_0.5": m["macro_f05"],
        "Precision": m["macro_precision"],
        "Recall": m["macro_recall"],
        "Singleton_F05": m["singleton_f05"],
        "US_F05": m["by_country"]["US"]["macro_f05"],
        "India_F05": m["by_country"]["India"]["macro_f05"],
    })

    # Exp 2: Reweighted ensemble (0.15 LGB + 0.85 CB) at T=0.75
    p_reweighted = 0.15 * p_lgb + 0.85 * p_cb
    t_pairs = [(pair_s1[i], pair_cand[i], float(p_reweighted[i])) for i in range(len(p_reweighted)) if p_reweighted[i] >= 0.75]
    preds = apply_many_to_one_consistency(t_pairs, val_sample_set)
    m = evaluate_predictions(val_gt, preds, val_sample_set, val_country_map)
    results_table.append({
        "Experiment": "Exp 2: Reweighted (0.15 LGB + 0.85 CB, T=0.75)",
        "F_0.5": m["macro_f05"],
        "Precision": m["macro_precision"],
        "Recall": m["macro_recall"],
        "Singleton_F05": m["singleton_f05"],
        "US_F05": m["by_country"]["US"]["macro_f05"],
        "India_F05": m["by_country"]["India"]["macro_f05"],
    })

    # STAGE R4: Compute entity-level aggregate features
    print("\n--- STAGE R4: ENTITY-LEVEL AGGREGATE FEATURES ---")
    extra_val_features = compute_entity_aggregates(pair_s1, pair_cand, X_val, FEATURE_NAMES)
    X_val_augmented = np.hstack([X_val, extra_val_features])
    print(f"Augmented feature matrix shape: {X_val_augmented.shape} (added 11 features)")

    # Exp 3: Train CatBoost with Augmented Features on val CV (5-fold entity-grouped)
    from sklearn.model_selection import GroupKFold
    gkf = GroupKFold(n_splits=3)
    oof_preds = np.zeros(len(y_val), dtype=np.float32)

    for fold, (trn_idx, val_idx) in enumerate(gkf.split(X_val_augmented, y_val, groups=pair_s1)):
        cb_aug = CatBoostClassifier(
            iterations=250,
            learning_rate=0.05,
            depth=6,
            loss_function="Logloss",
            eval_metric="Logloss",
            random_seed=42 + fold,
            verbose=0,
            thread_count=-1,
        )
        cb_aug.fit(
            X_val_augmented[trn_idx],
            y_val[trn_idx],
            eval_set=(X_val_augmented[val_idx], y_val[val_idx]),
            early_stopping_rounds=25,
        )
        oof_preds[val_idx] = cb_aug.predict_proba(X_val_augmented[val_idx])[:, 1]
        print(f"  Fold {fold+1} trained successfully!")

    # Evaluate OOF with augmented features at T=0.75
    t_pairs = [(pair_s1[i], pair_cand[i], float(oof_preds[i])) for i in range(len(oof_preds)) if oof_preds[i] >= 0.75]
    preds = apply_many_to_one_consistency(t_pairs, val_sample_set)
    m = evaluate_predictions(val_gt, preds, val_sample_set, val_country_map)
    results_table.append({
        "Experiment": "Exp 3: Augmented Features (CatBoost OOF, T=0.75)",
        "F_0.5": m["macro_f05"],
        "Precision": m["macro_precision"],
        "Recall": m["macro_recall"],
        "Singleton_F05": m["singleton_f05"],
        "US_F05": m["by_country"]["US"]["macro_f05"],
        "India_F05": m["by_country"]["India"]["macro_f05"],
    })

    # STAGE R6: Grid search threshold on augmented model
    print("\n--- STAGE R6: THRESHOLD & MARGIN GRID SEARCH ---")
    best_t = 0.75
    best_f05 = 0.0
    best_m = None

    for t in [0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.85]:
        t_pairs = [(pair_s1[i], pair_cand[i], float(oof_preds[i])) for i in range(len(oof_preds)) if oof_preds[i] >= t]
        preds = apply_many_to_one_consistency(t_pairs, val_sample_set)
        curr_m = evaluate_predictions(val_gt, preds, val_sample_set, val_country_map)
        print(f"  T={t:.2f}: Macro F_0.5={curr_m['macro_f05']:.4f}, P={curr_m['macro_precision']:.4f}, R={curr_m['macro_recall']:.4f}, Singletons={curr_m['singleton_f05']:.4f}")
        if curr_m["macro_f05"] > best_f05:
            best_f05 = curr_m["macro_f05"]
            best_t = t
            best_m = curr_m

    results_table.append({
        "Experiment": f"Exp 4: Optimal Threshold (T={best_t:.2f})",
        "F_0.5": best_m["macro_f05"],
        "Precision": best_m["macro_precision"],
        "Recall": best_m["macro_recall"],
        "Singleton_F05": best_m["singleton_f05"],
        "US_F05": best_m["by_country"]["US"]["macro_f05"],
        "India_F05": best_m["by_country"]["India"]["macro_f05"],
    })

    print("\n" + "=" * 60)
    print("EXPERIMENT DISCIPLINE TABLE")
    print("=" * 60)
    df_res = pd.DataFrame(results_table)
    print(df_res.to_string(index=False))


if __name__ == "__main__":
    run_experiments()
