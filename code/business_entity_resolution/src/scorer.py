"""
Standalone local F_0.5 metric scorer for Business Entity Resolution.
Evaluates macro-averaged F_0.5 with exact handling of all edge cases:
- F_0.5 = (1.25 * P * R) / (0.25 * P + R)
- singleton + 0 predicted matches -> 1.0
- singleton + >=1 predicted match -> 0.0
- non-singleton + P=0 + R=0 -> 0.0
- zero-predictions-for-non-singleton -> 0.0
- all-false-positives -> 0.0
"""

import argparse
from typing import Dict, Iterable, Optional, Set, Tuple
import numpy as np


def compute_entity_metrics(
    true_matches: Set[str], pred_matches: Set[str]
) -> Tuple[float, float, float]:
    """
    Compute (F_0.5, Precision, Recall) for a single Source 1 entity.

    Parameters:
    -----------
    true_matches: set of true matched entity IDs (empty for singletons)
    pred_matches: set of predicted matched entity IDs (empty for no match)

    Returns:
    --------
    (f_05, precision, recall)
    """
    n_true = len(true_matches)
    n_pred = len(pred_matches)

    # Edge cases for singletons (true_matches is empty)
    if n_true == 0:
        if n_pred == 0:
            # Correctly predicted singleton: full score
            return 1.0, 1.0, 1.0
        else:
            # False merge on singleton: zero score
            return 0.0, 0.0, 1.0

    # Non-singleton entity (true_matches has >= 1 elements)
    if n_pred == 0:
        # Zero predictions for non-singleton: zero score
        return 0.0, 0.0, 0.0

    tp = len(true_matches.intersection(pred_matches))
    if tp == 0:
        # All false positives
        return 0.0, 0.0, 0.0

    precision = tp / n_pred
    recall = tp / n_true

    # F_0.5 with beta = 0.5: (1 + beta^2) * P * R / (beta^2 * P + R)
    # = (1.25 * P * R) / (0.25 * P + R)
    denom = 0.25 * precision + recall
    if denom <= 0.0:
        return 0.0, precision, recall

    f_05 = (1.25 * precision * recall) / denom
    return f_05, precision, recall


def evaluate_predictions(
    ground_truth: Dict[str, Set[str]],
    predictions: Dict[str, Set[str]],
    s1_entity_ids: Optional[Iterable[str]] = None,
    country_map: Optional[Dict[str, str]] = None,
) -> Dict[str, any]:
    """
    Compute macro-averaged F_0.5 across all specified Source-1 entities.

    Parameters:
    -----------
    ground_truth: dict mapping source1_entity_id -> set of true matched entity IDs
    predictions: dict mapping source1_entity_id -> set of predicted matched entity IDs
    s1_entity_ids: optional iterable of source1_entity_ids to evaluate on (evaluates all in ground_truth if None)
    country_map: optional dict mapping source1_entity_id -> country string

    Returns:
    --------
    Dict containing overall, singleton, non-singleton, and by-country macro F_0.5 metrics.
    """
    if s1_entity_ids is None:
        eval_ids = list(ground_truth.keys())
    else:
        eval_ids = list(s1_entity_ids)

    n_entities = len(eval_ids)
    if n_entities == 0:
        return {
            "macro_f05": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "singleton_f05": 0.0,
            "non_singleton_f05": 0.0,
            "total_entities": 0,
            "singleton_count": 0,
            "non_singleton_count": 0,
            "by_country": {},
        }

    f05_scores = np.zeros(n_entities, dtype=np.float64)
    prec_scores = np.zeros(n_entities, dtype=np.float64)
    rec_scores = np.zeros(n_entities, dtype=np.float64)
    is_singleton = np.zeros(n_entities, dtype=bool)

    country_scores: Dict[str, list] = {}

    for idx, s1_id in enumerate(eval_ids):
        true_set = ground_truth.get(s1_id, set())
        pred_set = predictions.get(s1_id, set())

        f05, prec, rec = compute_entity_metrics(true_set, pred_set)
        f05_scores[idx] = f05
        prec_scores[idx] = prec
        rec_scores[idx] = rec

        if len(true_set) == 0:
            is_singleton[idx] = True

        if country_map and s1_id in country_map:
            c = country_map[s1_id]
            if c not in country_scores:
                country_scores[c] = []
            country_scores[c].append(f05)

    macro_f05 = float(np.mean(f05_scores))
    macro_precision = float(np.mean(prec_scores))
    macro_recall = float(np.mean(rec_scores))

    singleton_mask = is_singleton
    non_singleton_mask = ~is_singleton

    singleton_f05 = (
        float(np.mean(f05_scores[singleton_mask]))
        if np.any(singleton_mask)
        else 0.0
    )
    non_singleton_f05 = (
        float(np.mean(f05_scores[non_singleton_mask]))
        if np.any(non_singleton_mask)
        else 0.0
    )

    by_country_results = {}
    for c, scores in country_scores.items():
        by_country_results[c] = {
            "macro_f05": float(np.mean(scores)),
            "count": len(scores),
        }

    return {
        "macro_f05": macro_f05,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "singleton_f05": singleton_f05,
        "non_singleton_f05": non_singleton_f05,
        "total_entities": n_entities,
        "singleton_count": int(np.sum(singleton_mask)),
        "non_singleton_count": int(np.sum(non_singleton_mask)),
        "by_country": by_country_results,
    }


def verify_synthetic_cases() -> bool:
    """
    Verify F_0.5 calculations against the required synthetic test cases:
    1. singleton + no-predictions -> 1.0
    2. singleton + false prediction -> 0.0
    3. perfect match -> 1.0
    4. partial recall (e.g. true={A,B}, pred={A,B,C} and pred={A})
    5. all-false-positives -> 0.0
    6. zero-predictions-for-non-singleton -> 0.0
    """
    tests = [
        # Case 1: singleton + no-predictions -> 1.0
        ("singleton + no-predictions", set(), set(), 1.0, 1.0, 1.0),
        # Case 2: singleton + false prediction -> 0.0
        ("singleton + false prediction", set(), {"S2-100"}, 0.0, 0.0, 1.0),
        # Case 3: perfect match -> 1.0
        ("perfect match (single)", {"S2-100"}, {"S2-100"}, 1.0, 1.0, 1.0),
        ("perfect match (multi)", {"S2-100", "S3-200"}, {"S2-100", "S3-200"}, 1.0, 1.0, 1.0),
        # Case 4a: partial recall / precision tradeoff (P=2/3, R=2/2=1.0)
        # F_0.5 = (1.25 * (2/3) * 1.0) / (0.25 * (2/3) + 1.0) = 5/7 ≈ 0.7142857
        ("partial recall with FP (P=2/3, R=1)", {"S2-47", "S3-812"}, {"S2-47", "S2-193", "S3-812"}, 5 / 7, 2 / 3, 1.0),
        # Case 4b: partial recall without FP (P=1.0, R=1/2=0.5)
        # F_0.5 = (1.25 * 1.0 * 0.5) / (0.25 * 1.0 + 0.5) = 0.625 / 0.75 = 5/6 ≈ 0.8333333
        ("partial recall without FP (P=1, R=1/2)", {"S2-100", "S3-200"}, {"S2-100"}, 5 / 6, 1.0, 0.5),
        # Case 5: all-false-positives -> 0.0
        ("all-false-positives", {"S2-100"}, {"S2-999", "S3-888"}, 0.0, 0.0, 0.0),
        # Case 6: zero-predictions-for-non-singleton -> 0.0
        ("zero-predictions-for-non-singleton", {"S2-100", "S3-200"}, set(), 0.0, 0.0, 0.0),
    ]

    all_passed = True
    print("\n--- Verifying Synthetic Scorer Cases ---")
    for name, true_set, pred_set, expected_f05, expected_p, expected_r in tests:
        f05, p, r = compute_entity_metrics(true_set, pred_set)
        f05_ok = np.isclose(f05, expected_f05, atol=1e-6)
        p_ok = np.isclose(p, expected_p, atol=1e-6)
        r_ok = np.isclose(r, expected_r, atol=1e-6)
        passed = f05_ok and p_ok and r_ok

        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {name}: F0.5={f05:.4f} (exp={expected_f05:.4f}), P={p:.4f} (exp={expected_p:.4f}), R={r:.4f} (exp={expected_r:.4f})")
        if not passed:
            all_passed = False

    return all_passed


if __name__ == "__main__":
    passed = verify_synthetic_cases()
    if not passed:
        raise SystemExit("Synthetic test cases failed!")
    print("All synthetic scorer test cases PASSED successfully.")
