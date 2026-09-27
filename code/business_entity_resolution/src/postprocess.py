"""
Global consistency post-processing module for Business Entity Resolution.
Implements many-to-one consistency (Source-1 entities can match multiple S2/S3 records,
but each S2/S3 record can belong to at most one Source-1 entity).
Uses greedy score sorting without bipartite restriction.
"""

from typing import Dict, List, Optional, Set, Tuple


def apply_many_to_one_consistency(
    scored_pairs: List[Tuple[str, str, float]],
    all_s1_ids: Set[str],
    contested_margin_drop: Optional[float] = None,
) -> Dict[str, Set[str]]:
    """
    Enforce global many-to-one consistency:
    - Each S2/S3 candidate ID is assigned to at most one Source-1 entity.
    - If multiple S1 entities claim the same S2/S3 ID, assign it to the S1 entity
      with the highest predicted probability.
    - If contested_margin_drop is specified and (top_prob - runner_up_prob) < margin,
      drop the candidate to avoid false merges.

    Parameters:
    -----------
    scored_pairs: list of (s1_id, candidate_id, probability) above threshold
    all_s1_ids: complete set of all S1 IDs (ensures every entity has an entry, even singletons)
    contested_margin_drop: optional float margin (e.g. 0.05). If difference between
      top and second score is less than this margin, neither gets the match.

    Returns:
    --------
    mapping: dict mapping s1_id -> set of matched entity IDs
    """
    # Initialize output for all S1 entities (singletons start with empty set)
    final_matches: Dict[str, Set[str]] = {s1_id: set() for s1_id in all_s1_ids}

    # Group claims by candidate_id: cand_id -> list of (s1_id, prob)
    claims_by_candidate: Dict[str, List[Tuple[str, float]]] = {}
    for s1_id, cand_id, prob in scored_pairs:
        claims_by_candidate.setdefault(cand_id, []).append((s1_id, prob))

    for cand_id, claims in claims_by_candidate.items():
        if len(claims) == 1:
            # Uncontested match: assign directly
            s1_id, _ = claims[0]
            if s1_id in final_matches:
                final_matches[s1_id].add(cand_id)
        else:
            # Contested match: sort descending by score
            claims.sort(key=lambda x: x[1], reverse=True)
            top_s1, top_prob = claims[0]
            runner_up_s1, runner_up_prob = claims[1]

            if (
                contested_margin_drop is not None
                and (top_prob - runner_up_prob) < contested_margin_drop
            ):
                # Unclear distinction: drop candidate to avoid false positive
                continue
            else:
                # Winner takes candidate
                if top_s1 in final_matches:
                    final_matches[top_s1].add(cand_id)

    return final_matches
