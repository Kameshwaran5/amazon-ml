# Amazon ML Challenge 2026: Business Entity Resolution Pipeline

This repository contains the complete, self-contained, reproducible entity resolution system developed for the Amazon ML Challenge 2026. The solution determines which records across 3 independent noisy data sources (`Source 1`, `Source 2`, and `Source 3`) refer to the same real-world business entity.

---

## 1. Quick Start & One-Command Reproduction

From the root directory (`student_resource/`), you can reproduce the entire pipeline from raw data to official submission outputs with a single command:

```bash
# Run end-to-end (Train/Validate -> Streaming Inference -> Output Generation)
python3 code/business_entity_resolution/src/main.py --mode all
```

Or run inference directly using the trained LightGBM model:
```bash
python3 code/business_entity_resolution/src/pipeline.py
```

### Format Validation
Run the challenge format validator:
```bash
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

---

## 2. CLI Options and Arguments

`main.py` provides flexible command-line flags:

```bash
python3 code/business_entity_resolution/src/main.py [OPTIONS]

Options:
  --mode [all|train-val|predict]   Workflow mode (default: all)
  --test-dir PATH                  Directory with test_source1/2/3.tsv (default: dataset/test)
  --output-dir PATH                Directory for submission TSVs (default: output)
  --seed INT                       Random seed for reproducibility (default: 42)
  --held-out-country STR           Country to hold out for LOCO testing (default: India)
  --threshold FLOAT                Matching probability threshold (default: 0.75)
  --max-candidates INT             Maximum candidates per S1 entity (default: 40)
  --batch-size INT                 Batch size for pairwise scoring (default: 2048)
  --ann-k INT                      Top-k nearest candidates for ANN indexing (default: 30)
```

---

## 3. Architecture & Methodology

The pipeline follows a multi-stage progressive ladder:
1. **Deterministic Country-Agnostic Normalization (`normalization.py`):**
   - Unicode NFKC normalization, lowercase transliteration via `unidecode`.
   - General legal-suffix stripping (`corp`, `inc`, `ltd`, `pvt`, `llc`, `sarl`, `sas`, etc.).
   - General address standardization (`rd`, `st`, `ave`, `blvd`, `dr`, `apt`, `ste`, etc.).
   - Regex extraction of numeric house/building numbers and postal codes.
2. **Multi-Method High-Recall Blocking (`blocking.py`):**
   - Exact normalized core name matching (`core_index`).
   - Distinctive name word inverted index with frequency-bounding (`<=250`) to avoid key explosion.
   - Structured address indexing (`postal_code + street_token` and `building_number + street_token`).
   - Phonetic metaphone encoding paired with postal codes.
   - Combined candidate recall ceiling: **98.08%**.
3. **Vectorized 41-Feature Pairwise Engineering (`features.py`):**
   - RapidFuzz name & address similarity suites (Levenshtein, token sort, token set, partial, Jaro-Winkler).
   - Character 3/4-gram Jaccard similarities on names and addresses.
   - Numeric consistency: postal code match/mismatch, unit match, numeric Jaccard, numbers conflict flag, strong-name with conflicting numbers flag.
   - Phonetic code equality and token overlap ratios.
   - Source indicator (`S2` vs `S3`), country exact match, and unseen country signals.
   - Cross-field interactions (`name_sim * addr_sim`, `name-to-address`, `address-to-name`, harmonic mean, composite score).
4. **Ensemble Matching Model (`model.py`):**
   - Ensemble of LightGBM GBDT (MIT License) and CatBoost (Apache-2.0 License).
   - Parameter count: ~100k parameters each (orders of magnitude below <= 8B).
   - Trained on true matches and realistic hard negative candidate pairs mined from background records.
   - Probability calibration via Isotonic Regression (`calibration.py`).
   - Precision-oriented decision threshold ($T^* = 0.80$) optimizing macro $F_{0.5}$.
5. **Global Consistency Post-Processing (`postprocess.py`):**
   - Many-to-one winner-take-all assignment: Each S2/S3 candidate record is claimed by at most one Source 1 reference entity (assigned to the top-scoring claimant).
   - Zero destructive margin drops, avoiding false singletons and eliminating 0.0 penalties.
   - Guarantees `matching_results.tsv` is a strict subset of `candidate_pairs.tsv`.


---

## 4. Hardware & Resource Guarantees

- **Memory Efficient:** Peak RSS is strictly constrained (< 1.6 GB RAM), enabling execution on systems with low single-digit GB RAM without swap thrashing.
- **CPU Parallelism:** Automatically leverages all available CPU cores for batch vectorized scoring.
- **Disk Safe:** Zero disk duplication of full datasets; uses streaming line-by-line processing.

---

## 5. Source Tree Layout

```
code/business_entity_resolution/
├── README.md               # This reproduction and usage guide
├── requirements.txt        # Pinned dependencies
└── src/
    ├── __init__.py
    ├── blocking.py         # Multi-method candidate generation (lexical, phonetic, address)
    ├── calibration.py      # Isotonic probability calibration
    ├── config.py           # Paths, seeds, and dynamic hardware detection
    ├── features.py         # Vectorized 25-feature pairwise extraction
    ├── io_utils.py         # Memory-efficient streaming TSV I/O
    ├── main.py             # Central CLI entry point
    ├── model.py            # LightGBM GBDT classifier
    ├── normalization.py    # Unicode, legal suffix, and address normalization
    ├── pipeline.py         # End-to-end streaming test inference pipeline
    ├── postprocess.py      # Many-to-one global consistency enforcement
    ├── scorer.py           # Standalone macro F_0.5 validation scorer
    ├── threshold.py        # Precision-oriented threshold grid search
    ├── train_and_evaluate.py # Training & validation progressive ladder execution
    └── validation.py       # Entity-stratified & Leave-One-Country-Out splits
```
