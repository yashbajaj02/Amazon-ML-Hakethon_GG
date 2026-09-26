# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** GG  
**Team Members:** Khusavant, Yash Bajaj  
**Submission Date:** September 2026

---

## 1. Executive Summary
We developed a scalable two-stage Business Entity Resolution framework consisting of country-partitioned sublinear character n-gram TF-IDF blocking coupled with a pairwise Gradient Boosted Decision Tree (LightGBM) matcher. Pairwise similarity features are extracted across normalized names, addresses, and geographic markers. The decision threshold was calibrated directly on a stratified validation set to maximize the macro-averaged $F_{0.5}$ score, explicitly penalizing false merges and protecting singletons.

---

## 2. Methodology

### 2.1 Problem Analysis
- **Scale:** The dataset comprises ~2.2M reference Source 1 entities and ~10M Source 2/3 records, making exhaustive $O(N^2)$ pairwise comparison intractable (~8.8 trillion pairs).
- **Singletons:** 5.58% of Source 1 entities have zero true matches in Source 2/3. In macro $F_{0.5}$, singletons award 1.0 for an empty prediction and 0.0 for any false merge, demanding conservative decision boundaries.
- **Unseen Countries:** While training data covers `US` and `India`, the test set introduces `France`. Country blocking must remain dynamic and robust to unseen territory labels.
- **Noise Patterns:** Significant variations in legal entity suffixes (Corp vs Corporation, Ltd vs Limited), address token permutations, and missing components.

### 2.2 Solution Strategy

**Approach Type:** Multi-Key Blocking + Pairwise GBDT Classifier (LightGBM)  
**Core Innovation:** Sublinear character n-gram TF-IDF candidate generation executed in chunked matrix batches (bounding memory) combined with RapidFuzz token-level similarity features and macro $F_{0.5}$ threshold calibration.

---

## 3. Candidate Generation (Blocking)

- **Blocking keys used:** Country partitioning, sublinear character 3-5 gram TF-IDF vectorization with sparse cosine similarity top-K retrieval (top 30 candidates per entity).
- **Candidate pairs generated:** Reduced comparison space by >99.88% (from 260M to ~150K pairs per 5k sample).
- **How you ensured true matches were not lost:** Sublinear term frequency prevents high-frequency stopwords from dominating, and character n-grams capture misspellings, prefixes, and transliterations. This achieved a **99.44% candidate recall ceiling** on validation data.

---

## 4. Matching Model

**Features used:**
- Name features: `name_ratio`, `name_partial`, `name_token_sort`, `name_token_set`, `name_jaccard`, `exact_name`
- Address features: `addr_ratio`, `addr_partial`, `addr_token_sort`, `addr_token_set`, `addr_jaccard`, `exact_addr`
- Structural & Guardrail features: `num_conflict` (conflicting building numbers), `num_exact`, `num_overlap`, `branch_mismatch` (asymmetric presence of division tokens like central, west, holdings, services)
- Cross-entity features: Relative length differences in name and address, country equality indicator (`same_country`)

**Model type:** LightGBM Binary Classifier (`n_estimators=300`, `learning_rate=0.05`, `num_leaves=31`, `max_depth=6`) trained with hard-negative mining (220,000 distractor pairs from same street/city).  
**Threshold selection method:** Grid search threshold optimization on an entity-stratified validation split directly targeting macro $F_{0.5}$ (Optimal Calibrated Threshold: **0.60**).

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** **0.9774** on holdout validation data with hard-negative distractors.
- **Candidate Blocking Recall:** **97.70%** (via pruned IDF token index)
- **Singleton Accuracy:** **94.02%** (5.98% predicted empty vs 5.58% ground truth)
- **Common false positives (wrong merges):** Franchises or corporate chains sharing near-identical names on the same avenue (largely mitigated by `num_conflict` and `branch_mismatch`).
- **Common false negatives (missed matches):** Drastically abbreviated names with non-overlapping address notations.

---

## 6. Conclusion
The combination of memory-bounded character n-gram blocking and pairwise LightGBM classification achieves high candidate recall and precision on the Amazon ML Challenge 2026. The solution operates fully offline, uses an MIT-licensed lightweight model (< 5MB), and satisfies all competition constraints.

---

## 7. Compliance Statement
- Model License: MIT License (LightGBM, Scikit-learn, RapidFuzz)
- Parameter Count: < 5 Million (well under 8B limit)
- External Lookups: Zero external APIs, geocoders, or web queries.
