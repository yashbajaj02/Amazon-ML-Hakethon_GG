# Business Entity Resolution Pipeline

This repository contains the end-to-end Machine Learning pipeline for the **Amazon ML Challenge 2026: Business Entity Resolution Challenge**.

---

## 1. Setup & Environment

Ensure you have Python 3.8+ installed. Install required packages:

```bash
pip install -r requirements.txt
```

---

## 2. Directory Structure

```text
code/business_entity_resolution/
├── src/
│   ├── config.py                  # Paths and hyperparameter configurations
│   ├── data/
│   │   ├── loader.py              # TSV file reader and ground truth parser
│   │   └── preprocessor.py        # Text & address normalization
│   ├── blocking/
│   │   └── candidate_generator.py # Candidate blocking via TF-IDF & country grouping
│   ├── features/
│   │   └── feature_extractor.py   # Multi-metric string similarity features
│   ├── models/
│   │   └── matcher.py             # Pairwise LightGBM classifier & threshold optimizer
│   ├── evaluation/
│   │   └── metrics.py             # Macro F_0.5 evaluator (including singletons)
│   └── pipeline/
│       └── run_pipeline.py        # End-to-end execution runner
├── README.md                      # Pipeline reproduction guide
└── requirements.txt               # Pinned dependencies
```

---

## 3. Running the Pipeline

To run the pipeline on the test dataset and generate both `candidate_pairs.tsv` and `matching_results.tsv` in the `output/` directory:

```bash
python3 -m src.pipeline.run_pipeline --split test
```

To run on training data for validation:

```bash
python3 -m src.pipeline.run_pipeline --split train
```

---

## 4. Validating the Submission

Run the challenge validator to ensure both files conform to all competition requirements:

```bash
python3 ../../utils/validate_submission.py \
    --matching ../../output/matching_results.tsv \
    --candidate ../../output/candidate_pairs.tsv \
    --test-dir ../../dataset/test
```
