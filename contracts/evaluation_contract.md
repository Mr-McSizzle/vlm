# SatQuery Evaluation Contract

This document outlines the strict requirements for the authoritative evaluation pipeline. To prevent historical issues with fabricated or "mocked" evaluation data, these principles are hard-coded into the pipeline.

## Core Rules
1. **No Mocking by Default**: Evaluation must use real inference from the specified model endpoint.
2. **Deterministic Inputs**: The pipeline always reads precisely from `data/unified/test.jsonl`. No hardcoded text or `test.jpg` files are permitted unless specified in the test records.
3. **Image Order Preservation**: Temporal inputs (like `change_vqa`) depend on `images[0]` and `images[1]` matching exactly the before/after sequence.
4. **Task-Aware Scoring**:
    - **Yes/No**: Case-insensitive exact match.
    - **Numeric**: Integer extraction from response matching integer from target.
    - **Categorical**: Exact match on normalized strings.
    - **Free-form**: Tagged as `manual_review` if it cannot be deterministically scored.
5. **No Invented Metrics**: Unscorable records are tracked independently from accuracy metrics.

## Evaluation Workflow

### 1. Evaluate Base Model
```bash
python evaluate_authoritative.py \
    --model base \
    --output outputs/evaluation_base.json
```

### 2. Evaluate Adapted Model
```bash
python evaluate_authoritative.py \
    --model adapted \
    --checkpoint checkpoints/final_adapter \
    --output outputs/evaluation_adapted.json
```

### 3. Compare & Summarize
```bash
python compare_models.py \
    --base outputs/evaluation_base.json \
    --adapted outputs/evaluation_adapted.json
```

## Outputs
- `EVALUATION_SUMMARY.md`: Overview of successes, failures, and accuracy matrices.
- `FAILURE_ANALYSIS.md`: Auto-categorized breakdown of inference deviations (e.g., *Hallucination*, *Counting Error*, *Temporal Reasoning*).
- `EVALUATION_MANIFEST.json`: Cryptographic and environment fingerprint for reproducibility.

## Hardware Requirements
This evaluation runs standard 7B model inference across the entire dataset. It is **not** supported on <5GB VRAM cards (e.g. RTX 3050). Run in the target **Colab Tesla T4** (15GB VRAM) environment.
