# Final VLM Quality Audit

## 1. Evaluation Validity (Task 1 & 5)
**Status: INVALID (SEVERE BUGS FOUND)**

The final evaluation pipeline (`evaluate_final.py`) is fundamentally broken and produces completely invalid results:
1. **Static Test Image**: It completely ignores the actual test set (`data/unified/test.jsonl`). Instead, it loads a single hard-coded image (`test.jpg`) for all tests.
2. **Change-VQA Duplication Bug**: For the two-image Change-VQA test, it constructs `images=[img, img]` using the *exact same identical image object twice*. The model correctly outputs that "the images are identical" because they literally are.
3. **Dummy Metric**: The standard `evaluate.py` script relies on a dummy `SatQueryVLM` class in `inference.py` that just returns `"Dummy response to: {prompt}"`. 

## 2. Checkpoint Validity (Task 2)
**Status: INVALID (LOCAL SMOKE TEST LOADED)**

Inspecting `checkpoints/final_adapter/` reveals the presence of a file named `LOCAL_SMOKE_TEST_ONLY.txt`. This proves that the T4-trained checkpoint was not properly loaded or synced into the environment; the evaluation was executed against the broken 4-step local smoke-test adapter from Phase 4, resulting in severe hallucinations.

## 3. Answer Truncation (Task 4)
**Bug Identified**: The outputs appear artificially cut off because `evaluate_final.py` hard-codes `max_new_tokens=50` during the `.generate()` call, forcefully halting the model's generation mid-sentence regardless of the EOS token.

## 4. Training Data Integrity (Task 6 & 7)
**Status: CORRUPT (CAPTION DATA BROKEN)**

Inspection of `data/unified/train.jsonl` reveals severe data formatting bugs:
- **BigEarthNet (Caption) is missing images**: The records for BigEarthNet have `"images": []`.
- **BigEarthNet targets are garbage**: The answers are bounding-box coordinates (e.g., `[0.0 0.33, 0.28 0.8]`) instead of valid semantic captions, suggesting a critical flaw in the data acquisition pipeline for this subset. 

Example training prompt generated:
`USER: Describe this satellite image.\nASSISTANT: [0.0 0.33, 0.28 0.8]`

## 5. Model Quality Assessment (Task 8)
**Classification**: **D. EVALUATION INVALID — evaluation pipeline must be fixed first**

The outputs observed (e.g., Change-VQA failing, hallucinations) are direct symptoms of feeding the exact same image twice to a locally generated smoke-test adapter, combined with truncation bugs and corrupted captioning records.

## 6. Retraining Recommendation
See `outputs/RETRAINING_RECOMMENDATION.md`. We must fix the evaluation script and dataset parsing bugs before ANY further training is conducted.

## 7. Raw Control Test (Task 3)
A mock control test reading real records from `test.jsonl` was executed. The raw outputs are preserved in `outputs/QUALITY_AUDIT_RAW.json`.
