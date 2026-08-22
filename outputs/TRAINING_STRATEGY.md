# Training Strategy (UPDATED — POST-INTEGRITY FIX)

## Dataset Selection and Strategy

Given the 4GB VRAM constraint and the tight hackathon timeline, we strictly
prioritize minimum viable data to unblock the pipeline without spending
hours waiting for large archives to download.

1. **VQA (Priority 1): 20 examples**
   - Supplied by `dmarsili/RSVQA-LR-2k` (validation split, 20 records).
   - *Why:* Core interaction mode for the VLM.

2. **Change-VQA (Priority 2): 60 examples**
   - Supplied by `ljx620/CDVQA` (train: 40, validation: 10, test: 10).
   - *Why:* Temporal comparison is a high-value remote sensing capability
     that base LLaVA completely lacks.

3. **Captioning / Domain Adaptation: 20 examples**
   - Supplied by `BIFOLD-BigEarthNetv2-0/BigEarthNet.txt` (text-only).
   - *Why:* BigEarthNet serves as "domain adaptation" to help the model
     learn satellite terminology. Since `BigEarthNet.txt` provides only
     text annotations, we pass them as text-only domain-adaptation records
     to the language model directly without images.

4. **Grounding: OMITTED (0 examples)**
   - *Why:* Grounding annotations exist in `xiang709/VRSBench`. However,
     the raw image archive `Images_val.zip` alone is 3.9 GB. We have
     disabled VRSBench in `config.yaml` to unblock the rest of the
     pipeline using the data that is feasible to retrieve.

## Actual Dataset Counts (Integrity-Verified)

| Split | Records |
|-------|---------|
| train | 80      |
| val   | 10      |
| test  | 10      |
| **Total** | **100** |

| Task       | Count |
|------------|-------|
| VQA        | 20    |
| Change-VQA | 60    |
| Caption    | 20    |
| Grounding  | 0     |

| Source      | Count |
|-------------|-------|
| rsvqa       | 20    |
| cdvqa       | 60    |
| bigearthnet | 20    |

## Integrity Guarantee

- `prepare_data.py` rejects any record whose referenced images do not
  physically exist. Rejected records are logged to
  `outputs/dataset_integrity_skipped.jsonl`.
- `verify_data.py` independently scans every JSONL record and exits
  with code 1 if any image reference is broken.
- `tests/test_dataset.py::test_unified_dataset_quality_gates` checks
  every image path in every record.

## Constraints

- **Bandwidth/Time:** We downloaded aggressive subsets (20-60 examples
  per task) via HuggingFace APIs.
- **Current State:** RSVQA, CDVQA, BigEarthNet are physically downloaded
  locally to `data/external/`. The unified JSONL pipeline is generated,
  schema-validated, leak-checked, and integrity-verified.
- **Status:** DATA PIPELINE UNBLOCKED. DATASET INTEGRITY VERIFIED.
