# Dataset Integrity Audit

Generated: 2026-08-23

## Root Cause Analysis

The repeated `AssertionError: Missing image` failures were caused by a
combination of:

**A. Stale unified JSONL generated before current dataset files existed.**
The unified `train.jsonl` / `val.jsonl` / `test.jsonl` were generated from
a prior `download_subsets.py` run with different download limits. When
`download_subsets.py` was later re-run with different limits, the old
unified JSONL files were not regenerated — they still referenced filenames
from the previous download that no longer existed on disk.

**B. Fragile error handling in `prepare_data.py`.**
When a missing image was detected for RSVQA/CDVQA, the old code raised
`FileNotFoundError`, which was silently caught by a bare `except Exception`
and counted as `invalid`. No log was produced, and the statistics dict was
never populated with actual source/task counts.

**C. `verify_data.py` did not check image existence at all.**
The old `verify_data.py` only generated a preview markdown. It did not
scan image paths, did not exit with a non-zero code on failure, and
provided no integrity guarantee.

## Fix Applied

1. **`prepare_data.py` rewritten** with an explicit `_all_images_exist()`
   gate. Records are added to the unified dataset ONLY if every referenced
   image physically exists. Missing-image records are logged to
   `outputs/dataset_integrity_skipped.jsonl` with the exact missing paths.
   Statistics are now properly populated.

2. **`verify_data.py` rewritten** as a standalone integrity checker.
   It scans every record in every split, checks every image path, prints
   per-split and per-task statistics, and exits with code 1 if any image
   reference is broken.

3. **Unified dataset regenerated** from the physical files currently in
   `data/external/`.

## Physical Data Inventory

### RSVQA (`data/external/rsvqa`)
- Physical image files: **20** (`validation_0.jpg` through `validation_19.jpg`)
- Annotation records: **20** (`validation.json`)
- Unified records referencing RSVQA: **20**
- Valid image references: **20**
- Missing image references: **0**

### CDVQA (`data/external/cdvqa`)
- Physical image files: **120** (pairs `_1.jpg`/`_2.jpg` for 60 records)
- Annotation records: **60** (`train.json`: 40, `validation.json`: 10, `test.json`: 10)
- Unified records referencing CDVQA: **60**
- Valid image references: **120** (2 per record)
- Missing image references: **0**

### BigEarthNet.txt (`data/external/bigearthnet_txt`)
- Physical image files: **0** (text-only dataset — no images expected)
- Annotation records: **20** (`all_data.json`)
- Unified records referencing BigEarthNet: **20**
- Image references: **0** (text-only caption task)

### VRSBench (`data/external/vrsbench`)
- Status: **disabled in config.yaml** (`enabled: false`)
- Not included in unified dataset

## Final Unified Dataset Counts

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

## Skipped Records

**0** records skipped due to missing images.

See `outputs/dataset_integrity_skipped.jsonl` for the skip log (empty).

## Verification Results

- `python verify_data.py`: **PASSED** (exit code 0, zero missing images)
- `pytest -q tests/test_dataset.py`: **3/3 PASSED**
- No fake/placeholder images were created
- No tests were modified or weakened
