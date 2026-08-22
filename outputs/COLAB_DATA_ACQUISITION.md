# Colab Data Acquisition Guide

## Problem

The Git repository correctly excludes large binary data via `.gitignore`:

```
data/external/
*.jpg
```

After cloning, the unified JSONL files (`data/unified/*.jsonl`) exist and
reference 140 image files, but those image files are not present in the
clone. This is expected and correct — images must be downloaded separately.

## Data Sources

### RSVQA (VQA task)
- **HuggingFace dataset:** `dmarsili/RSVQA-LR-2k`
- **Split:** `validation` (first 20 records)
- **Files produced:** 20 JPEG images
- **Approx size:** ~0.1 MB
- **Output directory:** `data/external/rsvqa/images/`
- **Filename pattern:** `validation_0.jpg` through `validation_19.jpg`

### CDVQA (Change-VQA task)
- **HuggingFace dataset:** `ljx620/CDVQA`
- **Splits and limits:**
  - `train`: first 40 records → 80 image files (pairs)
  - `validation`: first 10 records → 20 image files (pairs)
  - `test`: first 10 records → 20 image files (pairs)
- **Files produced:** 120 JPEG images (60 pairs)
- **Approx size:** ~5 MB
- **Output directory:** `data/external/cdvqa/images/`
- **Filename pattern:** `cdvqa-{split}-{idx:08d}_1.jpg` and `_2.jpg`

### BigEarthNet.txt (Caption/domain adaptation task)
- **HuggingFace dataset:** `BIFOLD-BigEarthNetv2-0/BigEarthNet.txt`
- **Split:** `all_data` (first 20 records)
- **Files produced:** 0 images (text-only annotations)
- **Approx size:** ~10 KB
- **Output directory:** `data/external/bigearthnet_txt/`

## Total Download

| Item | Files | Size |
|------|-------|------|
| RSVQA images | 20 | ~0.1 MB |
| CDVQA images | 120 | ~5 MB |
| BigEarthNet text | 0 images | ~10 KB |
| **Total** | **140 images** | **~5 MB** |

## Exact Commands for Colab

```bash
# 1. Clone the repository (if not already done)
git clone https://github.com/Mr-McSizzle/vlm.git
cd vlm

# 2. Install VLM dependencies
pip install -r requirements-vlm.txt

# 3. Download the exact dataset subsets
python acquire_data.py

# 4. Verify dataset integrity
python verify_data.py

# 5. Run quality-gate tests
pytest -q tests/test_dataset.py
```

### Step 3 details

`acquire_data.py` will:
- Download 20 RSVQA images from HuggingFace
- Download 120 CDVQA images from HuggingFace
- Download 20 BigEarthNet text records from HuggingFace
- Skip any files that already exist (resumable)
- Verify that all 140 image references in `data/unified/*.jsonl` are satisfied
- Exit with code 0 on success, code 1 on failure

Estimated time: **1-3 minutes** on Colab (5 MB download over HF streaming API).

### Resumability

The script is fully resumable. If interrupted, re-run `python acquire_data.py`
and it will skip already-downloaded files.

### Verify-only mode

To check whether images are present without downloading:

```bash
python acquire_data.py --verify
```

## Expected Directory Structure After Acquisition

```
vlm/
├── data/
│   ├── external/
│   │   ├── rsvqa/
│   │   │   ├── images/
│   │   │   │   ├── validation_0.jpg
│   │   │   │   ├── ...
│   │   │   │   └── validation_19.jpg    (20 files)
│   │   │   └── validation.json
│   │   ├── cdvqa/
│   │   │   ├── images/
│   │   │   │   ├── cdvqa-train-00000000_1.jpg
│   │   │   │   ├── cdvqa-train-00000000_2.jpg
│   │   │   │   ├── ...
│   │   │   │   └── cdvqa-val-00000009_2.jpg   (120 files)
│   │   │   ├── train.json
│   │   │   ├── validation.json
│   │   │   └── test.json
│   │   └── bigearthnet_txt/
│   │       └── all_data.json
│   └── unified/
│       ├── train.jsonl      (80 records)
│       ├── val.jsonl        (10 records)
│       ├── test.jsonl       (10 records)
│       ├── statistics.json
│       └── manifest.json
```

## Verification Command

```bash
python verify_data.py
```

Expected output:
```
TRAIN:
  records = 80
  valid image references = 116
  missing image references = 0

VAL:
  records = 10
  valid image references = 14
  missing image references = 0

TEST:
  records = 10
  valid image references = 10
  missing image references = 0

TOTAL:
  records = 100
  valid image references = 140
  missing image references = 0

[OK] DATASET INTEGRITY VERIFIED - zero missing image references.
```

## Invariant

After running `acquire_data.py`, the following invariant must hold:

```
data/unified/*.jsonl → every referenced image → physically exists
```

Do NOT proceed to model training until `verify_data.py` exits with code 0.
