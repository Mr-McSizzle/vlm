# Part 5 Initial State

## What already works
- Base LLaVA-1.5 7B inference works in 4-bit mode on the RTX 3050 GPU.
- Baseline evaluation pipeline (Part 4) is fully functional, saving results and summaries.

## What files already exist
- `/vlm/infer.py`, `/vlm/inference.py`, `/vlm/schemas.py`, `/vlm/prompts.py`
- `/vlm/test_llava.py`, `/vlm/dataset.py`, `/vlm/prepare_data.py`, `/vlm/train_lora.py`, `/vlm/evaluate.py`, `/vlm/evaluate_baseline.py`
- Tests in `/vlm/tests/`

## What Part 4 baseline artifacts exist
- `/vlm/data/baseline_questions.json`
- `/vlm/data/baseline_manifest.json`
- `/vlm/data/baseline_results.json`
- `/vlm/data/baseline_summary.json`

## What is missing for training
- **The actual source datasets.** (BigEarthNet, VRSBench, RSVQA, CDVQA were not found in the repository).
- Actual dataset annotations for parsing.

## What files we will extend
- `/vlm/prepare_data.py` (to define the parsing architecture)
- `/vlm/dataset.py` (to handle unified format)
- `/vlm/config.yaml` (to define strategy)
- `/vlm/tests/test_dataset.py`

## What files we will avoid touching
- `baseline_*.json` outputs from Part 4.
- Core inference code.
- Model weights.
