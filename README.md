# SatQuery AI VLM Module

## BASELINE EVALUATION

To run the baseline evaluation with the untouched base model (`llava-hf/llava-1.5-7b-hf`):

```bash
# Single image baseline
python vlm/evaluate_baseline.py --image <single-image>

# Change VQA baseline
python vlm/evaluate_baseline.py --image <before-image> --image2 <after-image>
```

**Note:**
- This evaluates the **BASE MODEL** only.
- No LoRA adapter is loaded during this run.
- These outputs become the BEFORE comparison point.
- After LoRA training is complete, the exact same evaluation questions must be rerun to measure the impact of the fine-tuning.
