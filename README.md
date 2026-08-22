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

## P3 VLM Public Interface

The VLM is now cleanly encapsulated for use by the wider SatQuery system (like the P3 controller).
You do not need to deal with Hugging Face Transformers, LLaVA processors, PEFT adapters, CUDA, or 4-bit quantization config.

Simply import and call `vlm_answer`:

```python
from vlm.inference import vlm_answer

# Single-image tasks
response = vlm_answer(
    images="path/to/image.jpg", 
    question="What is the number of buildings?",
    task="vqa"
)
print(response["answer"])

# Two-image tasks (Change-VQA)
response = vlm_answer(
    images=["path/to/before.jpg", "path/to/after.jpg"], 
    question="What changed?",
    task="change_vqa"
)
```

The model loader is automatically cached in memory on the first call, so subsequent calls are very fast.
See `contracts/vlm_contract.md` for full schema details!
