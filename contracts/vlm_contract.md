# SatQuery VLM Public Interface Contract (P3 Contract)

This document defines the stable, reusable Python interface exposed by the VLM component for use by the P3 controller. 
The P3 controller does NOT need to understand LLaVA, Transformers, PEFT, tokenization, or model checkpoint logic.

## Function Name
`vlm_answer`

## Import Path
```python
from inference import vlm_answer
```

## Arguments

| Argument | Type | Required | Description |
|---|---|---|---|
| `images` | `str` or `List[str]` | Yes | A single image path (string) or a list of image paths (strings). |
| `question` | `str` | Yes | The natural language question or prompt to ask the model. |
| `evidence` | `dict` or `str` or `None` | No | Optional supporting evidence (e.g. from CD or retrieval). Must be natively JSON-serializable if dict. |
| `task` | `str` or `None` | No | Explicitly declares the task format. |

## Allowed Image Counts & Task Values

- **`vqa`**: Requires exactly **1** image. Standard visual question answering.
- **`change_vqa`**: Requires exactly **2** images (`images[0]` = earlier, `images[1]` = later). 
- **`caption`**: Requires exactly **1** image. Detailed image description.
- **`grounding`**: Requires exactly **1** image.

*Note: If `task` is `None`, the VLM will infer the task based on the image count (2 images -> change_vqa) and question keywords.*

## Evidence Schema (Optional)
Evidence is completely optional. If provided as a `dict`, it should be flat or simply nested JSON-serializable data. Example:
```json
{
    "type": "change_detection",
    "changed_fraction": 0.21,
    "dominant_region": "south"
}
```

## Output Schema
Returns ONLY JSON-serializable Python dictionaries. Never returns tensors, PIL objects, or raw model objects.

### Success Output Example
```json
{
    "answer": "The total number of buildings is 12.",
    "regions": [],
    "model": "satquery-vlm",
    "checkpoint": "checkpoints/final_adapter",
    "metadata": {
        "task": "vqa",
        "latency_seconds": 2.45,
        "status": "ok"
    }
}
```

### Error Output Example
```json
{
    "answer": "",
    "regions": [],
    "model": "satquery-vlm",
    "checkpoint": "unknown",
    "metadata": {
        "status": "error",
        "error_type": "ValueError",
        "error_message": "change_vqa requires exactly 2 images. Got 1."
    }
}
```

### Grounding Regions
If the model outputs a parseable `REGION: [x1, y1, x2, y2]`, it will be stripped from the `answer` string and added as a list of floats `[0.1, 0.2, 0.8, 0.9]` to `regions`. If parsing fails, `regions` remains empty and `metadata["parse_warning"]` is added.

## Examples

### 1. Single Image VQA
```python
result = vlm_answer(
    images="data/sample_1.jpg", 
    question="Are there any airplanes visible?",
    task="vqa"
)
print(result["answer"])
```

### 2. Temporal Change-VQA
```python
result = vlm_answer(
    images=["data/before.jpg", "data/after.jpg"], 
    question="Did the forested area decrease?",
    task="change_vqa"
)
print(result["answer"])
```
