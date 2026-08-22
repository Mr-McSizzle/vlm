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
| `evidence` | `dict` or `None` | No | Optional supporting evidence from P2 (Prithvi). Must adhere to schema. |
| `task` | `str` or `None` | No | Explicitly declares the task format. |

## Evidence Schema (Optional)
Evidence must be a JSON-serializable dictionary with one of the following schemas:

### Change Detection
```json
{
    "type": "change_detection",
    "changed_fraction": 0.21,
    "dominant_region": "south",
    "mask_path": "/optional/path.png"
}
```

### Segmentation
```json
{
    "type": "segmentation",
    "classes": ["water", "building"],
    "regions": ["north", "center"],
    "mask_path": "/optional/path.png"
}
```

### Fusion
```json
{
    "type": "fusion",
    "optical_summary": "clear vegetation and water boundaries",
    "sar_summary": "strong radar response over built-up areas",
    "agreement": 0.87,
    "confidence": 0.95
}
```

## Output Schema
Returns ONLY JSON-serializable Python dictionaries. Never returns tensors, PIL objects, or raw model objects.

```json
{
    "answer": "string",
    "task": "string",
    "regions": [],
    "confidence": null,
    "model": "satquery-vlm",
    "checkpoint": "checkpoints/final_adapter",
    "metadata": {
        "status": "ok",
        "latency_seconds": 0.0,
        "truncated": false,
        "raw_output_available": true
    }
}
```

### 1. VQA Example
```json
{
    "answer": "The number of buildings is 12.",
    "task": "vqa",
    "regions": [],
    "confidence": null,
    "model": "satquery-vlm",
    "checkpoint": "checkpoints/final_adapter",
    "metadata": {
        "status": "ok",
        "latency_seconds": 2.45,
        "truncated": false,
        "raw_output_available": true
    }
}
```

### 2. Change-VQA Example
```json
{
    "answer": "The detected change affects approximately 21% of the scene...",
    "task": "change_vqa",
    "regions": [],
    "confidence": null,
    "model": "satquery-vlm",
    "checkpoint": "checkpoints/final_adapter",
    "metadata": {
        "status": "ok",
        "latency_seconds": 3.12,
        "truncated": false,
        "raw_output_available": true,
        "evidence_used": true,
        "evidence_type": "change_detection",
        "evidence_source": "external_perception"
    }
}
```

### 3. Grounding Example
```json
{
    "answer": "A building.",
    "task": "grounding",
    "regions": [
        {"x1": 10.0, "y1": 20.5, "x2": 30.0, "y2": 40.0}
    ],
    "confidence": null,
    "model": "satquery-vlm",
    "checkpoint": "checkpoints/final_adapter",
    "metadata": {
        "status": "ok",
        "latency_seconds": 1.45,
        "truncated": false,
        "raw_output_available": true
    }
}
```

### 4. Parser Warning Example
```json
{
    "answer": "",
    "task": "grounding",
    "regions": [],
    "confidence": null,
    "model": "satquery-vlm",
    "checkpoint": "checkpoints/final_adapter",
    "metadata": {
        "status": "warning",
        "parse_warning": "Coordinates out of bounds (0-100) or invalid dimensions.",
        "latency_seconds": 1.45,
        "truncated": false,
        "raw_output_available": true
    }
}
```

### 5. Error Example
```json
{
    "answer": "",
    "task": "vqa",
    "regions": [],
    "confidence": null,
    "model": "satquery-vlm",
    "checkpoint": "unknown",
    "metadata": {
        "status": "error",
        "error_type": "ValueError",
        "error_message": "Malformed evidence object: ..."
    }
}
```
