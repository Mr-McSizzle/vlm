# SatQuery AI — VLM Module

**Vision-language reasoning module for satellite imagery, change analysis, and structured geospatial question answering.**

This repository contains the multimodal model layer used by the wider SatQuery AI system. It wraps the underlying model stack behind a stable Python interface so controller code does not need to manage Transformers, PEFT adapters, CUDA configuration, or model-loading details directly.

## Public interface

The primary entry point is `vlm_answer`:

```python
from vlm.inference import vlm_answer

response = vlm_answer(
    images="path/to/image.jpg",
    question="What is visible in this region?",
    task="vqa",
)

print(response["answer"])
```

For paired-image change analysis:

```python
response = vlm_answer(
    images=["before.jpg", "after.jpg"],
    question="What changed?",
    task="change_vqa",
)
```

## Model workflow

```text
image(s)
   ↓
task + question
   ↓
SatQuery VLM interface
   ↓
base / adapted multimodal model
   ↓
structured answer + metadata
```

## Baseline vs adapted evaluation

The repository separates evaluation of the untouched base model from evaluation of an adapted checkpoint so improvements can be measured against the same question set.

Baseline example:

```bash
python vlm/evaluate_baseline.py --image <single-image>
```

Paired-image baseline:

```bash
python vlm/evaluate_baseline.py --image <before-image> --image2 <after-image>
```

Authoritative evaluation scripts in the repository support base / adapted comparisons and report generation.

## External perception evidence

`vlm_answer` can receive structured evidence from another perception component, for example:

```python
evidence = {
    "type": "change_detection",
    "changed_fraction": 0.21,
    "dominant_region": "south",
}
```

This lets the multimodal layer reason with explicit upstream evidence rather than forcing every observation to come from free-form visual inference.

## Optional fallback path

The module contains an optional Gemini multimodal fallback that can be enabled through environment configuration. The fallback is designed to preserve the same output contract and record provenance metadata such as whether fallback was used and why.

API keys should never be committed to the repository.

## Technology represented

- LLaVA / Hugging Face Transformers
- PEFT / LoRA adaptation
- PyTorch
- 4-bit / quantized model-loading paths
- Structured inference contracts
- VQA and change-VQA
- Optional multimodal fallback

## Research boundary

This is a research module, not a validated geospatial intelligence product. Stronger evaluation should include fixed benchmark datasets, geospatially separated holdouts, task-specific metrics, robustness testing, and explicit failure analysis.

## Role inside SatQuery

The wider SatQuery system separates perception, multimodal reasoning, controller orchestration, backend services, and the frontend. This repository is specifically the **VLM reasoning layer**.