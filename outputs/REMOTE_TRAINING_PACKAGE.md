# Remote Training Package

This document contains everything needed for a teammate to deploy and run the SatQuery VLM fine-tuning on a remote Linux GPU machine.

## Hardware Requirements
- **OS:** Ubuntu 22.04 LTS (or similar Linux distro)
- **GPU VRAM:** Minimum 16GB (e.g., RTX 4090, A10G). 24GB recommended.
- **CUDA:** 11.8 or 12.1+

## Software Requirements
```bash
pip install -r vlm/requirements.txt
```
*(Dependencies: torch, torchvision, transformers, datasets, peft, accelerate, bitsandbytes, pyyaml, pillow)*

## Base Model
- `llava-hf/llava-1.5-7b-hf` (Loaded in 4-bit NormalFloat using bitsandbytes)

## LoRA / QLoRA Configuration
- **r:** 8
- **alpha:** 16
- **dropout:** 0.05
- **target_modules:** `.*language_model.*\.(q_proj|k_proj|v_proj|o_proj)$`
*(The Vision Encoder remains 100% frozen.)*

## Dataset Manifest
The remote machine must have the following datasets placed relative to the working directory:
- `vlm/data/unified/train.jsonl`
- `vlm/data/unified/val.jsonl`
- `vlm/data/unified/test.jsonl`
- `vlm/data/external/rsvqa/images/`
- `vlm/data/external/cdvqa/images/`

## 1. Preflight Check
Run this command to verify the remote system satisfies all requirements before training:
```bash
python -m vlm.train_lora --preflight
```

## 2. Actual Training Command
To start the full QLoRA fine-tuning run:
```bash
python -m vlm.train_lora
```
*Note: Ensure you are in the project root directory when running this command so relative paths resolve correctly.*

## 3. Checkpoint Output Directory
The training script will output checkpoints to:
- `vlm/checkpoints/final_adapter`

## 4. Post-Training Evaluation Command
Once training completes, verify the new adapter against the evaluation suite:
```bash
python -m vlm.evaluate_final
```
This script will load the newly produced checkpoint, verify the adapter, run single-image VQA and two-image Change-VQA, and compare the outputs against the Part 4 baseline.
