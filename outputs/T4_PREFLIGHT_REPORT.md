# T4 Preflight Report

```
================================================================
T4 PREFLIGHT CHECK
================================================================

--- STEP 1: Environment ---
Python:    3.12.4
PyTorch:   2.12.0+cu130
CUDA available: True
CUDA version:   13.0
GPU:            NVIDIA GeForce RTX 3050 Laptop GPU
VRAM:           4.00 GB
Transformers:   4.56.2
PEFT:           0.20.0
bitsandbytes:   0.50.1
Accelerate:     1.14.0

--- STEP 5: Code inspection ---
  train_lora.py: No hardcoded Windows paths. OK
  train_lora.py: No hardcoded usernames. OK
  train_lora.py: 4-bit QLoRA config present. OK
  train_lora.py: Gradient checkpointing enabled. OK
  train_lora.py: Checkpoint saving present. OK
  train_lora.py: Gradient accumulation referenced. OK

--- STEP 6: Dataset verification ---
    cdvqa = 60
    rsvqa = 20
  
  Preview written to: C:\Users\krish\vlm\outputs\unified_dataset_preview.md
  
  [OK] DATASET INTEGRITY VERIFIED - zero missing image references.
  verify_data.py: PASS

--- STEP 7: T4 Training Recommendations ---
Total VRAM:      4.00 GB
Peak used (fwd): 0.0 MB
Headroom:        4095.5 MB

Recommended T4 training configuration:
  batch_size: 1
  gradient_accumulation_steps: 8
  effective_batch_size: 8
  learning_rate: 2e-4
  num_train_epochs: 3
  max_steps: -1 (use epochs)
  save_steps: 50
  logging_steps: 5
  max_grad_norm: 1.0
  warmup_ratio: 0.03
  fp16: true (via bnb compute dtype)
  gradient_checkpointing: true

  Rationale:
    - batch_size=1 is mandatory to fit in T4 VRAM
    - grad_accum=8 gives effective batch of 8 for stable gradients
    - 3 epochs over 80 training records = 240 steps
    - save every 50 steps = ~5 checkpoints for recovery
    - gradient checkpointing trades compute for VRAM

================================================================
```
