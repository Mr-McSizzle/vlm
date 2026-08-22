# T4 Preflight Report

```text
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

--- STEP 2: Load LLaVA-1.5 in 4-bit ---
VRAM before model load: 0.0 MB
Loading processor...
  Tokenizer vocab size: 32002
  Processor loaded OK.
Loading model in 4-bit NF4...

ERROR: OutOfMemoryError / Resource Exhausted
Cannot load LLaVA-1.5 7B model. The current local GPU only has 4.00 GB of VRAM.

--- STEP 5: Code inspection ---
  train_lora.py: No hardcoded Windows paths. OK
  train_lora.py: No hardcoded usernames. OK
  train_lora.py: 4-bit QLoRA config present. OK
  train_lora.py: Gradient checkpointing enabled. OK
  train_lora.py: Checkpoint saving present. OK
  train_lora.py: Gradient accumulation referenced. OK
  train_lora.py: Resume from checkpoint support added. OK
  config.yaml: max_steps removed (using num_train_epochs). OK

--- STEP 6: Dataset verification ---
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

--- STEP 7: T4 Training Recommendations ---
The following configuration has been written to config.yaml for T4:
  batch_size: 1
  gradient_accumulation_steps: 8
  effective_batch_size: 8
  learning_rate: 2e-4
  num_train_epochs: 3
  save_steps: 50
  logging_steps: 5

T4 PREFLIGHT BLOCKED
```

**Reason for block:**
The local execution environment is equipped with an NVIDIA GeForce RTX 3050 Laptop GPU (4.00 GB VRAM), not a Tesla T4 (15 GB VRAM). The LLaVA-1.5 7B model cannot be loaded into 4GB of VRAM to perform the real forward-pass measurements requested in Steps 2-4. 

To execute the preflight checks on the T4, run `python t4_preflight.py` inside the Colab environment.
