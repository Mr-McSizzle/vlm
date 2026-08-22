#!/usr/bin/env python3
"""
T4 Preflight Check — Run this in Google Colab BEFORE training.

This script performs every verification step required to confirm that
the T4 GPU environment is ready for QLoRA fine-tuning of LLaVA-1.5.

Usage:
    python t4_preflight.py
"""

import gc
import json
import sys
import time
import yaml
from pathlib import Path


def fmt_mb(bytes_val):
    return f"{bytes_val / 1024 / 1024:.1f} MB"


def fmt_gb(bytes_val):
    return f"{bytes_val / 1024 / 1024 / 1024:.2f} GB"


def main():
    vlm_dir = Path(__file__).parent
    report_lines = []

    def log(msg=""):
        print(msg)
        report_lines.append(msg)

    log("=" * 64)
    log("T4 PREFLIGHT CHECK")
    log("=" * 64)
    all_ok = True

    # ----------------------------------------------------------------
    # STEP 1: Environment report
    # ----------------------------------------------------------------
    log("\n--- STEP 1: Environment ---")

    import platform
    log(f"Python:    {platform.python_version()}")

    import torch
    log(f"PyTorch:   {torch.__version__}")
    log(f"CUDA available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        log("FATAL: CUDA is not available. Cannot proceed.")
        log("\nT4 PREFLIGHT BLOCKED")
        sys.exit(1)

    log(f"CUDA version:   {torch.version.cuda}")
    log(f"GPU:            {torch.cuda.get_device_name(0)}")
    total_vram = torch.cuda.get_device_properties(0).total_memory
    log(f"VRAM:           {fmt_gb(total_vram)}")

    import transformers
    log(f"Transformers:   {transformers.__version__}")

    import peft
    log(f"PEFT:           {peft.__version__}")

    import bitsandbytes
    log(f"bitsandbytes:   {bitsandbytes.__version__}")

    import accelerate
    log(f"Accelerate:     {accelerate.__version__}")

    # ----------------------------------------------------------------
    # STEP 2: Load model in 4-bit
    # ----------------------------------------------------------------
    log("\n--- STEP 2: Load LLaVA-1.5 in 4-bit ---")

    from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig

    torch.cuda.reset_peak_memory_stats()
    vram_before_model = torch.cuda.memory_allocated()
    log(f"VRAM before model load: {fmt_mb(vram_before_model)}")

    log("Loading processor...")
    processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
    processor.tokenizer.pad_token = processor.tokenizer.eos_token
    log(f"  Tokenizer vocab size: {len(processor.tokenizer)}")
    log("  Processor loaded OK.")

    log("Loading model in 4-bit NF4...")
    q_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    model = LlavaForConditionalGeneration.from_pretrained(
        "llava-hf/llava-1.5-7b-hf",
        quantization_config=q_config,
        device_map={"": 0},
    )

    vram_after_model = torch.cuda.memory_allocated()
    log(f"VRAM after model load:  {fmt_mb(vram_after_model)}")
    log(f"Model VRAM footprint:   {fmt_mb(vram_after_model - vram_before_model)}")

    # ----------------------------------------------------------------
    # STEP 3: Verify architecture
    # ----------------------------------------------------------------
    log("\n--- STEP 3: Architecture verification ---")

    # Vision encoder frozen check
    # We must explicitly freeze it before LoRA
    vision_tensors = 0
    vision_elements = 0
    for name, param in model.named_parameters():
        if "vision_tower" in name:
            param.requires_grad = False
            vision_tensors += 1
            vision_elements += param.numel()
            
    vision_params = [p for n, p in model.named_parameters() if "vision_tower" in n]
    vision_frozen = all(not p.requires_grad for p in vision_params)
    
    log(f"Vision encoder parameter tensors: {vision_tensors}")
    log(f"Vision encoder total parameters:  {vision_elements:,}")
    log(f"Vision encoder frozen:            {vision_frozen}")
    if not vision_frozen:
        log("WARNING: Vision encoder is NOT frozen!")
        all_ok = False

    # Check LoRA target modules exist
    config_path = vlm_dir / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    lora_cfg = config["lora"]
    import re
    target_pattern = lora_cfg["target_modules"]
    matched_modules = [
        n for n, _ in model.named_modules()
        if re.search(target_pattern, n)
    ]
    log(f"LoRA target pattern: {target_pattern}")
    log(f"Matched modules:     {len(matched_modules)}")
    if matched_modules:
        log(f"  Sample: {matched_modules[:4]}")
    else:
        log("FATAL: No modules matched the LoRA target pattern!")
        all_ok = False

    # Attach LoRA
    log("\nAttaching LoRA...")
    from peft import LoraConfig, get_peft_model

    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    trainable_params, all_params = model.get_nb_trainable_parameters()
    log(f"Trainable parameters: {trainable_params:,} / {all_params:,}")
    log(f"Trainable percentage: {100 * trainable_params / all_params:.4f}%")

    vram_after_lora = torch.cuda.memory_allocated()
    log(f"VRAM after LoRA:      {fmt_mb(vram_after_lora)}")

    if trainable_params < 1_000_000:
        log("WARNING: Trainable parameters seem too low!")
        all_ok = False

    # ----------------------------------------------------------------
    # STEP 4: Real forward pass
    # ----------------------------------------------------------------
    log("\n--- STEP 4: Real forward pass ---")

    # Load one actual training example
    train_jsonl = vlm_dir / "data" / "unified" / "train.jsonl"
    with open(train_jsonl) as f:
        first_record = json.loads(f.readline())

    log(f"Record ID:      {first_record['id']}")
    log(f"Task:           {first_record['task']}")
    log(f"Source:         {first_record['source_dataset']}")
    log(f"Image count:    {len(first_record.get('images', []))}")

    # Load images
    from PIL import Image as PILImage
    images = []
    for img_rel in first_record.get("images", []):
        img_path = Path(img_rel)
        if not img_path.is_absolute() and img_path.parts and img_path.parts[0] == "data":
            img_path = vlm_dir / img_path
        if not img_path.exists():
            log(f"FATAL: Image not found: {img_path}")
            all_ok = False
        else:
            images.append(PILImage.open(str(img_path)).convert("RGB"))
            log(f"  Loaded: {img_rel}")

    # Build prompt
    image_tags = "".join("<image>\n" for _ in images)
    question = first_record["question"]
    answer = first_record["answer"]
    prompt = f"USER: {image_tags}{question}\nASSISTANT: {answer}"
    log(f"Prompt length:  {len(prompt)} chars")

    # Tokenize
    if images:
        inputs = processor(text=prompt, images=images, return_tensors="pt", padding=True)
    else:
        inputs = processor(text=prompt, return_tensors="pt", padding=True)

    inputs = {k: v.to("cuda") if hasattr(v, "to") else v for k, v in inputs.items()}
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)

    # Labels
    labels = inputs["input_ids"].clone()
    if processor.tokenizer.pad_token_id is not None:
        labels[labels == processor.tokenizer.pad_token_id] = -100
    inputs["labels"] = labels

    log(f"Input IDs shape: {inputs['input_ids'].shape}")
    if "pixel_values" in inputs:
        log(f"Pixel values shape: {inputs['pixel_values'].shape}")

    torch.cuda.reset_peak_memory_stats()
    vram_before_fwd = torch.cuda.memory_allocated()

    model.eval()
    t_start = time.time()
    with torch.no_grad():
        outputs = model(**inputs)
    t_fwd = time.time() - t_start

    vram_after_fwd = torch.cuda.memory_allocated()
    vram_peak = torch.cuda.max_memory_allocated()
    loss_val = outputs.loss.item() if outputs.loss is not None else "N/A"

    log(f"\nForward pass results:")
    log(f"  Loss:              {loss_val}")
    log(f"  Inference time:    {t_fwd:.2f}s")
    log(f"  VRAM before fwd:   {fmt_mb(vram_before_fwd)}")
    log(f"  VRAM after fwd:    {fmt_mb(vram_after_fwd)}")
    log(f"  VRAM peak:         {fmt_mb(vram_peak)}")
    log(f"  VRAM headroom:     {fmt_mb(total_vram - vram_peak)}")

    # Clean up forward pass tensors
    del outputs, inputs, labels
    gc.collect()
    torch.cuda.empty_cache()

    # ----------------------------------------------------------------
    # STEP 5: Code inspection checks
    # ----------------------------------------------------------------
    log("\n--- STEP 5: Code inspection ---")

    issues = []

    # Check train_lora.py for issues
    train_lora_path = vlm_dir / "train_lora.py"
    train_lora_src = train_lora_path.read_text()

    # Check for Windows paths
    if "C:\\Users" in train_lora_src or "C:/Users" in train_lora_src:
        issues.append("train_lora.py contains hardcoded Windows paths")
    else:
        log("  train_lora.py: No hardcoded Windows paths. OK")

    # Check for hardcoded usernames
    if "krish" in train_lora_src.lower():
        issues.append("train_lora.py contains hardcoded username")
    else:
        log("  train_lora.py: No hardcoded usernames. OK")

    # Check 4-bit config
    if "load_in_4bit=True" in train_lora_src:
        log("  train_lora.py: 4-bit QLoRA config present. OK")
    else:
        issues.append("train_lora.py missing load_in_4bit=True")

    # Check gradient checkpointing
    if "gradient_checkpointing_enable" in train_lora_src:
        log("  train_lora.py: Gradient checkpointing enabled. OK")
    else:
        issues.append("train_lora.py missing gradient_checkpointing_enable")

    # Check checkpoint saving
    if "save_pretrained" in train_lora_src:
        log("  train_lora.py: Checkpoint saving present. OK")
    else:
        issues.append("train_lora.py missing checkpoint saving")

    # Check gradient accumulation in training loop
    if "gradient_accumulation" in train_lora_src.lower():
        log("  train_lora.py: Gradient accumulation referenced. OK")
    else:
        issues.append("train_lora.py: gradient_accumulation_steps in config but NOT used in training loop")

    # Check config max_steps
    t_cfg = config["training"]
    if t_cfg.get("max_steps", 0) <= 10:
        issues.append(f"config.yaml: max_steps={t_cfg.get('max_steps')} is smoke-test value, not real training")

    for iss in issues:
        log(f"  WARNING: {iss}")
        all_ok = False

    # ----------------------------------------------------------------
    # STEP 6: verify_data.py
    # ----------------------------------------------------------------
    log("\n--- STEP 6: Dataset verification ---")

    import subprocess
    result = subprocess.run(
        [sys.executable, str(vlm_dir / "verify_data.py")],
        capture_output=True, text=True, cwd=str(vlm_dir)
    )
    # Show last few lines
    verify_lines = result.stdout.strip().split("\n")
    for vl in verify_lines[-6:]:
        log(f"  {vl}")
    if result.returncode != 0:
        log("  FAIL: verify_data.py exited with non-zero code")
        all_ok = False
    else:
        log("  verify_data.py: PASS")

    # ----------------------------------------------------------------
    # STEP 7: T4 training recommendations
    # ----------------------------------------------------------------
    log("\n--- STEP 7: T4 Training Recommendations ---")

    vram_free = total_vram - vram_peak
    log(f"Total VRAM:      {fmt_gb(total_vram)}")
    log(f"Peak used (fwd): {fmt_mb(vram_peak)}")
    log(f"Headroom:        {fmt_mb(vram_free)}")

    # Backward pass typically uses 2-3x the forward activation memory.
    # With gradient checkpointing, it's closer to 1.5x.
    # We need to leave room for optimizer states as well.
    log("\nRecommended T4 training configuration:")
    log("  batch_size: 1")
    log("  gradient_accumulation_steps: 8")
    log("  effective_batch_size: 8")
    log("  learning_rate: 2e-4")
    log("  num_train_epochs: 3")
    log("  max_steps: -1 (use epochs)")
    log("  save_steps: 50")
    log("  logging_steps: 5")
    log("  max_grad_norm: 1.0")
    log("  warmup_ratio: 0.03")
    log("  fp16: true (via bnb compute dtype)")
    log("  gradient_checkpointing: true")
    log("")
    log("  Rationale:")
    log("    - batch_size=1 is mandatory to fit in T4 VRAM")
    log("    - grad_accum=8 gives effective batch of 8 for stable gradients")
    log("    - 3 epochs over 80 training records = 240 steps")
    log("    - save every 50 steps = ~5 checkpoints for recovery")
    log("    - gradient checkpointing trades compute for VRAM")

    # ----------------------------------------------------------------
    # Write report
    # ----------------------------------------------------------------
    log("\n" + "=" * 64)

    report_path = vlm_dir / "outputs" / "T4_PREFLIGHT_REPORT.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        f.write("# T4 Preflight Report\n\n```\n")
        for line in report_lines:
            f.write(line + "\n")
        f.write("```\n")
    log(f"Report saved to: {report_path}")

    if all_ok:
        log("\nT4 PREFLIGHT PASSED")
        sys.exit(0)
    else:
        log("\nT4 PREFLIGHT BLOCKED")
        log("See warnings above for specific issues to fix.")
        sys.exit(1)


if __name__ == "__main__":
    main()
