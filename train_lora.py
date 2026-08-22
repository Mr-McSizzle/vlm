import os
import sys
import yaml
import torch
import time
import argparse
from pathlib import Path

def preflight_check():
    print("Running Preflight Checks...")
    
    # 1. CUDA
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available.")
        sys.exit(1)
    print(f"OK: CUDA is available ({torch.cuda.get_device_name(0)})")
    
    # 2. VRAM
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"INFO: Detected VRAM: {vram_gb:.2f} GB")
    if vram_gb < 14.0:
        print("WARNING: VRAM is less than 14GB. Full training might OOM or heavily swap.")
    else:
        print("OK: Sufficient VRAM detected.")
        
    # 3. Dependencies
    try:
        import transformers
        import peft
        import bitsandbytes
        import accelerate
        from PIL import Image
        print("OK: All core dependencies are installed.")
    except ImportError as e:
        print(f"ERROR: Missing dependency - {e}")
        sys.exit(1)
        
    # 4. Dataset Files
    vlm_dir = Path(__file__).parent
    required_files = [
        vlm_dir / "config.yaml",
        vlm_dir / "data/unified/train.jsonl",
        vlm_dir / "data/unified/val.jsonl",
        vlm_dir / "data/unified/test.jsonl"
    ]
    for f in required_files:
        if not f.exists():
            print(f"ERROR: Missing required file: {f}")
            sys.exit(1)
    print("OK: Dataset manifests and config found.")
    
    # 5. Model Access (HuggingFace)
    print("OK: Model access assumed (will download if not cached).")
    
    print("\nPreflight checks passed! You are ready to start training.")
    sys.exit(0)

def main():
    parser = argparse.ArgumentParser(description="SatQuery VLM LoRA Trainer")
    parser.add_argument("--preflight", action="store_true", help="Run system checks and exit")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to checkpoint directory to resume from")
    args = parser.parse_args()
    
    if args.preflight:
        preflight_check()
        
    from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, PeftModel
    from torch.utils.data import DataLoader
    from dataset import VLMDataset, DataCollator

    vlm_dir = Path(__file__).parent
    config_path = vlm_dir / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
        
    out_dir = vlm_dir / config["training"]["output_dir"]
    os.makedirs(out_dir, exist_ok=True)
    
    print("Loading processor...")
    processor = AutoProcessor.from_pretrained('llava-hf/llava-1.5-7b-hf')
    processor.tokenizer.pad_token = processor.tokenizer.eos_token
    
    print("Loading datasets...")
    train_jsonl = vlm_dir / "data/unified/train.jsonl"
    train_ds = VLMDataset(str(train_jsonl), processor)
    print(f"Dataset size: {len(train_ds)}")
    
    collator = DataCollator(processor)
    loader = DataLoader(train_ds, batch_size=config["training"]["batch_size"], shuffle=True, collate_fn=collator)
    
    print("Loading model in 4-bit...")
    q_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    model = LlavaForConditionalGeneration.from_pretrained(
        'llava-hf/llava-1.5-7b-hf',
        quantization_config=q_config,
        device_map={"": 0}
    )
    
    print("\n--- ATTACHING LORA ---")
    lora_cfg = config["lora"]
    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
        task_type="CAUSAL_LM"
    )
    
    if args.resume_from_checkpoint:
        print(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
        # When resuming PEFT, we load the adapter onto the base model
        model = PeftModel.from_pretrained(model, args.resume_from_checkpoint, is_trainable=True)
    else:
        model = get_peft_model(model, peft_config)
    
    # Enable gradient checkpointing to save VRAM
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    
    # EXPLICITLY FREEZE VISION TOWER
    vision_frozen_count = 0
    for name, param in model.named_parameters():
        if "vision_tower" in name:
            param.requires_grad = False
            vision_frozen_count += param.numel()
    print(f"Explicitly frozen {vision_frozen_count:,} parameters in the vision tower.")
    
    # Hard assertion: ensure no vision parameters are trainable
    vision_unfrozen = [n for n, p in model.named_parameters() if "vision_tower" in n and p.requires_grad]
    if len(vision_unfrozen) > 0:
        raise RuntimeError(f"Vision encoder is not fully frozen. Unfrozen parameters: {vision_unfrozen}")
    
    trainable_params, all_param = model.get_nb_trainable_parameters()
    print(f"Trainable: {trainable_params} / {all_param} ({100 * trainable_params / all_param:.4f}%)")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["training"]["learning_rate"]))
    
    if args.resume_from_checkpoint:
        opt_path = os.path.join(args.resume_from_checkpoint, "optimizer.pt")
        if os.path.exists(opt_path):
            optimizer.load_state_dict(torch.load(opt_path))
            print("Optimizer state restored.")
    
    print("\n--- TRAINING ---")
    model.train()
    start_time = time.time()
    
    # Training Loop configurations
    steps = config["training"].get("max_steps", -1)
    save_steps = config["training"].get("save_steps", 500)
    logging_steps = config["training"].get("logging_steps", 10)
    grad_accum_steps = config["training"].get("gradient_accumulation_steps", 1)
    
    # Total examples = epochs * loader length
    total_epochs = config["training"].get("num_train_epochs", 1)
    
    step = 0
    global_step = 0
    initial_loss = None
    accumulated_loss = 0.0
    
    for epoch in range(total_epochs):
        for i, batch in enumerate(loader):
            if steps > 0 and global_step >= steps: 
                break
            
            inputs = {k: v.to('cuda') if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            # Ensure float16 for images
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)
                
            outputs = model(**inputs)
            loss = outputs.loss / grad_accum_steps
            
            if initial_loss is None: 
                initial_loss = loss.item() * grad_accum_steps
                
            loss.backward()
            accumulated_loss += loss.item()
            
            if (i + 1) % grad_accum_steps == 0 or (i + 1) == len(loader):
                optimizer.step()
                optimizer.zero_grad()
                
                if (global_step + 1) % logging_steps == 0:
                    print(f"Epoch {epoch+1}/{total_epochs} | Global Step {global_step+1} | Loss: {accumulated_loss:.4f}")
                
                accumulated_loss = 0.0
                
                if (global_step + 1) % save_steps == 0:
                    ckpt_dir = os.path.join(out_dir, f"checkpoint-{global_step+1}")
                    print(f"Saving intermediate checkpoint to {ckpt_dir}...")
                    model.save_pretrained(ckpt_dir)
                    torch.save(optimizer.state_dict(), os.path.join(ckpt_dir, "optimizer.pt"))
                
                global_step += 1
            
            step += 1
            
        if steps > 0 and global_step >= steps: 
            break

    runtime = time.time() - start_time
    
    print(f"\nSaving final checkpoint to {out_dir}...")
    model.save_pretrained(out_dir)
    torch.save(optimizer.state_dict(), os.path.join(out_dir, "optimizer.pt"))
    print(f"TRAINING COMPLETE! Time: {runtime:.2f}s")
    
if __name__ == "__main__":
    main()
