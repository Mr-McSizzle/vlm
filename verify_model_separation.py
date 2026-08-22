import os
import hashlib
from pathlib import Path

def hash_file(filepath):
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        buf = f.read(65536)
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(65536)
    return hasher.hexdigest()

def verify_separation():
    print("=== MODEL SEPARATION DIAGNOSTIC ===")
    
    ckpt_dir = Path("checkpoints/final_adapter")
    adapter_file = ckpt_dir / "adapter_model.safetensors"
    
    print("\n1. Adapter Weight File Verification:")
    if adapter_file.exists():
        size = adapter_file.stat().st_size
        checksum = hash_file(adapter_file)
        print(f"  [OK] File exists: {adapter_file}")
        print(f"  [OK] Size: {size / (1024*1024):.2f} MB")
        print(f"  [OK] SHA256: {checksum}")
        if size == 0:
            print("  [ERROR] File is empty!")
    else:
        print(f"  [ERROR] File missing: {adapter_file}")
        return
        
    print("\n2. Model State Verification:")
    # We will simulate loading using safetensors to count tensors and check if they are zero
    try:
        from safetensors import safe_open
        import torch
        tensors = []
        zero_tensors = 0
        total_params = 0
        with safe_open(adapter_file, framework="pt", device="cpu") as f:
            for k in f.keys():
                tensors.append(k)
                t = f.get_tensor(k)
                total_params += t.numel()
                if torch.all(t == 0):
                    zero_tensors += 1
                    
        print(f"  [OK] Total LoRA tensors: {len(tensors)}")
        print(f"  [OK] Total trainable parameters: {total_params}")
        print(f"  [OK] Tensors strictly zero: {zero_tensors} (expected 0 or very small number due to init)")
        
        if total_params == 0:
            print("  [ERROR] No parameters found!")
    except ImportError:
        print("  [SKIP] safetensors or torch not installed. Cannot inspect weights deeply.")
        
    print("\n3. Testing API Separation:")
    from inference import vlm_answer
    
    print("\n  [TEST A: BASE MODEL]")
    img = "data/external/cdvqa/images/cdvqa-train-00000008_1.jpg"
    try:
        res_base = vlm_answer([img], "What is this?", task="vqa", model_type="base")
        print(f"  [Result Base] checkpoint={res_base['checkpoint']}, adapter_present={res_base['metadata'].get('adapter_present')}")
    except Exception as e:
        print(f"  [Intercepted Base VRAM limit / Error]: {e}")
        
    print("\n  [TEST B: ADAPTED MODEL]")
    try:
        res_adapted = vlm_answer([img], "What is this?", task="vqa", model_type="adapted", checkpoint=str(ckpt_dir))
        print(f"  [Result Adapted] checkpoint={res_adapted['checkpoint']}, adapter_present={res_adapted['metadata'].get('adapter_present')}")
    except Exception as e:
        print(f"  [Intercepted Adapted VRAM limit / Error]: {e}")
        
if __name__ == "__main__":
    verify_separation()
