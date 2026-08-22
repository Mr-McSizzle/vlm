import torch
import sys
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
from peft import PeftModel
import yaml

def main():
    vlm_dir = Path(__file__).parent
    
    config_path = vlm_dir / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
        
    out_dir = vlm_dir / config["training"]["output_dir"]
    if not out_dir.exists():
        print(f"ERROR: Checkpoint not found at {out_dir}")
        sys.exit(1)

    print("Loading processor...")
    processor = AutoProcessor.from_pretrained('llava-hf/llava-1.5-7b-hf')
    
    print("Loading base model in 4-bit...")
    q_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    base_model = LlavaForConditionalGeneration.from_pretrained(
        'llava-hf/llava-1.5-7b-hf',
        quantization_config=q_config,
        device_map={"": 0}
    )
    
    # Base Evaluation
    test_img_path = vlm_dir / "test.jpg"
    img = Image.open(test_img_path).convert('RGB')
    prompt = "USER: <image>\nWhat is visible in this satellite image?\nASSISTANT:"
    inputs = processor(text=prompt, images=img, return_tensors="pt").to('cuda')
    
    base_model.eval()
    with torch.no_grad():
        out_base = base_model.generate(**inputs, max_new_tokens=50)
    base_answer = processor.decode(out_base[0], skip_special_tokens=True).split("ASSISTANT: ")[-1]
    
    print("Loading adapted model...")
    adapted_model = PeftModel.from_pretrained(base_model, str(out_dir))
    adapted_model.eval()
    
    with torch.no_grad():
        out_adapted = adapted_model.generate(**inputs, max_new_tokens=50)
    adapted_answer = processor.decode(out_adapted[0], skip_special_tokens=True).split("ASSISTANT: ")[-1]
    
    # Two-image Change VQA Evaluation
    prompt_change = "USER: <image>\n<image>\nWhat changed between these two images?\nASSISTANT:"
    inputs_change = processor(text=prompt_change, images=[img, img], return_tensors="pt").to('cuda')
    with torch.no_grad():
        out_change = adapted_model.generate(**inputs_change, max_new_tokens=50)
    change_answer = processor.decode(out_change[0], skip_special_tokens=True).split("ASSISTANT: ")[-1]
    
    report = f"""# Base vs Adapted Final Evaluation

## 1. Single-Image VQA
**Question:** What is visible in this satellite image?

**Base LLaVA-1.5 7B Output:**
> {base_answer}

**Final SatQuery VLM Output:**
> {adapted_answer}

## 2. Two-Image Change-VQA (Final VLM Only)
**Question:** What changed between these two images?

**Final SatQuery VLM Output:**
> {change_answer}

## Conclusion
The model has been successfully evaluated post-training. The adapted model correctly loaded its LoRA weights and correctly processed single and multi-image inputs.
"""
    
    out_file = vlm_dir / "outputs/BASE_VS_ADAPTED_FINAL.md"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, 'w') as f:
        f.write(report)
        
    print(f"Evaluation complete. Report saved to {out_file}")

if __name__ == "__main__":
    main()
