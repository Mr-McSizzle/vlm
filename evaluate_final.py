import json
import time
import argparse
from pathlib import Path
from PIL import Image

def evaluate_final(model_type, checkpoint_path=None):
    import torch
    from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
    from peft import PeftModel
    import yaml

    vlm_dir = Path(__file__).parent
    
    config_path = vlm_dir / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
        
    max_tokens = config.get("generation", {}).get("max_new_tokens", 128)

    print("Loading processor...")
    processor = AutoProcessor.from_pretrained('llava-hf/llava-1.5-7b-hf')
    
    print("Loading base model in 4-bit...")
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
    
    if model_type == "adapted":
        if not checkpoint_path:
            checkpoint_path = vlm_dir / config["training"]["output_dir"]
        print(f"Loading adapted model from {checkpoint_path}...")
        model = PeftModel.from_pretrained(model, str(checkpoint_path))
    
    model.eval()

    test_file = vlm_dir / "data/unified/test.jsonl"
    records = []
    with open(test_file) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
                
    results = []
    
    for record in records:
        images_loaded = []
        for img_rel in record.get("images", []):
            img_path = Path(img_rel)
            if not img_path.is_absolute():
                img_path = vlm_dir / img_path
            images_loaded.append(Image.open(img_path).convert('RGB'))
            
        prompt = "USER: " + "<image>\n" * len(images_loaded) + record["question"] + "\nASSISTANT:"
        
        t0 = time.time()
        if images_loaded:
            inputs = processor(text=prompt, images=images_loaded, return_tensors="pt", padding=True).to('cuda')
            if 'pixel_values' in inputs:
                inputs['pixel_values'] = inputs['pixel_values'].to(torch.float16)
        else:
            inputs = processor(text=prompt, return_tensors="pt", padding=True).to('cuda')

        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_tokens)
            
        ans = processor.decode(out[0], skip_special_tokens=True).split("ASSISTANT: ")[-1].strip()
        latency = time.time() - t0
        
        results.append({
            "id": record["id"],
            "task": record["task"],
            "source_dataset": record["source_dataset"],
            "images": record.get("images", []),
            "question": record["question"],
            "expected_answer": record["answer"],
            "model_answer": ans,
            "checkpoint": "base" if model_type == "base" else str(checkpoint_path),
            "latency_seconds": latency
        })
        
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["base", "adapted"], required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output", type=str, default="outputs/FINAL_EVALUATION_RAW.json")
    args = parser.parse_args()
    
    results = evaluate_final(args.model, args.checkpoint)
    
    out_file = Path(__file__).parent / args.output
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved evaluation to {out_file}")
