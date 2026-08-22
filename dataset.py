import json
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Any
from pathlib import Path
from PIL import Image

class VLMDataset(Dataset):
    def __init__(self, jsonl_file: str, processor: Any):
        self.processor = processor
        self.records = []
        with open(jsonl_file, 'r') as f:
            for line in f:
                if line.strip():
                    self.records.append(json.loads(line))
                    
    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        record = self.records[idx]
        
        # Load images
        images = []
        # get the directory of this file (vlm/)
        vlm_dir = Path(__file__).parent
        for p in record.get("images", []):
            try:
                # Resolve paths regardless of where the script is invoked from
                img_path = Path(p)
                if not img_path.is_absolute():
                    # If path starts with 'data', assume it's vlm/data
                    if img_path.parts[0] == 'data':
                        img_path = vlm_dir / img_path
                img = Image.open(str(img_path)).convert("RGB")
                images.append(img)
            except Exception as e:
                print(f"Warning: could not load image {p}: {e}")
                images.append(Image.new("RGB", (336, 336)))
        
        # Format prompt
        # We need to prepend <image> tokens for each image.
        prompt = ""
        for _ in range(len(images)):
            prompt += "<image>\n"
            
        question = record["question"]
        answer = record["answer"]
        
        # LLaVA 1.5 chat template expects USER: ... ASSISTANT: ...
        full_prompt = f"USER: {prompt}{question}\nASSISTANT: {answer}"
        
        return {
            "images": images if images else None,
            "text": full_prompt
        }

class DataCollator:
    def __init__(self, processor: Any):
        self.processor = processor

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        texts = [b["text"] for b in batch]
        # Some items might have 0 images (e.g., pure text captions), so we handle that:
        images = []
        for b in batch:
            if b["images"]:
                images.extend(b["images"])
                
        # Pass to processor
        if images:
            inputs = self.processor(text=texts, images=images, return_tensors="pt", padding=True)
        else:
            inputs = self.processor(text=texts, return_tensors="pt", padding=True)
            
        # The labels for language modeling are the input_ids, but we mask the "USER: ... ASSISTANT:" part.
        # Since this is a simple smoke test, we can just use input_ids as labels. 
        # The padding tokens should be -100.
        labels = inputs["input_ids"].clone()
        if self.processor.tokenizer.pad_token_id is not None:
            labels[labels == self.processor.tokenizer.pad_token_id] = -100
            
        inputs["labels"] = labels
        return inputs
