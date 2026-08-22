import time
import re
import yaml
import argparse
import logging
from pathlib import Path
from typing import List, Union, Dict, Any, Optional
from PIL import Image

# Import prompt builder
from prompts import build_llava_prompt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VLMRunner:
    _instance = None

    def __init__(self):
        self.processor = None
        self.model = None
        self.max_new_tokens = 128
        self.checkpoint_dir = None
        self.is_loaded = False
        
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load_model(self):
        if self.is_loaded:
            return

        import torch
        from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
        from peft import PeftModel

        vlm_dir = Path(__file__).parent
        self.checkpoint_dir = vlm_dir / "checkpoints" / "final_adapter"

        # 1. Check for smoke-test marker
        smoke_marker = self.checkpoint_dir / "LOCAL_SMOKE_TEST_ONLY.txt"
        if smoke_marker.exists():
            raise RuntimeError(f"Model load blocked: Smoke test marker found in {self.checkpoint_dir}")

        # 2. Check if checkpoint exists
        if not (self.checkpoint_dir / "adapter_config.json").exists():
            raise RuntimeError(f"Model load blocked: Genuine checkpoint missing in {self.checkpoint_dir}")

        # Load config
        config_path = vlm_dir / "config.yaml"
        if config_path.exists():
            with open(config_path) as f:
                config = yaml.safe_load(f)
                self.max_new_tokens = config.get("generation", {}).get("max_new_tokens", 128)

        logger.info("Loading LLaVA-1.5 7B base model and processor...")
        self.processor = AutoProcessor.from_pretrained('llava-hf/llava-1.5-7b-hf')
        
        q_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        base_model = LlavaForConditionalGeneration.from_pretrained(
            'llava-hf/llava-1.5-7b-hf',
            quantization_config=q_config,
            device_map="auto"
        )
        
        logger.info(f"Loading adapter from {self.checkpoint_dir}...")
        self.model = PeftModel.from_pretrained(base_model, str(self.checkpoint_dir))
        self.model.eval()
        self.is_loaded = True

    def infer(self, images: List[Image.Image], question: str, task: str, evidence: dict = None) -> dict:
        import torch
        self.load_model()
        
        prompt = build_llava_prompt(question, len(images), task, evidence)
        
        inputs = self.processor(text=prompt, images=images, return_tensors="pt", padding=True).to(self.model.device)
        if 'pixel_values' in inputs:
            inputs['pixel_values'] = inputs['pixel_values'].to(torch.float16)
            
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=self.max_new_tokens,
                do_sample=False  # Deterministic generation
            )
            
        answer = self.processor.decode(outputs[0], skip_special_tokens=True)
        # Parse out the assistant response
        if "ASSISTANT: " in answer:
            answer = answer.split("ASSISTANT: ")[-1].strip()
        else:
            answer = answer.strip()
            
        return {"raw_answer": answer, "checkpoint": str(self.checkpoint_dir)}


def infer_task(num_images: int, question: str) -> str:
    if num_images == 2:
        return "change_vqa"
    q_lower = question.lower()
    if "describe" in q_lower or "caption" in q_lower:
        return "caption"
    if "where" in q_lower or "locate" in q_lower or "bounding box" in q_lower:
        return "grounding"
    return "vqa"

def vlm_answer(images: Union[str, List[str]], question: str, evidence: Optional[Any] = None, task: Optional[str] = None) -> dict:
    """
    Public interface for the SatQuery VLM.
    """
    t0 = time.time()
    
    # Normalize images to list
    if not isinstance(images, list):
        images = [images]
        
    try:
        # Load images
        loaded_images = []
        for img_path in images:
            if not Path(img_path).exists():
                raise FileNotFoundError(f"Image not found: {img_path}")
            try:
                loaded_images.append(Image.open(img_path).convert('RGB'))
            except Exception as e:
                raise ValueError(f"Unreadable image {img_path}: {e}")
                
        num_images = len(loaded_images)
        if num_images == 0:
            raise ValueError("Zero images provided.")
        if num_images > 2:
            raise ValueError(f"Too many images provided ({num_images}). Max is 2.")
            
        # Infer task if not provided
        if task is None:
            task = infer_task(num_images, question)
            
        # Validate task/image counts
        if task == "change_vqa" and num_images != 2:
            raise ValueError(f"change_vqa requires exactly 2 images. Got {num_images}.")
        elif task in ["vqa", "caption", "grounding"] and num_images != 1:
            raise ValueError(f"{task} requires exactly 1 image. Got {num_images}.")
            
        if task not in ["vqa", "change_vqa", "caption", "grounding"]:
            raise ValueError(f"Unsupported task: {task}")
            
        # Execute model
        runner = VLMRunner.get_instance()
        res = runner.infer(loaded_images, question, task, evidence)
        
        raw_answer = res["raw_answer"]
        checkpoint = res["checkpoint"]
        
        # Parse grounding if applicable
        regions = []
        parse_warning = None
        match = re.search(r'REGION:\s*\[([0-9\.\s,]+)\]', raw_answer)
        if match:
            try:
                coords = [float(x.strip()) for x in match.group(1).split(',')]
                if len(coords) == 4 and all(0.0 <= c <= 1.0 for c in coords):
                    regions.append(coords)
                else:
                    parse_warning = "Coordinates out of bounds or invalid format."
            except Exception:
                parse_warning = "Failed to parse region coordinates."
                
        # Clean answer text from region tag if parsed
        clean_answer = re.sub(r'REGION:\s*\[.*?\]', '', raw_answer).replace('ANSWER:', '').strip()

        metadata = {
            "task": task,
            "latency_seconds": round(time.time() - t0, 3),
            "status": "ok"
        }
        if parse_warning:
            metadata["parse_warning"] = parse_warning

        return {
            "answer": clean_answer,
            "regions": regions,
            "model": "satquery-vlm",
            "checkpoint": checkpoint,
            "metadata": metadata
        }

    except Exception as e:
        return {
            "answer": "",
            "regions": [],
            "model": "satquery-vlm",
            "checkpoint": "unknown",
            "metadata": {
                "status": "error",
                "error_type": type(e).__name__,
                "error_message": str(e)
            }
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLI interface for VLM")
    parser.add_argument("--images", nargs="+", required=True, help="Path(s) to image(s)")
    parser.add_argument("--question", type=str, required=True, help="Question to ask")
    parser.add_argument("--task", type=str, default=None, choices=["vqa", "change_vqa", "caption", "grounding"], help="Task type")
    
    args = parser.parse_args()
    
    result = vlm_answer(
        images=args.images,
        question=args.question,
        task=args.task
    )
    
    import json
    print(json.dumps(result, indent=2))
