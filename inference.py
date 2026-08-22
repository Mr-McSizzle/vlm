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
            llm_int8_enable_fp32_cpu_offload=True
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
            
        generated_ids = outputs[0]
        input_length = inputs['input_ids'].shape[1]
        num_generated = len(generated_ids) - input_length
        truncated = (num_generated >= self.max_new_tokens)
        
        answer = self.processor.decode(generated_ids, skip_special_tokens=True)
            
        return {"raw_answer": answer, "checkpoint": str(self.checkpoint_dir), "truncated": truncated}


def infer_task(num_images: int, question: str) -> str:
    if num_images == 2:
        return "change_vqa"
    q_lower = question.lower()
    if "describe" in q_lower or "caption" in q_lower:
        return "caption"
    if "where" in q_lower or "locate" in q_lower or "bounding box" in q_lower:
        return "grounding"
    return "vqa"

def clean_model_output(raw_text: str) -> str:
    clean_text = raw_text
    
    # Remove everything before and including ASSISTANT: if present
    if "ASSISTANT:" in clean_text:
        clean_text = clean_text.split("ASSISTANT:")[-1]
    elif "ASSISTANT" in clean_text:
        clean_text = clean_text.split("ASSISTANT")[-1]
        
    # Remove USER: echoes if any
    if "USER:" in clean_text:
        clean_text = clean_text.split("USER:")[0]
        
    # Strip obvious ANSWER: headers
    clean_text = re.sub(r'^\s*ANSWER:\s*', '', clean_text, flags=re.IGNORECASE)
    
    return clean_text.strip()

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
            
        # Process external evidence if provided
        evidence_text = None
        evidence_metadata = {
            "evidence_used": False,
            "evidence_type": None,
            "evidence_source": None
        }
        evidence_confidence = None
        if evidence is not None:
            from contracts.evidence_schema import parse_evidence
            parsed_result = parse_evidence(evidence)
            evidence_text = parsed_result["text_context"]
            evidence_metadata = {
                "evidence_used": True,
                "evidence_type": parsed_result["evidence_type"],
                "evidence_source": "external_perception"
            }
            if parsed_result.get("confidence") is not None:
                evidence_confidence = parsed_result["confidence"]
            
        # Execute model
        runner = VLMRunner.get_instance()
        res = runner.infer(loaded_images, question, task, evidence_text)
        
        raw_answer = res["raw_answer"]
        checkpoint = res["checkpoint"]
        truncated = res["truncated"]
        
        # Clean the output text
        clean_answer = clean_model_output(raw_answer)
        
        # Detect incomplete grounding via missing bracket
        if task == "grounding" and "REGION:" in clean_answer and "]" not in clean_answer.split("REGION:")[-1]:
            truncated = True

        # Parse grounding if applicable
        regions = []
        parse_warning = None
        match = re.search(r'REGION:\s*\[([0-9\.\s,-]+)\]', clean_answer)
        if match:
            try:
                coords = [float(x.strip()) for x in match.group(1).split(',')]
                if len(coords) == 4:
                    x1, y1, x2, y2 = coords
                    if 0 <= x1 <= 100 and 0 <= y1 <= 100 and 0 <= x2 <= 100 and 0 <= y2 <= 100 and x1 < x2 and y1 < y2:
                        regions.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2})
                    else:
                        parse_warning = "Coordinates out of bounds (0-100) or invalid dimensions."
                else:
                    parse_warning = "Invalid number of coordinates in region."
            except Exception:
                parse_warning = "Failed to parse region coordinates."
                
        elif task == "grounding":
            parse_warning = "No bounding box found in model output."
            
        # Strip the region tag if successfully found, or if we want it completely clean
        clean_answer = re.sub(r'REGION:\s*\[.*?\]', '', clean_answer).strip()
        # Sometimes ANSWER: follows the REGION tag
        clean_answer = re.sub(r'^\s*ANSWER:\s*', '', clean_answer, flags=re.IGNORECASE).strip()

        metadata = {
            "latency_seconds": round(time.time() - t0, 3),
            "status": "warning" if parse_warning else "ok",
            "truncated": truncated,
            "raw_output_available": True
        }
        metadata.update(evidence_metadata)
        if evidence_confidence is not None:
            metadata["evidence_confidence"] = evidence_confidence
        if parse_warning:
            metadata["parse_warning"] = parse_warning

        return {
            "answer": clean_answer,
            "task": task,
            "regions": regions,
            "confidence": None,
            "model": "satquery-vlm",
            "checkpoint": checkpoint,
            "metadata": metadata
        }

    except Exception as e:
        return {
            "answer": "",
            "task": task if task else "unknown",
            "regions": [],
            "confidence": None,
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
