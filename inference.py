import torch
from typing import List, Optional, Any
try:
    from PIL import Image
except ImportError:
    pass

from .schemas import VLMRequest, VLMResponse
from .prompts import get_prompt

# Placeholder for actual model loading logic
# from transformers import AutoProcessor, AutoModelForCausalLM

class SatQueryVLM:
    def __init__(self, model_name: str = "llava-hf/llava-1.5-7b-hf"):
        self.model_name = model_name
        # self.processor = AutoProcessor.from_pretrained(model_name)
        # self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
        print(f"Initialized dummy SatQueryVLM with {model_name}")

    def generate(self, request: VLMRequest) -> VLMResponse:
        prompt = get_prompt(request.task_type, request.question, request.evidence)
        
        # Dummy response logic
        return VLMResponse(
            answer=f"Dummy response to: {prompt}",
            confidence=0.99,
            raw_output={"dummy": True}
        )

# Global singleton for easy import
_vlm_instance = None

def vlm_answer(images: List[Any], question: str, evidence: Optional[str] = None) -> dict:
    """
    Main interface for VLM answering.
    """
    global _vlm_instance
    if _vlm_instance is None:
        _vlm_instance = SatQueryVLM()
        
    req = VLMRequest(images=images, question=question, evidence=evidence, task_type="vqa")
    resp = _vlm_instance.generate(req)
    return resp.model_dump()
