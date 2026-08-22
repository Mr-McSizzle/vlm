import os
import logging
from PIL import Image
from typing import List, Optional

logger = logging.getLogger(__name__)

def gemini_answer(
    images: List[Image.Image],
    question: str,
    evidence: Optional[str] = None,
    task: Optional[str] = None
) -> dict:
    """
    Fallback Gemini verification path.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set")
        
    model_name = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")
    timeout = int(os.environ.get("GEMINI_TIMEOUT_SECONDS", "30"))
    
    if len(images) > 2:
        raise ValueError("Max 2 images supported")
        
    try:
        import google.generativeai as genai
    except ImportError:
        raise RuntimeError("google-generativeai package not installed")
        
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    
    prompt_parts = []
    if evidence:
        prompt_parts.append(f"Context: {evidence}")
    prompt_parts.append(f"Question: {question}")
    prompt_parts.extend(images)
    
    generation_config = genai.types.GenerationConfig(
        max_output_tokens=256
    )
    
    try:
        response = model.generate_content(
            prompt_parts,
            generation_config=generation_config,
            request_options={"timeout": timeout}
        )
        text = response.text.strip()
    except Exception as e:
        raise RuntimeError(f"Gemini API failure: {str(e)}")
        
    return {
        "answer": text,
        "task": task if task else "unknown",
        "regions": [],
        "confidence": None,
        "model": "gemini-fallback",
        "checkpoint": None,
        "metadata": {
            "status": "ok",
            "provider": "gemini",
            "fallback": True
        }
    }
