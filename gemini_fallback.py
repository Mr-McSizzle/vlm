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
    Fallback Gemini verification path using the new google-genai SDK.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set")
        
    model_name = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")
    timeout = int(os.environ.get("GEMINI_TIMEOUT_SECONDS", "60"))
    
    if len(images) > 2:
        raise ValueError("Max 2 images supported")
        
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        raise RuntimeError("google-genai package not installed")
        
    http_options = types.HttpOptions(
        timeout=timeout,
        retry_options=types.HttpRetryOptions(attempts=2)
    )
    client = genai.Client(api_key=api_key, http_options=http_options)
    
    contents = []
    if evidence:
        contents.append(f"Context: {evidence}")
    contents.append(f"Question: {question}")
    contents.extend(images)
    
    config = types.GenerateContentConfig(
        max_output_tokens=256
    )
    
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=contents,
            config=config
        )
        text = response.text.strip()
    except Exception as e:
        err_str = str(e).lower()
        if "timeout" in err_str or "deadline" in err_str:
            raise RuntimeError("timeout")
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
