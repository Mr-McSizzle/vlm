from typing import Optional, List, Any
from pydantic import BaseModel, Field

class VLMRequest(BaseModel):
    images: List[Any] = Field(..., description="List of image paths, URLs, or PIL Images")
    question: str = Field(..., description="The question or prompt to ask")
    evidence: Optional[str] = Field(None, description="Optional text evidence/context")
    task_type: str = Field("vqa", description="Type of task: vqa, captioning, change_vqa, grounding")

class VLMResponse(BaseModel):
    answer: str = Field(..., description="The generated answer from the VLM")
    confidence: Optional[float] = Field(None, description="Optional confidence score")
    raw_output: Optional[Any] = Field(None, description="Raw model output dictionary")

class DatasetRecord(BaseModel):
    image_paths: List[str]
    prompt: str
    target: str
