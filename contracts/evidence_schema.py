from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, ValidationError

class ChangeDetectionEvidence(BaseModel):
    type: str = Field("change_detection", pattern="^change_detection$")
    changed_fraction: float
    dominant_region: str
    mask_path: Optional[str] = None

class SegmentationEvidence(BaseModel):
    type: str = Field("segmentation", pattern="^segmentation$")
    classes: List[str]
    regions: List[str]
    mask_path: Optional[str] = None

class FusionEvidence(BaseModel):
    type: str = Field("fusion", pattern="^fusion$")
    optical_summary: str
    sar_summary: str
    agreement: float
    confidence: float

def parse_evidence(evidence_dict: Any) -> Dict[str, Any]:
    """
    Parses and sanitizes the incoming evidence dictionary.
    Returns a dictionary containing the validated type and the formatted text context.
    """
    if not isinstance(evidence_dict, dict):
        raise ValueError("Evidence must be a JSON-serializable dictionary.")

    evidence_type = evidence_dict.get("type")
    
    evidence_confidence = None
    try:
        if evidence_type == "change_detection":
            parsed = ChangeDetectionEvidence(**evidence_dict)
            text = (f"This information was produced by an external perception model.\n"
                    f"External evidence indicates a changed fraction of {parsed.changed_fraction:.2f}, "
                    f"with the dominant region being '{parsed.dominant_region}'.")
        elif evidence_type == "segmentation":
            parsed = SegmentationEvidence(**evidence_dict)
            text = (f"This information was produced by an external perception model.\n"
                    f"External evidence identified classes: {', '.join(parsed.classes)} "
                    f"in regions: {', '.join(parsed.regions)}.")
        elif evidence_type == "fusion":
            parsed = FusionEvidence(**evidence_dict)
            text = (f"This information was produced by an external perception model.\n"
                    f"Optical summary: {parsed.optical_summary}. "
                    f"SAR summary: {parsed.sar_summary}. "
                    f"Agreement: {parsed.agreement:.2f}.")
            evidence_confidence = parsed.confidence
        else:
            raise ValueError(f"Unknown or missing evidence type: {evidence_type}")
    except ValidationError as e:
        raise ValueError(f"Malformed evidence object: {e}")

    return {
        "evidence_type": evidence_type,
        "text_context": text,
        "confidence": evidence_confidence
    }
