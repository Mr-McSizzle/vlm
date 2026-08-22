import pytest
from pydantic import ValidationError
from schemas import VLMRequest, VLMResponse, DatasetRecord

def test_vlm_request_valid():
    req = VLMRequest(images=["img1.jpg"], question="What is this?")
    assert req.task_type == "vqa"
    assert req.images == ["img1.jpg"]

def test_vlm_request_missing_fields():
    with pytest.raises(ValidationError):
        VLMRequest(images=["img1.jpg"]) # missing question

def test_vlm_response_valid():
    res = VLMResponse(answer="A building", confidence=0.9)
    assert res.answer == "A building"
    assert res.confidence == 0.9

def test_dataset_record():
    rec = DatasetRecord(image_paths=["a.png"], prompt="Q", target="A")
    assert rec.prompt == "Q"
