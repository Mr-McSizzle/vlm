import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from PIL import Image
import json

import sys
sys.path.append(str(Path(__file__).parent.parent))
from inference import vlm_answer, infer_task

@pytest.fixture
def mock_vlm_runner():
    with patch("inference.VLMRunner.get_instance") as mock_get_instance:
        mock_runner = MagicMock()
        mock_runner.infer.return_value = {
            "raw_answer": "Dummy answer",
            "checkpoint": "/dummy/checkpoint",
            "truncated": False
        }
        mock_get_instance.return_value = mock_runner
        yield mock_runner

@pytest.fixture
def dummy_image(tmp_path):
    img_path = tmp_path / "test.jpg"
    img = Image.new("RGB", (100, 100))
    img.save(img_path)
    return str(img_path)

@pytest.fixture
def dummy_image2(tmp_path):
    img_path = tmp_path / "test2.jpg"
    img = Image.new("RGB", (100, 100))
    img.save(img_path)
    return str(img_path)

# --- Output Parsing Tests ---

def test_normal_vqa_output(mock_vlm_runner, dummy_image):
    mock_vlm_runner.infer.return_value = {
        "raw_answer": "This is a test.",
        "checkpoint": "/dummy/checkpoint",
        "truncated": False
    }
    res = vlm_answer([dummy_image], "What is this?", task="vqa")
    assert res["task"] == "vqa"
    assert res["answer"] == "This is a test."
    assert res["regions"] == []
    assert res["confidence"] is None
    assert res["metadata"]["status"] == "ok"
    assert res["metadata"]["truncated"] is False

def test_change_vqa_output(mock_vlm_runner, dummy_image, dummy_image2):
    mock_vlm_runner.infer.return_value = {
        "raw_answer": "Yes, there is a change.",
        "checkpoint": "/dummy/checkpoint",
        "truncated": False
    }
    res = vlm_answer([dummy_image, dummy_image2], "What changed?", task="change_vqa")
    assert res["task"] == "change_vqa"
    assert res["answer"] == "Yes, there is a change."
    assert res["regions"] == []
    assert res["metadata"]["status"] == "ok"

def test_grounding_coordinates_valid(mock_vlm_runner, dummy_image):
    mock_vlm_runner.infer.return_value = {
        "raw_answer": "REGION: [10, 20.5, 30, 40] ANSWER: A building.",
        "checkpoint": "/dummy/checkpoint",
        "truncated": False
    }
    res = vlm_answer([dummy_image], "Where is the building?", task="grounding")
    assert res["task"] == "grounding"
    assert len(res["regions"]) == 1
    assert res["regions"][0] == {"x1": 10.0, "y1": 20.5, "x2": 30.0, "y2": 40.0}
    assert res["answer"] == "A building."
    assert res["metadata"]["status"] == "ok"

def test_grounding_coordinates_malformed(mock_vlm_runner, dummy_image):
    # Test out of bounds and x1 >= x2
    mock_vlm_runner.infer.return_value = {
        "raw_answer": "REGION: [110, 20, 30, 40] ANSWER: A building.",
        "checkpoint": "/dummy/checkpoint",
        "truncated": False
    }
    res = vlm_answer([dummy_image], "Where is the building?", task="grounding")
    assert res["task"] == "grounding"
    assert len(res["regions"]) == 0
    assert res["metadata"]["status"] == "warning"
    assert "Coordinates out of bounds" in res["metadata"]["parse_warning"]

def test_prompt_echo_removal(mock_vlm_runner, dummy_image):
    mock_vlm_runner.infer.return_value = {
        "raw_answer": "USER: <image>What is this? ASSISTANT: ANSWER: A forest.",
        "checkpoint": "/dummy/checkpoint",
        "truncated": False
    }
    res = vlm_answer([dummy_image], "What is this?", task="vqa")
    assert res["answer"] == "A forest."
    assert res["metadata"]["status"] == "ok"

def test_truncation_detection(mock_vlm_runner, dummy_image):
    mock_vlm_runner.infer.return_value = {
        "raw_answer": "The building is located at REGION: [10, 20, 30",
        "checkpoint": "/dummy/checkpoint",
        "truncated": False # We'll let the inference logic detect it via missing bracket
    }
    res = vlm_answer([dummy_image], "Where is it?", task="grounding")
    assert res["metadata"]["truncated"] is True
    assert res["metadata"]["status"] == "warning"
    assert "No bounding box found" in res["metadata"]["parse_warning"]

def test_evidence_metadata_confidence(mock_vlm_runner, dummy_image):
    evidence = {
        "type": "fusion",
        "optical_summary": "clear",
        "sar_summary": "strong response",
        "agreement": 0.9,
        "confidence": 0.95
    }
    res = vlm_answer([dummy_image], "Describe this.", task="vqa", evidence=evidence)
    assert res["metadata"]["evidence_used"] is True
    assert res["metadata"]["evidence_confidence"] == 0.95

def test_malformed_model_output(mock_vlm_runner, dummy_image):
    mock_vlm_runner.infer.return_value = {
        "raw_answer": "",
        "checkpoint": "/dummy/checkpoint",
        "truncated": False
    }
    res = vlm_answer([dummy_image], "Where is it?", task="grounding")
    assert res["metadata"]["status"] == "warning"
    assert "No bounding box found" in res["metadata"]["parse_warning"]

def test_json_serialization(mock_vlm_runner, dummy_image):
    res = vlm_answer([dummy_image], "What is this?", task="vqa")
    json_str = json.dumps(res)
    assert "satquery-vlm" in json_str

def test_repeated_inference(mock_vlm_runner, dummy_image):
    vlm_answer([dummy_image], "Q1")
    vlm_answer([dummy_image], "Q2")
    assert mock_vlm_runner.infer.call_count == 2

# --- Real Integration Test ---
@pytest.mark.integration
def test_real_integration(dummy_image, dummy_image2):
    pytest.skip("Skipping real integration on local env to avoid fatal access violation during model load.")
