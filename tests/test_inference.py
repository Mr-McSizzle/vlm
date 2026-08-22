import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from PIL import Image

import sys
sys.path.append(str(Path(__file__).parent.parent))
from inference import vlm_answer, infer_task

@pytest.fixture
def mock_vlm_runner():
    with patch("inference.VLMRunner.get_instance") as mock_get_instance:
        mock_runner = MagicMock()
        mock_runner.infer.return_value = {
            "raw_answer": "Dummy answer",
            "checkpoint": "/dummy/checkpoint"
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

# --- Unit Tests with Mocks ---

def test_vqa_valid(mock_vlm_runner, dummy_image):
    res = vlm_answer([dummy_image], "What is this?", task="vqa")
    assert res["metadata"]["status"] == "ok"
    assert res["answer"] == "Dummy answer"

def test_change_vqa_valid(mock_vlm_runner, dummy_image, dummy_image2):
    res = vlm_answer([dummy_image, dummy_image2], "What changed?", task="change_vqa")
    assert res["metadata"]["status"] == "ok"
    
def test_caption(mock_vlm_runner, dummy_image):
    res = vlm_answer([dummy_image], "Describe the image", task="caption")
    assert res["metadata"]["status"] == "ok"

def test_grounding_parser(mock_vlm_runner, dummy_image):
    mock_vlm_runner.infer.return_value = {
        "raw_answer": "REGION: [0.1, 0.2, 0.8, 0.9] ANSWER: A building.",
        "checkpoint": "/dummy/checkpoint"
    }
    res = vlm_answer([dummy_image], "Where is the building?", task="grounding")
    assert res["metadata"]["status"] == "ok"
    assert res["regions"] == [[0.1, 0.2, 0.8, 0.9]]
    assert res["answer"] == "A building."

def test_missing_image():
    res = vlm_answer(["non_existent.jpg"], "What is this?")
    assert res["metadata"]["status"] == "error"
    assert "FileNotFoundError" in res["metadata"]["error_type"]

def test_invalid_image_count(dummy_image):
    res = vlm_answer([dummy_image], "What changed?", task="change_vqa")
    assert res["metadata"]["status"] == "error"
    assert "ValueError" in res["metadata"]["error_type"]
    assert "requires exactly 2 images" in res["metadata"]["error_message"]

def test_evidence_provided(mock_vlm_runner, dummy_image):
    res = vlm_answer([dummy_image], "What is this?", evidence={"type": "test"})
    assert res["metadata"]["status"] == "ok"
    mock_vlm_runner.infer.assert_called_once()
    assert mock_vlm_runner.infer.call_args[0][3] == {"type": "test"}

def test_evidence_omitted(mock_vlm_runner, dummy_image):
    res = vlm_answer([dummy_image], "What is this?")
    assert res["metadata"]["status"] == "ok"
    assert mock_vlm_runner.infer.call_args[0][3] is None

def test_repeated_inference(mock_vlm_runner, dummy_image):
    vlm_answer([dummy_image], "Q1")
    vlm_answer([dummy_image], "Q2")
    assert mock_vlm_runner.infer.call_count == 2

def test_json_serialization(mock_vlm_runner, dummy_image):
    import json
    res = vlm_answer([dummy_image], "What is this?")
    # Should not throw exception
    json_str = json.dumps(res)
    assert "satquery-vlm" in json_str

# --- Real Integration Test ---

@pytest.mark.integration
def test_real_integration(dummy_image, dummy_image2):
    # This will fail if checkpoint does not exist or if it OOMs!
    try:
        res = vlm_answer([dummy_image], "Describe this image.", task="vqa")
        if res["metadata"]["status"] == "error":
            pytest.fail(f"Real integration failed: {res['metadata']}")
    except Exception as e:
        pytest.fail(f"Real integration threw exception: {e}")
