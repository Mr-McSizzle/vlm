import pytest
import os
import json
import sys
from unittest.mock import patch, MagicMock
from PIL import Image

from inference import vlm_answer
from gemini_fallback import gemini_answer

# Create mocks for google-genai
mock_google = MagicMock()
mock_genai = MagicMock()
mock_types = MagicMock()

sys.modules["google"] = mock_google
sys.modules["google.genai"] = mock_genai
sys.modules["google.genai.types"] = mock_types

# Expose genai within google
mock_google.genai = mock_genai

@pytest.fixture
def dummy_images(tmp_path):
    img_path = tmp_path / "test.jpg"
    img = Image.new("RGB", (10, 10))
    img.save(img_path)
    img_path2 = tmp_path / "test2.jpg"
    img2 = Image.new("RGB", (10, 10))
    img2.save(img_path2)
    return [str(img_path), str(img_path2)]

def test_gemini_not_called_when_primary_succeeds(dummy_images, monkeypatch):
    monkeypatch.setenv("GEMINI_FALLBACK_ENABLED", "true")
    
    with patch("inference.VLMRunner.get_instance") as mock_runner:
        mock_instance = MagicMock()
        mock_instance.infer.return_value = {
            "raw_answer": "valid answer",
            "checkpoint": "test",
            "truncated": False
        }
        mock_runner.return_value = mock_instance
        
        with patch("gemini_fallback.gemini_answer") as mock_gemini:
            res = vlm_answer(dummy_images[0], "What is this?", task="vqa")
            
            mock_gemini.assert_not_called()
            assert res["metadata"]["fallback_used"] is False
            assert res["answer"] == "valid answer"

def test_gemini_called_on_primary_failure(dummy_images, monkeypatch):
    monkeypatch.setenv("GEMINI_FALLBACK_ENABLED", "true")
    monkeypatch.setenv("GEMINI_API_KEY", "fake_key")
    
    with patch("inference.VLMRunner.get_instance") as mock_runner:
        mock_instance = MagicMock()
        mock_instance.infer.side_effect = RuntimeError("GPU OOM")
        mock_runner.return_value = mock_instance
        
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Gemini answer"
        mock_client.models.generate_content.return_value = mock_response
        mock_genai.Client.return_value = mock_client
        
        res = vlm_answer(dummy_images[0], "What is this?", task="vqa")
        
        assert res["metadata"]["fallback_used"] is True
        assert res["metadata"]["fallback_reason"] == "primary VLM error"
        assert res["answer"] == "Gemini answer"
        assert res["metadata"]["fallback_provider"] == "gemini"
        assert res["metadata"]["primary_model"] == "satquery-vlm"

def test_gemini_called_on_truncation(dummy_images, monkeypatch):
    monkeypatch.setenv("GEMINI_FALLBACK_ENABLED", "true")
    monkeypatch.setenv("GEMINI_API_KEY", "fake_key")
    
    with patch("inference.VLMRunner.get_instance") as mock_runner:
        mock_instance = MagicMock()
        mock_instance.infer.return_value = {
            "raw_answer": "truncated answer",
            "checkpoint": "test",
            "truncated": True
        }
        mock_runner.return_value = mock_instance
        
        with patch("gemini_fallback.gemini_answer") as mock_gemini:
            mock_gemini.return_value = {
                "answer": "Gemini complete answer",
                "metadata": {}
            }
            
            res = vlm_answer(dummy_images[0], "What is this?", task="vqa")
            
            mock_gemini.assert_called_once()
            assert res["metadata"]["fallback_used"] is True
            assert res["metadata"]["fallback_reason"] == "primary VLM truncated output"

def test_gemini_receives_both_images_for_change_vqa(dummy_images, monkeypatch):
    monkeypatch.setenv("GEMINI_FALLBACK_ENABLED", "true")
    monkeypatch.setenv("GEMINI_API_KEY", "fake_key")
    
    with patch("inference.VLMRunner.get_instance") as mock_runner:
        mock_instance = MagicMock()
        mock_instance.infer.return_value = {
            "raw_answer": "", # Empty answer triggers fallback
            "checkpoint": "test",
            "truncated": False
        }
        mock_runner.return_value = mock_instance
        
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Gemini answer"
        mock_client.models.generate_content.return_value = mock_response
        mock_genai.Client.return_value = mock_client
        
        res = vlm_answer(dummy_images, "Changed?", task="change_vqa")
        
        mock_client.models.generate_content.assert_called_once()
        args, kwargs = mock_client.models.generate_content.call_args
        contents = kwargs["contents"]
        assert len(contents) == 3
        assert isinstance(contents[1], Image.Image)
        assert isinstance(contents[2], Image.Image)

def test_gemini_receives_evidence(dummy_images, monkeypatch):
    monkeypatch.setenv("GEMINI_FALLBACK_ENABLED", "true")
    monkeypatch.setenv("GEMINI_API_KEY", "fake_key")
    
    with patch("inference.VLMRunner.get_instance") as mock_runner:
        mock_instance = MagicMock()
        mock_instance.infer.side_effect = RuntimeError("Fail")
        mock_runner.return_value = mock_instance
        
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Gemini answer"
        mock_client.models.generate_content.return_value = mock_response
        mock_genai.Client.return_value = mock_client
        
        evidence = {
            "type": "change_detection",
            "changed_fraction": 0.5,
            "dominant_region": "north"
        }
        
        res = vlm_answer(dummy_images[0], "What?", evidence=evidence, task="vqa")
        
        args, kwargs = mock_client.models.generate_content.call_args
        contents = kwargs["contents"]
        assert any("Context:" in str(p) for p in contents)
        assert any("north" in str(p) for p in contents)

def test_gemini_missing_api_key(dummy_images, monkeypatch):
    monkeypatch.setenv("GEMINI_FALLBACK_ENABLED", "true")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    
    with patch("inference.VLMRunner.get_instance") as mock_runner:
        mock_instance = MagicMock()
        mock_instance.infer.side_effect = RuntimeError("Fail")
        mock_runner.return_value = mock_instance
        
        res = vlm_answer(dummy_images[0], "What is this?", task="vqa")
        
        assert res["metadata"]["fallback_used"] is False
        assert res["metadata"]["fallback_unavailable"] is True
        assert res["metadata"]["status"] == "error"

def test_gemini_api_failure(dummy_images, monkeypatch):
    monkeypatch.setenv("GEMINI_FALLBACK_ENABLED", "true")
    monkeypatch.setenv("GEMINI_API_KEY", "fake_key")
    
    with patch("inference.VLMRunner.get_instance") as mock_runner:
        mock_instance = MagicMock()
        mock_instance.infer.side_effect = RuntimeError("Primary Fail")
        mock_runner.return_value = mock_instance
        
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = Exception("API Server Down")
        mock_genai.Client.return_value = mock_client
        
        res = vlm_answer(dummy_images[0], "What is this?", task="vqa")
        
        assert res["metadata"]["fallback_used"] is False
        assert "API Server Down" in res["metadata"]["fallback_error"]
        assert res["metadata"]["status"] == "error"
        
def test_json_serialization(dummy_images, monkeypatch):
    monkeypatch.setenv("GEMINI_FALLBACK_ENABLED", "true")
    monkeypatch.setenv("GEMINI_API_KEY", "fake_key")
    
    with patch("inference.VLMRunner.get_instance") as mock_runner:
        mock_instance = MagicMock()
        mock_instance.infer.side_effect = RuntimeError("Primary Fail")
        mock_runner.return_value = mock_instance
        
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Gemini answer"
        mock_client.models.generate_content.return_value = mock_response
        mock_genai.Client.return_value = mock_client
        
        res = vlm_answer(dummy_images[0], "What is this?", task="vqa")
        
        try:
            json_str = json.dumps(res)
            assert "Gemini answer" in json_str
        except Exception as e:
            pytest.fail(f"Result not JSON serializable: {e}")

def test_gemini_api_timeout(dummy_images, monkeypatch):
    monkeypatch.setenv("GEMINI_FALLBACK_ENABLED", "true")
    monkeypatch.setenv("GEMINI_API_KEY", "fake_key")
    monkeypatch.setenv("GEMINI_TIMEOUT_SECONDS", "60")
    
    with patch("inference.VLMRunner.get_instance") as mock_runner:
        mock_instance = MagicMock()
        mock_instance.infer.side_effect = RuntimeError("Primary Fail")
        mock_runner.return_value = mock_instance
        
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = Exception("ReadTimeout occurred")
        mock_genai.Client.return_value = mock_client
        
        res = vlm_answer(dummy_images[0], "What is this?", task="vqa")
        
        mock_genai.Client.assert_called()
        args, kwargs = mock_genai.Client.call_args
        assert "http_options" in kwargs
        
        assert res["metadata"]["fallback_used"] is False
        assert res["metadata"]["fallback_error"] == "timeout"
        assert res["metadata"]["status"] == "error"

def test_gemini_model_selection(dummy_images, monkeypatch):
    monkeypatch.setenv("GEMINI_FALLBACK_ENABLED", "true")
    monkeypatch.setenv("GEMINI_API_KEY", "fake_key")
    
    with patch("inference.VLMRunner.get_instance") as mock_runner:
        mock_instance = MagicMock()
        mock_instance.infer.side_effect = RuntimeError("Primary Fail")
        mock_runner.return_value = mock_instance
        
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Gemini answer"
        mock_client.models.generate_content.return_value = mock_response
        mock_genai.Client.return_value = mock_client
        
        # Test Default
        monkeypatch.delenv("GEMINI_MODEL", raising=False)
        res = vlm_answer(dummy_images[0], "What is this?", task="vqa")
        
        args, kwargs = mock_client.models.generate_content.call_args
        assert kwargs["model"] == "gemini-3.5-flash-lite"
        
        # Test Override
        monkeypatch.setenv("GEMINI_MODEL", "gemini-2.0-pro")
        res2 = vlm_answer(dummy_images[0], "What is this?", task="vqa")
        args2, kwargs2 = mock_client.models.generate_content.call_args
        assert kwargs2["model"] == "gemini-2.0-pro"
