import pytest
import os
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import sys
sys.path.append(str(Path(__file__).parent.parent))

from inference import VLMRunner

@patch("inference.LlavaForConditionalGeneration.from_pretrained")
@patch("inference.PeftModel.from_pretrained")
@patch("inference.AutoProcessor.from_pretrained")
def test_model_separation_base(mock_proc, mock_peft, mock_base):
    runner = VLMRunner(model_type="base", checkpoint_dir=None)
    runner.is_loaded = False
    runner.load_model()
    mock_base.assert_called_once()
    mock_peft.assert_not_called()
    assert runner.checkpoint_dir == Path("base")

@patch("inference.LlavaForConditionalGeneration.from_pretrained")
@patch("inference.PeftModel.from_pretrained")
@patch("inference.AutoProcessor.from_pretrained")
def test_model_separation_adapted(mock_proc, mock_peft, mock_base, tmp_path):
    ckpt = tmp_path / "checkpoints" / "final_adapter"
    ckpt.mkdir(parents=True)
    (ckpt / "adapter_config.json").write_text("{}")
    (ckpt / "adapter_model.safetensors").write_text("dummy weights")
    
    runner = VLMRunner(model_type="adapted", checkpoint_dir=str(ckpt))
    mock_peft.return_value.peft_config = {}
    
    runner.is_loaded = False
    runner.load_model()
    
    mock_base.assert_called_once()
    mock_peft.assert_called_once()

@patch("inference.LlavaForConditionalGeneration.from_pretrained")
@patch("inference.PeftModel.from_pretrained")
@patch("inference.AutoProcessor.from_pretrained")
def test_model_separation_adapter_missing(mock_proc, mock_peft, mock_base, tmp_path):
    ckpt = tmp_path / "checkpoints" / "final_adapter"
    ckpt.mkdir(parents=True)
    
    runner = VLMRunner(model_type="adapted", checkpoint_dir=str(ckpt))
    with pytest.raises(RuntimeError, match="Genuine checkpoint missing"):
        runner.load_model()
