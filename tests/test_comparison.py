import pytest
import json
import os
import subprocess
from pathlib import Path

def test_comparison_canonical_schema(tmp_path):
    base_json = tmp_path / "evaluation_base.json"
    adapted_json = tmp_path / "evaluation_adapted.json"
    
    import sys
    from compare_models import main
    
    # Use the EXACT structure from evaluate_authoritative.py
    base_data = [{
        "id": "123", 
        "source_dataset": "rsvqa",
        "task": "vqa", 
        "images": ["test.jpg"],
        "question": "What is this?",
        "expected_answer": "car",
        "prediction": {
            "answer": "truck",
            "metadata": {"status": "ok"}
        }, 
        "model": "base",
        "checkpoint": "none",
        "success": True
    }]
    adapted_data = [{
        "id": "123", 
        "source_dataset": "rsvqa",
        "task": "vqa", 
        "images": ["test.jpg"],
        "question": "What is this?",
        "expected_answer": "car",
        "prediction": {
            "answer": "car",
            "metadata": {"status": "ok"}
        },
        "model": "adapted",
        "checkpoint": "checkpoints/final_adapter",
        "success": True
    }]
    
    with open(base_json, "w") as f: json.dump(base_data, f)
    with open(adapted_json, "w") as f: json.dump(adapted_data, f)
    
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    out_md = out_dir / "BASE_VS_ADAPTED_FINAL.md"
    
    # Stale content
    with open(out_md, "w") as f:
        f.write("large, red, and mostly empty field\nsuccessful domain alignment")
        
    sys.argv = ["compare_models.py", "--base", str(base_json), "--adapted", str(adapted_json)]
    main()
    
    with open(out_md, "r") as f:
        content = f.read()
        
    assert "large, red, and mostly empty field" not in content
    assert "truck" in content
    assert "car" in content
    assert "## Record ID: 123" in content
    assert content.count("## Record ID:") == 1
    # Check that datetime warning doesn't happen and timestamp format is generated_at = ...+...
    assert "generated_at = " in content

def test_comparison_hard_validations(tmp_path):
    base_json = tmp_path / "evaluation_base.json"
    adapted_json = tmp_path / "evaluation_adapted.json"
    
    import sys
    from compare_models import main
    
    # 1. ID mismatch
    with open(base_json, "w") as f: json.dump([{"id": "1", "model": "a", "checkpoint": "a"}], f)
    with open(adapted_json, "w") as f: json.dump([{"id": "2", "model": "a", "checkpoint": "a"}], f)
    sys.argv = ["compare_models.py", "--base", str(base_json), "--adapted", str(adapted_json)]
    with pytest.raises(ValueError, match="ID mismatch"):
        main()

    # 2. Expected answer differs
    base_data = [{"id": "1", "task": "vqa", "expected_answer": "yes", "model": "base", "checkpoint": "none"}]
    adapted_data = [{"id": "1", "task": "vqa", "expected_answer": "no", "model": "adapted", "checkpoint": "path"}]
    with open(base_json, "w") as f: json.dump(base_data, f)
    with open(adapted_json, "w") as f: json.dump(adapted_data, f)
    with pytest.raises(ValueError, match="Expected answer differs"):
        main()

    # 3. Task differs
    base_data = [{"id": "1", "task": "vqa", "expected_answer": "yes", "model": "base", "checkpoint": "none"}]
    adapted_data = [{"id": "1", "task": "change_vqa", "expected_answer": "yes", "model": "adapted", "checkpoint": "path"}]
    with open(base_json, "w") as f: json.dump(base_data, f)
    with open(adapted_json, "w") as f: json.dump(adapted_data, f)
    with pytest.raises(ValueError, match="Task differs"):
        main()

    # 4. Missing model provenance (for new schema)
    base_data = [{"id": "1", "task": "vqa", "expected_answer": "yes", "images": [], "timestamp": "now"}]
    adapted_data = [{"id": "1", "task": "vqa", "expected_answer": "yes", "images": [], "model": "adapted", "checkpoint": "path", "timestamp": "now"}]
    with open(base_json, "w") as f: json.dump(base_data, f)
    with open(adapted_json, "w") as f: json.dump(adapted_data, f)
    with pytest.raises(ValueError, match="Model provenance missing in base"):
        main()
