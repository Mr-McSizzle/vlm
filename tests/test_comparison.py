import pytest
import json
import os
import subprocess
from pathlib import Path

def test_comparison_overwrite_stale(tmp_path):
    # Set up dummy evaluation jsons
    base_json = tmp_path / "evaluation_base.json"
    adapted_json = tmp_path / "evaluation_adapted.json"
    
    # We will simulate calling compare_models.py. Let's patch sys.argv and call main
    import sys
    from compare_models import main
    
    # Write initial JSON
    base_data = [{
        "id": "123", "task": "vqa", "question": "What is this?", "expected_answer": "car",
        "prediction": {"answer": "truck"}, "success": True
    }]
    adapted_data = [{
        "id": "123", "task": "vqa", "question": "What is this?", "expected_answer": "car",
        "prediction": {"answer": "car"}, "success": True
    }]
    
    with open(base_json, "w") as f: json.dump(base_data, f)
    with open(adapted_json, "w") as f: json.dump(adapted_data, f)
    
    # Write STALE content to BASE_VS_ADAPTED_FINAL.md in the current working dir
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    out_md = out_dir / "BASE_VS_ADAPTED_FINAL.md"
    
    with open(out_md, "w") as f:
        f.write("large, red, and mostly empty field\nsuccessful domain alignment")
        
    # Execute comparison
    sys.argv = ["compare_models.py", "--base", str(base_json), "--adapted", str(adapted_json)]
    main()
    
    # Verify stale content is gone
    with open(out_md, "r") as f:
        content = f.read()
        
    assert "large, red, and mostly empty field" not in content
    assert "successful domain alignment" not in content
    
    # Verify exact predictions
    assert "truck" in content
    assert "car" in content
    
    # Verify exact ID sections
    assert "## Record ID: 123" in content
    assert "## Record ID:" in content
    assert content.count("## Record ID:") == 1

def test_comparison_id_mismatch(tmp_path):
    base_json = tmp_path / "evaluation_base.json"
    adapted_json = tmp_path / "evaluation_adapted.json"
    
    import sys
    from compare_models import main
    
    # Write JSON with mismatching IDs
    base_data = [{"id": "123"}]
    adapted_data = [{"id": "456"}]
    
    with open(base_json, "w") as f: json.dump(base_data, f)
    with open(adapted_json, "w") as f: json.dump(adapted_data, f)
    
    sys.argv = ["compare_models.py", "--base", str(base_json), "--adapted", str(adapted_json)]
    with pytest.raises(ValueError, match="ID mismatch between base and adapted evaluations"):
        main()
