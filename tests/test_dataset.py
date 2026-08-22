import json
import pytest
import tempfile
import torch
from pathlib import Path
from vlm.dataset import VLMDataset, DataCollator

@pytest.fixture
def dummy_jsonl(tmp_path):
    file_path = tmp_path / "dummy.jsonl"
    records = [
        {"image_paths": ["a.png"], "prompt": "Q1", "target": "A1"},
        {"image_paths": ["b.png"], "prompt": "Q2", "target": "A2"}
    ]
    with open(file_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return str(file_path)

def test_vlm_dataset(dummy_jsonl):
    ds = VLMDataset(dummy_jsonl)
    assert len(ds) == 2
    item = ds[0]
    assert item["prompt"] == "Q1"
    assert item["target"] == "A1"

def test_data_collator():
    collator = DataCollator(processor=None)
    batch = [
        {"image_paths": ["a.png"], "prompt": "Q1", "target": "A1"},
        {"image_paths": ["b.png"], "prompt": "Q2", "target": "A2"}
    ]
    out = collator(batch)
    assert "input_ids" in out
    assert out["input_ids"].shape == (2, 3)

def test_unified_dataset_quality_gates():
    unified_dir = Path("vlm/data/unified")
    if not unified_dir.exists():
        pytest.skip("Unified dataset directory not found.")
        
    split_ids = {"train.jsonl": set(), "val.jsonl": set(), "test.jsonl": set()}
    
    for split in ["train.jsonl", "val.jsonl", "test.jsonl"]:
        filepath = unified_dir / split
        if not filepath.exists(): continue
            
        with open(filepath, "r") as f2:
            for line in f2:
                record = json.loads(line)
                assert "id" in record
                assert "source_dataset" in record
                assert "task" in record
                assert "images" in record
                assert "question" in record
                assert "answer" in record
                
                assert record["task"] in ["vqa", "change_vqa", "grounding", "caption"]
                if record["source_dataset"] == "bigearthnet":
                    assert len(record["images"]) == 0
                else:
                    assert len(record["images"]) in [1, 2]
                
                for img in record["images"]:
                    assert Path(img).exists(), f"Missing image: {img}"
                    
                assert len(record["question"].strip()) > 0
                assert len(record["answer"].strip()) > 0
                
                # Check duplicates within split
                assert record["id"] not in split_ids[split]
                split_ids[split].add(record["id"])
                
    # Check leakage
    train_ids = split_ids["train.jsonl"]
    val_ids = split_ids["val.jsonl"]
    test_ids = split_ids["test.jsonl"]
    
    assert train_ids.isdisjoint(val_ids), "Train/Val leakage detected!"
    assert train_ids.isdisjoint(test_ids), "Train/Test leakage detected!"
    assert val_ids.isdisjoint(test_ids), "Val/Test leakage detected!"
