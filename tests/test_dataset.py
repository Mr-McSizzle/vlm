import json
import pytest
import tempfile
import torch
from pathlib import Path
from dataset import VLMDataset, DataCollator

@pytest.fixture
def dummy_jsonl(tmp_path):
    file_path = tmp_path / "dummy.jsonl"
    records = [
        {"images": ["a.png"], "question": "Q1", "answer": "A1"},
        {"images": ["b.png"], "question": "Q2", "answer": "A2"}
    ]
    with open(file_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return str(file_path)

class DummyProcessor:
    def __init__(self):
        pass
    def __call__(self, text=None, images=None, return_tensors=None, padding=None):
        return {"input_ids": torch.tensor([[1,2,3], [4,5,6]])}
    @property
    def tokenizer(self):
        class DummyTokenizer:
            pad_token_id = 0
        return DummyTokenizer()

def test_vlm_dataset(dummy_jsonl):
    processor = DummyProcessor()
    ds = VLMDataset(dummy_jsonl, processor)
    assert len(ds) == 2
    item = ds[0]
    assert "text" in item

def test_data_collator():
    processor = DummyProcessor()
    collator = DataCollator(processor=processor)
    batch = [
        {"images": None, "text": "USER: Q1\nASSISTANT: A1"},
        {"images": None, "text": "USER: Q2\nASSISTANT: A2"}
    ]
    out = collator(batch)
    assert "input_ids" in out
    assert out["input_ids"].shape == (2, 3)

def test_unified_dataset_quality_gates():
    vlm_dir = Path(__file__).parent.parent
    unified_dir = vlm_dir / "data" / "unified"
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
                    img_path = Path(img)
                    if not img_path.is_absolute() and img_path.parts[0] == 'data':
                        img_path = vlm_dir / img_path
                    assert img_path.exists(), f"Missing image: {img}"
                    
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
