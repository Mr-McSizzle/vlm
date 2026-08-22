import json
import argparse
import random
import yaml
from pathlib import Path
from datetime import datetime

def load_config(config_path="config.yaml"):
    if not Path(config_path).exists():
        return {}
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def build_unified_dataset(config, config_path):
    vlm_dir = Path(config_path).parent
    unified_dir = vlm_dir / "data/unified"
    unified_dir.mkdir(parents=True, exist_ok=True)
    
    records = []
    skipped = 0
    invalid = 0
    duplicates = 0
    seen_ids = set()
    
    # In a real scenario, this would load BigEarthNet, VRSBench, RSVQA, CDVQA
    # Because the source files are missing, the processed lists will be empty.
    # The pipeline architecture is established here.
    
    datasets_to_process = config.get("dataset", {}).get("sources", {})
    
    for ds_name, ds_info in datasets_to_process.items():
        if not ds_info.get("enabled", False):
            continue
            
        ds_path = Path(ds_info.get("path", ""))
        if not ds_path.is_absolute() and ds_path.parts and ds_path.parts[0] == 'data':
            ds_path = vlm_dir / ds_path
            
        if not ds_path.exists():
            print(f"Warning: Dataset path {ds_path} does not exist. Skipping {ds_name}.")
            continue
            
        print(f"Processing {ds_name}...")
        for split in ["train", "validation", "test", "all_data", "val"]:
            json_file = ds_path / f"{split}.json"
            if not json_file.exists(): continue
                
            with open(json_file, "r") as f:
                data = json.load(f)
                
            for item in data:
                try:
                    record = {
                        "id": f"{ds_name}_{len(records)}",
                        "source_dataset": ds_name,
                        "metadata": {}
                    }
                    
                    # Add image existence check
                    def check_images(img_paths):
                        for p in img_paths:
                            if not Path(p).exists(): return False
                        return True
                        
                    if ds_name == "vrsbench":
                        if "vqa" in ds_info.get("tasks", []) and "qa_pairs" in item:
                            for qa in item["qa_pairs"]:
                                if not check_images([item["local_image_path"]]): continue
                                r = record.copy()
                                r["id"] = f"{ds_name}_{len(records)}"
                                r["task"] = "vqa"
                                r["images"] = [item["local_image_path"]]
                                r["question"] = qa.get("question", "")
                                r["answer"] = qa.get("answer", "")
                                records.append(r)
                                
                    elif ds_name == "rsvqa":
                        if not check_images([item["local_image_path"]]): raise FileNotFoundError()
                        record["task"] = "vqa"
                        record["images"] = [item["local_image_path"]]
                        record["question"] = item.get("question", "")
                        record["answer"] = item.get("answer", "")
                        records.append(record)
                        
                    elif ds_name == "cdvqa":
                        if not check_images([item["local_image1"], item["local_image2"]]): raise FileNotFoundError()
                        record["task"] = "change_vqa"
                        record["images"] = [item["local_image1"], item["local_image2"]]
                        
                        conv = item.get("conversations", [])
                        q, a = "", ""
                        for c in conv:
                            if c.get("from") == "user": q = c.get("value", "")
                            elif c.get("from") in ["assistant", "gpt"]: a = c.get("value", "")
                        
                        q = q.replace("Image 1: <image>\nImage 2: <image>\n", "").strip()
                        
                        record["question"] = q
                        record["answer"] = a
                        records.append(record)
                        
                    elif ds_name == "bigearthnet":
                        record["task"] = "caption"
                        record["images"] = []
                        record["question"] = "Describe this satellite image."
                        record["answer"] = item.get("output", "")
                        records.append(record)
                except Exception as e:
                    invalid += 1

        
    random.seed(config.get("dataset", {}).get("random_seed", 42))
    random.shuffle(records)
    
    # Calculate splits
    total = len(records)
    train_split = config.get("dataset", {}).get("train_split", 0.8)
    val_split = config.get("dataset", {}).get("val_split", 0.1)
    
    train_end = int(total * train_split)
    val_end = train_end + int(total * val_split)
    
    train_records = records[:train_end]
    val_records = records[train_end:val_end]
    test_records = records[val_end:]
    
    def save_jsonl(recs, filename):
        path = unified_dir / filename
        with open(path, "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
                
    save_jsonl(train_records, "train.jsonl")
    save_jsonl(val_records, "val.jsonl")
    save_jsonl(test_records, "test.jsonl")
    
    # Generate statistics
    stats = {
        "total_examples": total,
        "examples_by_source": {},
        "examples_by_task": {},
        "examples_by_split": {
            "train": len(train_records),
            "val": len(val_records),
            "test": len(test_records)
        },
        "num_one_image": 0,
        "num_two_image": 0,
        "num_grounding": 0,
        "num_skipped": skipped,
        "num_duplicates": duplicates,
        "num_invalid": invalid,
        "timestamp": datetime.now().isoformat()
    }
    
    with open(unified_dir / "statistics.json", "w") as f:
        json.dump(stats, f, indent=2)
        
    # Manifest
    manifest = {
        "train": "train.jsonl",
        "val": "val.jsonl",
        "test": "test.jsonl",
        "statistics": "statistics.json"
    }
    with open(unified_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
        
    print(f"Data pipeline finished. Total unified records: {total}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="vlm/config.yaml")
    args = parser.parse_args()
    
    # If default relative path used but script is run from inside vlm, fix it
    config_path = Path(args.config)
    if not config_path.exists() and config_path.name == "config.yaml":
        fallback = Path(__file__).parent / "config.yaml"
        if fallback.exists():
            config_path = fallback
            
    cfg = load_config(str(config_path))
    build_unified_dataset(cfg, str(config_path))
