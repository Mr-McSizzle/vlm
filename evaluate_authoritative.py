import os
import json
import argparse
import time
import hashlib
from datetime import datetime
from pathlib import Path
import subprocess

from inference import vlm_answer
from scoring import score_prediction

def get_git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
    except Exception:
        return "unknown"

def get_package_versions():
    import transformers
    import peft
    import sys
    return {
        "python": sys.version,
        "transformers": transformers.__version__,
        "peft": peft.__version__
    }

def hash_file(filepath):
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        buf = f.read(65536)
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(65536)
    return hasher.hexdigest()

def summarize_dataset():
    base_dir = Path("data/unified")
    counts = {}
    test_tasks = {"vqa": 0, "change_vqa": 0, "caption": 0, "grounding": 0}
    
    for split in ["train", "val", "test"]:
        path = base_dir / f"{split}.jsonl"
        count = 0
        if path.exists():
            with open(path) as f:
                for line in f:
                    count += 1
                    if split == "test":
                        record = json.loads(line)
                        task = record.get("task", "vqa")
                        test_tasks[task] = test_tasks.get(task, 0) + 1
        counts[split] = count
        
    print("=== Dataset Summary ===")
    print(f"Train count: {counts.get('train', 0)}")
    print(f"Val count:   {counts.get('val', 0)}")
    print(f"Test count:  {counts.get('test', 0)}")
    print("\nTest set breakdown:")
    for t, c in test_tasks.items():
        print(f"  {t}: {c}")
    print("=======================\n")
    return counts, test_tasks

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["base", "adapted"])
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--mock", action="store_true", help="Use mock evaluation (for unit tests only). NEVER default.")
    
    args = parser.parse_args()
    
    # Validation
    if args.model == "adapted" and not args.checkpoint:
        raise ValueError("Adapted model requires --checkpoint path.")
        
    if not args.mock:
        print(f"Running REAL AUTHORITATIVE INFERENCE using model: {args.model}")
    else:
        print("WARNING: Running in MOCK mode. This is NOT an authoritative evaluation.")
        
    summarize_dataset()
    
    test_file = Path("data/unified/test.jsonl")
    if not test_file.exists():
        raise FileNotFoundError(f"Missing {test_file}")
        
    results = []
    
    with open(test_file, 'r') as f:
        records = [json.loads(line) for line in f]
        
    for idx, record in enumerate(records):
        print(f"Evaluating {idx+1}/{len(records)}: {record['id']}")
        images = record.get("images", [])
        
        # Image provenance check
        valid_images = True
        for img in images:
            if not Path(img).exists():
                print(f"  Missing image: {img}")
                valid_images = False
                break
                
        if not valid_images:
            results.append({
                "id": record["id"],
                "source_dataset": record.get("source_dataset", ""),
                "task": record.get("task", ""),
                "images": images,
                "question": record["question"],
                "expected_answer": record["answer"],
                "prediction": {},
                "model": args.model,
                "checkpoint": args.checkpoint if args.model == "adapted" else "none",
                "timestamp": datetime.utcnow().isoformat(),
                "success": False,
                "error": "Image file missing",
                "latency_seconds": 0.0
            })
            continue

        task = record.get("task", "vqa")
        
        if args.mock:
            # Fake prediction
            time.sleep(0.01)
            pred = {
                "answer": "Mock answer",
                "task": task,
                "regions": [],
                "confidence": None,
                "model": "satquery-vlm",
                "checkpoint": args.checkpoint,
                "metadata": {"status": "ok", "latency_seconds": 0.01, "mocked": True}
            }
        else:
            try:
                # We do not pass evidence in base evaluation unless explicitly specified in record. 
                # Currently unified dataset does not have external evidence attached.
                pred = vlm_answer(images=images, question=record["question"], task=task)
            except Exception as e:
                pred = {
                    "answer": "",
                    "metadata": {"status": "error", "error_type": type(e).__name__, "error_message": str(e)}
                }

        is_success = pred.get("metadata", {}).get("status") in ["ok", "warning"]
        
        res_record = {
            "id": record["id"],
            "source_dataset": record.get("source_dataset", ""),
            "task": task,
            "images": images, # Preserves chronological order exactly as defined
            "question": record["question"],
            "expected_answer": record["answer"],
            "prediction": pred,
            "model": args.model,
            "checkpoint": args.checkpoint if args.model == "adapted" else "none",
            "timestamp": datetime.utcnow().isoformat(),
            "success": is_success,
            "latency_seconds": pred.get("metadata", {}).get("latency_seconds", 0.0)
        }
        
        # Guard against mocking inside the real inference pipeline
        if not args.mock and pred.get("metadata", {}).get("mocked"):
            raise RuntimeError("CRITICAL ERROR: Real inference pipeline returned a mocked prediction.")
            
        results.append(res_record)
        
    # Write output
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
        
    print(f"Evaluation complete. Saved to {out_path}")
    
    # Reproducibility Manifest
    manifest = {
        "git_commit": get_git_commit(),
        "model": args.model,
        "adapter_path": args.checkpoint,
        "test_set_hash": hash_file(test_file),
        "timestamp": datetime.utcnow().isoformat(),
        "python_env": get_package_versions(),
        "mock_mode": args.mock
    }
    
    manifest_path = Path("outputs/EVALUATION_MANIFEST.json")
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

if __name__ == "__main__":
    main()
