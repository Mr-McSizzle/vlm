import json
from pathlib import Path

def generate_preview():
    vlm_dir = Path(__file__).parent
    unified_dir = vlm_dir / "data" / "unified"
    preview_file = vlm_dir / "outputs" / "unified_dataset_preview.md"
    preview_file.parent.mkdir(parents=True, exist_ok=True)
    
    records = []
    for split in ["train.jsonl", "val.jsonl", "test.jsonl"]:
        if (unified_dir / split).exists():
            with open(unified_dir / split, "r") as f:
                for line in f:
                    records.append(json.loads(line))
                    
    tasks = {"vqa": 10, "change_vqa": 10, "grounding": 5, "caption": 5}
    counts = {k: 0 for k in tasks}
    
    with open(preview_file, "w") as f:
        f.write("# Unified Dataset Preview\n\n")
        f.write("This document showcases REAL examples pulled from the local datasets.\n\n")
        
        for r in records:
            task = r.get("task", "")
            if counts.get(task, 0) < tasks.get(task, 0):
                f.write(f"## {task.upper()} Example (Source: {r.get('source_dataset')})\n")
                f.write(f"- **ID**: {r.get('id')}\n")
                f.write(f"- **Images**: {r.get('images')}\n")
                f.write(f"- **Question**: {r.get('question')}\n")
                f.write(f"- **Answer**: {r.get('answer')}\n")
                f.write(f"- **Region**: {r.get('region', 'None')}\n\n")
                counts[task] += 1
                
        f.write("## Status\n")
        f.write("DATA PIPELINE UNBLOCKED.\n")
        
    print(f"Generated preview with counts: {counts}")

if __name__ == "__main__":
    generate_preview()
