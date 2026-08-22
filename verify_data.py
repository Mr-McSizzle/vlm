"""
Dataset integrity verifier.

Scans every record in the unified train/val/test JSONL files and checks
that every referenced image path physically exists on disk.

Exits with code 0 ONLY if there are zero missing image references.
Exits with code 1 if any image reference is broken.

Also prints per-split statistics and task distribution.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict


def verify():
    vlm_dir = Path(__file__).parent
    unified_dir = vlm_dir / "data" / "unified"
    outputs_dir = vlm_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    if not unified_dir.exists():
        print("ERROR: Unified dataset directory not found.")
        sys.exit(1)

    all_ok = True
    total_records = 0
    total_valid = 0
    total_missing = 0
    task_counts = defaultdict(int)
    source_counts = defaultdict(int)
    missing_details = []

    for split in ["train.jsonl", "val.jsonl", "test.jsonl"]:
        fpath = unified_dir / split
        if not fpath.exists():
            print(f"WARNING: {split} not found")
            continue

        split_records = 0
        split_valid = 0
        split_missing = 0

        with open(fpath) as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                split_records += 1
                total_records += 1
                task_counts[record.get("task", "unknown")] += 1
                source_counts[record.get("source_dataset", "unknown")] += 1

                record_ok = True
                for img in record.get("images", []):
                    img_path = Path(img)
                    if not img_path.is_absolute() and img_path.parts and img_path.parts[0] == "data":
                        img_path = vlm_dir / img_path
                    if img_path.exists():
                        split_valid += 1
                        total_valid += 1
                    else:
                        split_missing += 1
                        total_missing += 1
                        record_ok = False
                        missing_details.append({
                            "split": split,
                            "record_id": record.get("id"),
                            "missing_image": img,
                        })

                if not record_ok:
                    all_ok = False

        split_name = split.replace(".jsonl", "").upper()
        print(f"\n{split_name}:")
        print(f"  records = {split_records}")
        print(f"  valid image references = {split_valid}")
        print(f"  missing image references = {split_missing}")

    print(f"\nTOTAL:")
    print(f"  records = {total_records}")
    print(f"  valid image references = {total_valid}")
    print(f"  missing image references = {total_missing}")

    print(f"\nTASK DISTRIBUTION:")
    for task in ["vqa", "change_vqa", "caption", "grounding"]:
        print(f"  {task.upper()} = {task_counts.get(task, 0)}")

    print(f"\nSOURCE DISTRIBUTION:")
    for src, count in sorted(source_counts.items()):
        print(f"  {src} = {count}")

    if missing_details:
        print(f"\nMISSING IMAGE DETAILS ({len(missing_details)} total):")
        for d in missing_details[:20]:
            print(f"  [{d['split']}] {d['record_id']}: {d['missing_image']}")
        if len(missing_details) > 20:
            print(f"  ... and {len(missing_details) - 20} more")

    # Generate preview markdown
    preview_file = outputs_dir / "unified_dataset_preview.md"
    all_records = []
    for split in ["train.jsonl", "val.jsonl", "test.jsonl"]:
        fpath = unified_dir / split
        if not fpath.exists():
            continue
        with open(fpath) as f:
            for line in f:
                if line.strip():
                    all_records.append(json.loads(line))

    tasks_to_preview = {"vqa": 10, "change_vqa": 10, "grounding": 5, "caption": 5}
    preview_counts = {k: 0 for k in tasks_to_preview}

    with open(preview_file, "w") as f:
        f.write("# Unified Dataset Preview\n\n")
        f.write("This document showcases REAL examples pulled from the local datasets.\n\n")
        f.write(f"**Total records:** {total_records}\n")
        f.write(f"**Missing image references:** {total_missing}\n\n")

        for task in ["vqa", "change_vqa", "caption", "grounding"]:
            f.write(f"**{task.upper()}:** {task_counts.get(task, 0)}\n")
        f.write("\n---\n\n")

        for r in all_records:
            task = r.get("task", "")
            if preview_counts.get(task, 0) < tasks_to_preview.get(task, 0):
                f.write(f"## {task.upper()} Example (Source: {r.get('source_dataset')})\n")
                f.write(f"- **ID**: {r.get('id')}\n")
                f.write(f"- **Images**: {r.get('images')}\n")
                f.write(f"- **Question**: {r.get('question')}\n")
                f.write(f"- **Answer**: {r.get('answer')}\n")
                f.write(f"- **Region**: {r.get('region', 'None')}\n\n")
                preview_counts[task] += 1

        f.write("## Status\n")
        if all_ok:
            f.write("DATASET INTEGRITY VERIFIED. All image references exist.\n")
        else:
            f.write(f"DATASET INTEGRITY FAILED. {total_missing} missing image references.\n")

    print(f"\nPreview written to: {preview_file}")

    if all_ok:
        print("\n[OK] DATASET INTEGRITY VERIFIED - zero missing image references.")
        sys.exit(0)
    else:
        print(f"\n[FAIL] DATASET INTEGRITY FAILED - {total_missing} missing image references.")
        sys.exit(1)


if __name__ == "__main__":
    verify()
