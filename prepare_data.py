import json
import argparse
import random
import yaml
from pathlib import Path
from datetime import datetime
from collections import defaultdict


def load_config(config_path="config.yaml"):
    if not Path(config_path).exists():
        return {}
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _resolve_image(img_rel_path, vlm_dir):
    """Resolve a dataset-relative image path to an absolute path."""
    p = Path(img_rel_path)
    if not p.is_absolute() and p.parts and p.parts[0] == "data":
        p = vlm_dir / p
    return p


def _all_images_exist(img_paths, vlm_dir):
    """Return (True, []) if every image exists, else (False, [missing, ...])."""
    missing = []
    for img in img_paths:
        resolved = _resolve_image(img, vlm_dir)
        if not resolved.exists():
            missing.append(img)
    return (len(missing) == 0), missing


def build_unified_dataset(config, config_path):
    vlm_dir = Path(config_path).parent
    unified_dir = vlm_dir / "data/unified"
    unified_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir = vlm_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    records = []
    skipped_records = []  # Written to dataset_integrity_skipped.jsonl
    invalid = 0

    datasets_to_process = config.get("dataset", {}).get("sources", {})

    for ds_name, ds_info in datasets_to_process.items():
        if not ds_info.get("enabled", False):
            continue

        ds_path = Path(ds_info.get("path", ""))
        if not ds_path.is_absolute() and ds_path.parts and ds_path.parts[0] == "data":
            ds_path = vlm_dir / ds_path

        if not ds_path.exists():
            print(f"Warning: Dataset path {ds_path} does not exist. Skipping {ds_name}.")
            continue

        print(f"Processing {ds_name}...")
        for split in ["train", "validation", "test", "all_data", "val"]:
            json_file = ds_path / f"{split}.json"
            if not json_file.exists():
                continue

            with open(json_file, "r") as f:
                data = json.load(f)

            for item in data:
                try:
                    record = {
                        "id": f"{ds_name}_{len(records) + len(skipped_records)}",
                        "source_dataset": ds_name,
                        "metadata": {},
                    }

                    if ds_name == "vrsbench":
                        if "vqa" in ds_info.get("tasks", []) and "qa_pairs" in item:
                            img_path = item["local_image_path"]
                            ok, missing = _all_images_exist([img_path], vlm_dir)
                            if not ok:
                                skipped_records.append({
                                    "id": record["id"],
                                    "source_dataset": ds_name,
                                    "missing_images": missing,
                                    "reason": "missing physical image",
                                })
                                continue
                            for qa in item["qa_pairs"]:
                                r = record.copy()
                                r["id"] = f"{ds_name}_{len(records) + len(skipped_records)}"
                                r["task"] = "vqa"
                                r["images"] = [img_path]
                                r["question"] = qa.get("question", "")
                                r["answer"] = qa.get("answer", "")
                                records.append(r)

                    elif ds_name == "rsvqa":
                        img_path = item["local_image_path"]
                        ok, missing = _all_images_exist([img_path], vlm_dir)
                        if not ok:
                            skipped_records.append({
                                "id": record["id"],
                                "source_dataset": ds_name,
                                "missing_images": missing,
                                "reason": "missing physical image",
                            })
                            continue
                        record["task"] = "vqa"
                        record["images"] = [img_path]
                        record["question"] = item.get("question", "")
                        record["answer"] = item.get("answer", "")
                        records.append(record)

                    elif ds_name == "cdvqa":
                        img1 = item["local_image1"]
                        img2 = item["local_image2"]
                        ok, missing = _all_images_exist([img1, img2], vlm_dir)
                        if not ok:
                            skipped_records.append({
                                "id": record["id"],
                                "source_dataset": ds_name,
                                "missing_images": missing,
                                "reason": "missing physical image",
                            })
                            continue
                        record["task"] = "change_vqa"
                        record["images"] = [img1, img2]

                        conv = item.get("conversations", [])
                        q, a = "", ""
                        for c in conv:
                            if c.get("from") == "user":
                                q = c.get("value", "")
                            elif c.get("from") in ["assistant", "gpt"]:
                                a = c.get("value", "")

                        q = q.replace("Image 1: <image>\nImage 2: <image>\n", "").strip()

                        record["question"] = q
                        record["answer"] = a
                        records.append(record)

                except Exception as e:
                    invalid += 1

    # Write the skip log
    skip_log_path = outputs_dir / "dataset_integrity_skipped.jsonl"
    with open(skip_log_path, "w") as f:
        for sr in skipped_records:
            f.write(json.dumps(sr) + "\n")
    print(f"Skipped {len(skipped_records)} records due to missing images (log: {skip_log_path})")

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

    # Generate statistics with actual populated counts
    examples_by_source = defaultdict(int)
    examples_by_task = defaultdict(int)
    num_one_image = 0
    num_two_image = 0
    num_grounding = 0
    for r in records:
        examples_by_source[r["source_dataset"]] += 1
        examples_by_task[r["task"]] += 1
        n_imgs = len(r.get("images", []))
        if n_imgs == 1:
            num_one_image += 1
        elif n_imgs == 2:
            num_two_image += 1
        if r["task"] == "grounding":
            num_grounding += 1

    stats = {
        "total_examples": total,
        "examples_by_source": dict(examples_by_source),
        "examples_by_task": dict(examples_by_task),
        "examples_by_split": {
            "train": len(train_records),
            "val": len(val_records),
            "test": len(test_records),
        },
        "num_one_image": num_one_image,
        "num_two_image": num_two_image,
        "num_grounding": num_grounding,
        "num_skipped_missing_images": len(skipped_records),
        "num_invalid_parse_errors": invalid,
        "timestamp": datetime.now().isoformat(),
    }

    with open(unified_dir / "statistics.json", "w") as f:
        json.dump(stats, f, indent=2)

    # Manifest
    manifest = {
        "train": "train.jsonl",
        "val": "val.jsonl",
        "test": "test.jsonl",
        "statistics": "statistics.json",
    }
    with open(unified_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Data pipeline finished. Total unified records: {total}")
    print(f"  By source: {dict(examples_by_source)}")
    print(f"  By task:   {dict(examples_by_task)}")
    print(f"  Splits:    train={len(train_records)}, val={len(val_records)}, test={len(test_records)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    # Resolve config path robustly
    config_path = Path(args.config)
    if not config_path.exists() and config_path.name == "config.yaml":
        fallback = Path(__file__).parent / "config.yaml"
        if fallback.exists():
            config_path = fallback

    cfg = load_config(str(config_path))
    build_unified_dataset(cfg, str(config_path))
