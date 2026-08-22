#!/usr/bin/env python3
"""
Colab Data Acquisition Script

Downloads exactly the image subsets required by the existing unified
dataset (data/unified/*.jsonl). Does NOT regenerate the unified JSONL.

Sources:
  - RSVQA:       dmarsili/RSVQA-LR-2k (validation split, first 20)
  - CDVQA:       ljx620/CDVQA (train first 40, validation first 10, test first 10)
  - BigEarthNet: BIFOLD-BigEarthNetv2-0/BigEarthNet.txt (text-only, first 20)

Usage:
  python acquire_data.py            # download all missing assets
  python acquire_data.py --verify   # verify only, no downloads
"""

import io
import json
import os
import sys
from pathlib import Path


def get_vlm_dir():
    return Path(__file__).parent


def download_rsvqa(vlm_dir, val_limit=20):
    """Download RSVQA validation images."""
    from datasets import load_dataset
    from PIL import Image

    out_dir = vlm_dir / "data" / "external" / "rsvqa"
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    # Check what already exists
    needed = []
    for i in range(val_limit):
        fname = f"validation_{i}.jpg"
        if not (img_dir / fname).exists():
            needed.append(i)

    if not needed:
        print(f"  RSVQA: all {val_limit} images already present, skipping download.")
        return True

    print(f"  RSVQA: {len(needed)} of {val_limit} images missing, downloading...")

    try:
        ds = load_dataset("dmarsili/RSVQA-LR-2k", split="validation", streaming=True)
    except Exception as e:
        print(f"  ERROR: Failed to load RSVQA dataset: {e}")
        return False

    records = []
    count = 0
    downloaded = 0
    for item in ds:
        img = item["image"]
        img_id = f"validation_{count}"
        img_path = img_dir / f"{img_id}.jpg"

        if not img_path.exists():
            img.convert("RGB").save(str(img_path))
            downloaded += 1

        # Also rebuild the annotation json for prepare_data.py compatibility
        item_copy = {k: v for k, v in item.items() if k != "image"}
        item_copy["local_image_path"] = f"data/external/rsvqa/images/{img_id}.jpg"
        records.append(item_copy)

        count += 1
        if count >= val_limit:
            break

    # Save annotation json
    with open(out_dir / "validation.json", "w") as f:
        json.dump(records, f)

    print(f"  RSVQA: downloaded {downloaded} images, {count} total records.")
    return True


def download_cdvqa(vlm_dir, train_limit=40, val_limit=10, test_limit=10):
    """Download CDVQA image pairs."""
    from datasets import load_dataset
    from PIL import Image

    out_dir = vlm_dir / "data" / "external" / "cdvqa"
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    splits = [
        ("train", train_limit),
        ("validation", val_limit),
        ("test", test_limit),
    ]

    all_ok = True
    for split_name, limit in splits:
        # Check what already exists by scanning expected filenames
        # CDVQA uses HF __key__ which follows pattern: cdvqa-{split}-{idx:08d}
        needed_count = 0
        for i in range(limit):
            key = f"cdvqa-{split_name}-{i:08d}"
            if split_name == "validation":
                key = f"cdvqa-val-{i:08d}"
            f1 = img_dir / f"{key}_1.jpg"
            f2 = img_dir / f"{key}_2.jpg"
            if not f1.exists() or not f2.exists():
                needed_count += 1

        if needed_count == 0:
            print(f"  CDVQA {split_name}: all {limit} pairs already present, skipping.")
            continue

        print(f"  CDVQA {split_name}: {needed_count} of {limit} pairs missing, downloading...")

        try:
            ds = load_dataset("ljx620/CDVQA", split=split_name, streaming=True)
        except Exception as e:
            print(f"  ERROR: Failed to load CDVQA {split_name}: {e}")
            all_ok = False
            continue

        records = []
        count = 0
        downloaded = 0
        for item in ds:
            img_id = item.get("__key__", f"cdvqa-{split_name}-{count:08d}")
            p1 = img_dir / f"{img_id}_1.jpg"
            p2 = img_dir / f"{img_id}_2.jpg"

            if not p1.exists() or not p2.exists():
                def get_img(i):
                    if isinstance(i, bytes):
                        return Image.open(io.BytesIO(i))
                    if isinstance(i, dict) and "bytes" in i:
                        return Image.open(io.BytesIO(i["bytes"]))
                    return i

                img1 = get_img(item["0.img"])
                img2 = get_img(item["1.img"])
                img1.convert("RGB").save(str(p1))
                img2.convert("RGB").save(str(p2))
                downloaded += 1

            record = {}
            if "json" in item:
                record = item["json"]
            record["local_image1"] = f"data/external/cdvqa/images/{img_id}_1.jpg"
            record["local_image2"] = f"data/external/cdvqa/images/{img_id}_2.jpg"
            records.append(record)

            count += 1
            if count % 20 == 0:
                print(f"    Downloaded {count}...")
            if count >= limit:
                break

        with open(out_dir / f"{split_name}.json", "w") as f:
            json.dump(records, f)

        print(f"  CDVQA {split_name}: downloaded {downloaded} pairs, {count} total records.")

    return all_ok


def download_bigearthnet(vlm_dir, limit=20):
    """Download BigEarthNet text annotations (no images needed)."""
    from datasets import load_dataset

    out_dir = vlm_dir / "data" / "external" / "bigearthnet_txt"
    out_dir.mkdir(parents=True, exist_ok=True)

    json_file = out_dir / "all_data.json"
    if json_file.exists():
        with open(json_file) as f:
            existing = json.load(f)
        if len(existing) >= limit:
            print(f"  BigEarthNet: {len(existing)} records already present, skipping.")
            return True

    print(f"  BigEarthNet: downloading {limit} text records...")

    try:
        ds = load_dataset(
            "BIFOLD-BigEarthNetv2-0/BigEarthNet.txt", split="all_data", streaming=True
        )
    except Exception as e:
        print(f"  ERROR: Failed to load BigEarthNet: {e}")
        return False

    records = []
    count = 0
    for item in ds:
        records.append(item)
        count += 1
        if count >= limit:
            break

    with open(json_file, "w") as f:
        json.dump(records, f)

    print(f"  BigEarthNet: saved {count} text records.")
    return True


def verify_all(vlm_dir):
    """Check that every image referenced in the unified JSONL exists."""
    unified_dir = vlm_dir / "data" / "unified"
    if not unified_dir.exists():
        print("ERROR: data/unified/ directory not found.")
        return False

    total_images = 0
    missing_images = 0
    missing_list = []

    for split in ["train.jsonl", "val.jsonl", "test.jsonl"]:
        fpath = unified_dir / split
        if not fpath.exists():
            print(f"WARNING: {split} not found.")
            continue
        with open(fpath) as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                for img in record.get("images", []):
                    total_images += 1
                    img_path = Path(img)
                    if not img_path.is_absolute() and img_path.parts and img_path.parts[0] == "data":
                        img_path = vlm_dir / img_path
                    if not img_path.exists():
                        missing_images += 1
                        missing_list.append(img)

    print(f"\n  Total image references: {total_images}")
    print(f"  Existing: {total_images - missing_images}")
    print(f"  Missing:  {missing_images}")

    if missing_images > 0:
        print(f"\n  First 10 missing:")
        for m in missing_list[:10]:
            print(f"    {m}")
        return False

    return True


def main():
    vlm_dir = get_vlm_dir()
    verify_only = "--verify" in sys.argv

    print("=" * 60)
    print("SatQuery VLM — Data Acquisition")
    print("=" * 60)
    print(f"Repository root: {vlm_dir}")
    print()

    if verify_only:
        print("[VERIFY ONLY MODE]")
        ok = verify_all(vlm_dir)
        if ok:
            print("\n[OK] All image references satisfied.")
            sys.exit(0)
        else:
            print("\n[FAIL] Missing image references detected.")
            sys.exit(1)

    print("[STEP 1/3] Downloading RSVQA...")
    rsvqa_ok = download_rsvqa(vlm_dir, val_limit=20)

    print("\n[STEP 2/3] Downloading CDVQA...")
    cdvqa_ok = download_cdvqa(vlm_dir, train_limit=40, val_limit=10, test_limit=10)

    print("\n[STEP 3/3] Downloading BigEarthNet text...")
    ben_ok = download_bigearthnet(vlm_dir, limit=20)

    print("\n" + "=" * 60)
    print("[VERIFICATION]")
    verify_ok = verify_all(vlm_dir)

    print("\n" + "=" * 60)
    if rsvqa_ok and cdvqa_ok and ben_ok and verify_ok:
        print("[OK] All datasets acquired and verified.")
        print("     Run 'python verify_data.py' for full integrity check.")
        sys.exit(0)
    else:
        print("[FAIL] Some datasets could not be fully acquired.")
        if not rsvqa_ok:
            print("  - RSVQA download failed")
        if not cdvqa_ok:
            print("  - CDVQA download failed")
        if not ben_ok:
            print("  - BigEarthNet download failed")
        if not verify_ok:
            print("  - Image verification failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
