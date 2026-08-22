import os
import json
from pathlib import Path
from datasets import load_dataset

vlm_dir = Path(__file__).parent

def download_vrsbench(train_limit=20, val_limit=10):
    print("VRSBench...")
    out_dir = vlm_dir / "data/external/vrsbench"
    os.makedirs(f"{out_dir}/images", exist_ok=True)
    
    for split in ["train", "validation", "test"]:
        print(f"  {split}...")
        try:
            ds = load_dataset("xiang709/VRSBench", split=split, streaming=True)
        except:
            continue
        records = []
        count = 0
        for item in ds:
            img = item['image']
            img_id = item.get('id', f"{split}_{count}")
            img_path = f"{out_dir}/images/{img_id}.jpg"
            img.convert("RGB").save(img_path)
            
            del item['image']
            # Make path relative to vlm directory for dataset manifest
            item['local_image_path'] = f"data/external/vrsbench/images/{img_id}.jpg"
            records.append(item)
            
            count += 1
            if count % 20 == 0: print(f"  Downloaded {count}...")
            if split == "train" and count >= train_limit: break
            elif split != "train" and count >= val_limit: break
            
        with open(f"{out_dir}/{split}.json", "w") as f:
            json.dump(records, f)

def download_cdvqa(train_limit=100, val_limit=10):
    print("CDVQA...")
    out_dir = vlm_dir / "data/external/cdvqa"
    os.makedirs(f"{out_dir}/images", exist_ok=True)
    
    for split in ["train", "validation", "test"]:
        print(f"  {split}...")
        try:
            ds = load_dataset("ljx620/CDVQA", split=split, streaming=True)
        except:
            continue
        records = []
        count = 0
        for item in ds:
            import io
            from PIL import Image
            
            def get_img(i):
                if isinstance(i, bytes): return Image.open(io.BytesIO(i))
                if isinstance(i, dict) and 'bytes' in i: return Image.open(io.BytesIO(i['bytes']))
                return i
                
            img1 = get_img(item['0.img'])
            img2 = get_img(item['1.img'])
            img_id = item.get('__key__', f"{split}_{count}")
            p1 = f"{out_dir}/images/{img_id}_1.jpg"
            p2 = f"{out_dir}/images/{img_id}_2.jpg"
            
            img1.convert("RGB").save(p1)
            img2.convert("RGB").save(p2)
            
            record = {}
            if 'json' in item:
                record = item['json']
            
            record['local_image1'] = f"data/external/cdvqa/images/{img_id}_1.jpg"
            record['local_image2'] = f"data/external/cdvqa/images/{img_id}_2.jpg"
            records.append(record)
            
            count += 1
            if count % 20 == 0: print(f"  Downloaded {count}...")
            if split == "train" and count >= train_limit: break
            elif split != "train" and count >= val_limit: break
            
        with open(f"{out_dir}/{split}.json", "w") as f:
            json.dump(records, f)

def download_rsvqa(val_limit=100):
    print("RSVQA...")
    out_dir = vlm_dir / "data/external/rsvqa"
    os.makedirs(f"{out_dir}/images", exist_ok=True)
    
    for split in ["validation"]:
        print(f"  {split}...")
        try:
            ds = load_dataset("dmarsili/RSVQA-LR-2k", split=split, streaming=True)
        except:
            continue
        records = []
        count = 0
        for item in ds:
            img = item['image']
            img_id = f"{split}_{count}"
            img_path = f"{out_dir}/images/{img_id}.jpg"
            img.convert("RGB").save(img_path)
            
            del item['image']
            item['local_image_path'] = f"data/external/rsvqa/images/{img_id}.jpg"
            records.append(item)
            
            count += 1
            if count % 20 == 0: print(f"  Downloaded {count}...")
            if count >= val_limit: break
            
        with open(f"{out_dir}/{split}.json", "w") as f:
            json.dump(records, f)

def download_bigearthnet(limit=100):
    print("BigEarthNet...")
    out_dir = vlm_dir / "data/external/bigearthnet_txt"
    os.makedirs(out_dir, exist_ok=True)
    
    for split in ["all_data"]:
        print(f"  {split}...")
        try:
            ds = load_dataset("BIFOLD-BigEarthNetv2-0/BigEarthNet.txt", split=split, streaming=True)
        except:
            continue
        records = []
        count = 0
        for item in ds:
            records.append(item)
            count += 1
            if count % 20 == 0: print(f"  Downloaded {count}...")
            if count >= limit: break
            
        with open(f"{out_dir}/{split}.json", "w") as f:
            json.dump(records, f)

if __name__ == "__main__":
    download_cdvqa(train_limit=40, val_limit=10)
    download_rsvqa(val_limit=20)
    download_bigearthnet(limit=20)
    # download_vrsbench(train_limit=20, val_limit=10)
