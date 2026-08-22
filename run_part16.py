import json
import traceback
from inference import vlm_answer

def run_real_tests():
    print("--- 1. Real VQA ---")
    img1 = "data/external/cdvqa/images/cdvqa-train-00000008_1.jpg"
    img2 = "data/external/cdvqa/images/cdvqa-train-00000008_2.jpg"
    
    try:
        res1 = vlm_answer([img1], "What is this?", task="vqa")
        print(json.dumps(res1, indent=2))
    except Exception as e:
        print(f"VQA Error: {e}")

    print("\n--- 2. Real Change-VQA ---")
    try:
        res2 = vlm_answer([img1, img2], "What changed?", task="change_vqa")
        print(json.dumps(res2, indent=2))
    except Exception as e:
        print(f"Change-VQA Error: {e}")

    print("\n--- 3. Real Evidence-aware Change-VQA ---")
    evidence = {
        "type": "change_detection",
        "changed_fraction": 0.35,
        "dominant_region": "north"
    }
    try:
        res3 = vlm_answer([img1, img2], "What changed?", task="change_vqa", evidence=evidence)
        print(json.dumps(res3, indent=2))
    except Exception as e:
        print(f"Evidence Change-VQA Error: {e}")

if __name__ == "__main__":
    run_real_tests()
