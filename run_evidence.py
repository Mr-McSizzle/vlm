import json
import traceback
from inference import vlm_answer

def run_real_evidence_test():
    evidence = {
        "type": "change_detection",
        "changed_fraction": 0.21,
        "dominant_region": "south"
    }

    img1 = "data/external/cdvqa/images/cdvqa-train-00000008_1.jpg"
    img2 = "data/external/cdvqa/images/cdvqa-train-00000008_2.jpg"

    print("--- Running Real Change-VQA with Evidence ---")
    try:
        res = vlm_answer([img1, img2], "What changed?", task="change_vqa", evidence=evidence)
        print(json.dumps(res, indent=2))
        
        if res["metadata"]["status"] == "error":
            print(f"Error intercepted successfully in metadata: {res['metadata']['error_message']}")
    except Exception as e:
        print(f"Caught underlying exception: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    run_real_evidence_test()
