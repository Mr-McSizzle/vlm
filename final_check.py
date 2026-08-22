import json
from inference import vlm_answer

def run_checks():
    # 1. Load one VQA example
    vqa_img = "data/external/rsvqa/images/validation_13.jpg"
    print("\n--- Running VQA Inference ---")
    vqa_res = vlm_answer(
        images=[vqa_img],
        question="What is the number of water areas on the left of a commercial building?",
        task="vqa"
    )
    print("VQA Result:")
    print(json.dumps(vqa_res, indent=2))
    
    # 2. Load one Change-VQA example
    cdvqa_img1 = "data/external/cdvqa/images/cdvqa-train-00000008_1.jpg"
    cdvqa_img2 = "data/external/cdvqa/images/cdvqa-train-00000008_2.jpg"
    print("\n--- Running Change-VQA Inference ---")
    cdvqa_res = vlm_answer(
        images=[cdvqa_img1, cdvqa_img2],
        question="Have the areas of low vegetation changed in the first image?",
        task="change_vqa"
    )
    print("Change-VQA Result:")
    print(json.dumps(cdvqa_res, indent=2))
    
    # Check if loaded correctly
    if vqa_res["metadata"]["status"] != "ok" or cdvqa_res["metadata"]["status"] != "ok":
        print("\n[FAIL] Inference returned error status.")
        return False
        
    return True

if __name__ == "__main__":
    success = run_checks()
    if not success:
        import sys
        sys.exit(1)
