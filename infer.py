import argparse
from inference import vlm_answer

def main():
    parser = argparse.ArgumentParser(description="Standalone inference test for SatQuery AI VLM")
    parser.add_argument("--image", type=str, required=True, help="Path to the image file")
    parser.add_argument("--question", type=str, required=True, help="Question to ask the VLM")
    parser.add_argument("--evidence", type=str, help="Optional text evidence", default=None)
    
    args = parser.parse_args()
    
    print(f"Processing image: {args.image}")
    print(f"Question: {args.question}")
    
    result = vlm_answer(images=[args.image], question=args.question, evidence=args.evidence)
    
    print("\n--- Output ---")
    import json
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
