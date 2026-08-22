import argparse
import json
from inference import vlm_answer
from dataset import VLMDataset

def evaluate(args):
    print(f"Evaluating VLM on dataset: {args.dataset}")
    dataset = VLMDataset(args.dataset)
    
    correct = 0
    total = len(dataset)
    
    for i in range(total):
        item = dataset[i]
        result = vlm_answer(
            images=item["image_paths"], 
            question=item["prompt"]
        )
        
        pred = result["answer"]
        target = item["target"]
        
        # Simple dummy metric (exact match on lowered string)
        if target.lower() in pred.lower():
            correct += 1
            
    accuracy = correct / total if total > 0 else 0
    print(f"Evaluation finished. Dummy Accuracy: {accuracy:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Path to evaluation JSONL dataset")
    
    args = parser.parse_args()
    evaluate(args)
