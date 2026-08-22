import argparse
import time
import json
import torch
from datetime import datetime
from pathlib import Path
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
from PIL import Image
from vlm.prompts import get_prompt

def format_prompt(task_type, question_text, num_images):
    base_prompt = get_prompt(task_type, question=question_text)
    image_tags = "\n".join(["<image>"] * num_images)
    return f"USER: {image_tags}\n{base_prompt}\nASSISTANT:"

def main():
    parser = argparse.ArgumentParser(description="Run Baseline VLM Evaluation")
    parser.add_argument("--image", type=str, required=True, help="Path to first image")
    parser.add_argument("--image2", type=str, default=None, help="Path to second image (optional)")
    args = parser.parse_args()

    vlm_dir = Path(__file__).parent
    # Create data dir
    data_dir = vlm_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    questions_file = data_dir / "baseline_questions.json"
    results_file = data_dir / "baseline_results.json"
    summary_file = data_dir / "baseline_summary.json"

    with open(questions_file, 'r') as f:
        questions = json.load(f)

    print("="*60)
    print("SATQUERY AI — BASELINE VLM EVALUATION")
    print("="*60)
    
    model_id = "llava-hf/llava-1.5-7b-hf"
    print(f"\nModel:\n{model_id}")
    print("\nAdapter:\nNONE (BASE MODEL)")
    print(f"\nImage 1:\n{args.image}")
    print(f"\nImage 2:\n{args.image2 if args.image2 else 'NONE'}\n")

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    
    processor = AutoProcessor.from_pretrained(model_id)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id, 
        quantization_config=quantization_config, 
        device_map={"": 0}
    )

    image1 = Image.open(args.image).convert("RGB")
    image2 = Image.open(args.image2).convert("RGB") if args.image2 else None

    results = []
    category_counts = {}
    task_counts = {}
    
    successful = 0
    failed = 0
    skipped = 0
    inference_times = []
    
    print("-" * 60)
    print(f"{'ID':<15} {'TASK':<15} {'STATUS':<12} {'TIME'}")
    print("-" * 60)
    
    start_total_time = time.time()
    
    for q in questions:
        q_id = q["id"]
        cat = q["category"]
        task = q["task"]
        expected_images = q["expected_images"]
        text = q["question"]
        
        category_counts[cat] = category_counts.get(cat, 0) + 1
        task_counts[task] = task_counts.get(task, 0) + 1
        
        base_result = {
            "question_id": q_id,
            "category": cat,
            "task": task,
            "question": text,
            "image_paths": [args.image] + ([args.image2] if args.image2 else []),
            "model": model_id,
            "adapter": None,
            "raw_answer": None,
            "parsed_answer": None,
            "regions": [],
            "grounding_parse_status": None,
            "success": False,
            "skipped": False,
            "error": None,
            "inference_time_seconds": None
        }

        if expected_images == 2 and not image2:
            base_result["skipped"] = True
            base_result["skip_reason"] = "Question requires two images but --image2 was not supplied."
            results.append(base_result)
            skipped += 1
            print(f"{q_id:<15} {task:<15} {'SKIPPED':<12} -")
            continue
            
        if task == "grounding":
            base_result["grounding_parse_status"] = "not_available"

        images_to_pass = [image1]
        if expected_images == 2:
            images_to_pass.append(image2)
            
        prompt = format_prompt(task, text, expected_images)
        
        try:
            inputs = processor(text=prompt, images=images_to_pass, return_tensors="pt").to("cuda")
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)

            start_infer = time.time()
            with torch.inference_mode():
                output = model.generate(**inputs, max_new_tokens=128, do_sample=False)
            infer_time = time.time() - start_infer
            
            generated_text = processor.decode(output[0], skip_special_tokens=True)
            answer = generated_text.split("ASSISTANT:")[-1].strip()
            
            base_result["raw_answer"] = answer
            base_result["parsed_answer"] = answer
            base_result["success"] = True
            base_result["inference_time_seconds"] = round(infer_time, 2)
            
            successful += 1
            inference_times.append(infer_time)
            print(f"{q_id:<15} {task:<15} {'SUCCESS':<12} {infer_time:.1f}s")
            
        except Exception as e:
            base_result["error"] = str(e)
            failed += 1
            print(f"{q_id:<15} {task:<15} {'FAILED':<12} -")
            
        results.append(base_result)
        
        # Save intermediate
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
            
        torch.cuda.empty_cache()

    total_time = time.time() - start_total_time
    avg_time = sum(inference_times) / len(inference_times) if inference_times else 0
    min_time = min(inference_times) if inference_times else 0
    max_time = max(inference_times) if inference_times else 0
    
    summary = {
        "model": model_id,
        "adapter": None,
        "total_questions_defined": len(questions),
        "questions_attempted": successful + failed,
        "questions_succeeded": successful,
        "questions_failed": failed,
        "questions_skipped": skipped,
        "average_inference_time_seconds": round(avg_time, 2),
        "min_inference_time_seconds": round(min_time, 2),
        "max_inference_time_seconds": round(max_time, 2),
        "total_evaluation_time_seconds": round(total_time, 2),
        "category_counts": category_counts,
        "task_counts": task_counts,
        "results_file": str(results_file),
        "timestamp": datetime.now().isoformat()
    }
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print("-" * 60)
    print("Total:")
    print(f"Attempted: {successful + failed}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Skipped: {skipped}")
    print(f"\nAverage inference time: {avg_time:.2f} sec")
    
    print("\n" + "-" * 60)
    print("SAMPLE OUTPUTS")
    print("-" * 60)
    
    sample_count = 0
    for r in results:
        if r["success"]:
            print(f"\n{r['question_id']}:")
            print(r["raw_answer"])
            sample_count += 1
            if sample_count >= 5:
                break
                
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
