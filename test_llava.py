import argparse
import time
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
from PIL import Image

def main():
    parser = argparse.ArgumentParser(description="Test LLaVA-1.5 7B")
    parser.add_argument("--image", type=str, required=True, help="Path to the image")
    parser.add_argument("--question", type=str, required=True, help="Question to ask")
    args = parser.parse_args()

    start_load = time.time()
    
    model_id = "llava-hf/llava-1.5-7b-hf"
    print(f"Loading processor: {model_id}...")
    processor = AutoProcessor.from_pretrained(model_id)
    
    print(f"Loading model: {model_id} in 4-bit mode directly to GPU...")
    # 4-bit config for 4GB VRAM (Forced to GPU to avoid CPU offload meta tensor bug)
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id, 
        quantization_config=quantization_config, 
        device_map={"": 0}
    )

    load_time = time.time() - start_load
    print(f"Model loaded in {load_time:.2f} seconds.")

    print(f"Opening image: {args.image}")
    image = Image.open(args.image).convert("RGB")

    prompt = f"USER: <image>\n{args.question}\nASSISTANT:"

    print("Processing inputs...")
    inputs = processor(text=prompt, images=image, return_tensors="pt").to("cuda")
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)

    print("Running inference...")
    start_infer = time.time()
    
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=200)
    
    infer_time = time.time() - start_infer

    print("Decoding generated tokens...")
    generated_text = processor.decode(output[0], skip_special_tokens=True)
    
    # Extract just the answer part after ASSISTANT:
    answer = generated_text.split("ASSISTANT:")[-1].strip()

    print("\n" + "="*50)
    print("QUESTION: ", args.question)
    print("ANSWER:   ", answer)
    print("="*50)

    # Hardware stats
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
    print(f"\nGPU Name: {gpu_name}")
    if torch.cuda.is_available():
        memory_props = torch.cuda.get_device_properties(0)
        print(f"VRAM Capacity: {memory_props.total_memory / (1024**3):.2f} GB")
        print(f"Max VRAM Allocated: {torch.cuda.max_memory_allocated(0) / (1024**3):.2f} GB")
    print(f"Inference Time: {infer_time:.2f} seconds\n")

if __name__ == "__main__":
    main()
