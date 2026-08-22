def build_llava_prompt(question: str, num_images: int, task: str, evidence_text: str = None) -> str:
    """
    Builds a standard LLaVA-1.5 prompt string with the correct number of <image> tokens.
    """
    # Build image tokens
    image_tokens = "<image>\n" * num_images
    
    # Process evidence
    evidence_prompt = ""
    if evidence_text:
        evidence_prompt = f"{evidence_text}\nPlease use the above evidence as supporting information to help answer the question, but do not blindly repeat it if inconsistent with the visual input.\n\n"

    # Task specific instructions
    if task == "change_vqa":
        task_instruction = "You are comparing two satellite images. Image 1 is earlier, Image 2 is later.\n"
    elif task == "caption":
        task_instruction = "Describe this satellite image in detail.\n"
    elif task == "grounding":
        task_instruction = "Identify the bounding box region for the requested object.\n"
    else:
        task_instruction = ""
        
    user_content = f"{image_tokens}{task_instruction}{evidence_prompt}{question}"
    
    return f"USER: {user_content.strip()}\nASSISTANT:"
