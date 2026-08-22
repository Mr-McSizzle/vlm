def build_llava_prompt(question: str, num_images: int, task: str, evidence: dict = None) -> str:
    """
    Builds a standard LLaVA-1.5 prompt string with the correct number of <image> tokens.
    """
    # Build image tokens
    image_tokens = "<image>\n" * num_images
    
    # Process evidence
    evidence_text = ""
    if evidence:
        if isinstance(evidence, dict):
            evidence_str = ", ".join([f"{k}: {v}" for k, v in evidence.items()])
        else:
            evidence_str = str(evidence)
        evidence_text = f"Context: {evidence_str}\n"

    # Task specific instructions (optional, but requested to centralize prompt handling)
    if task == "change_vqa":
        task_instruction = "You are comparing two satellite images. Image 1 is earlier, Image 2 is later.\n"
    elif task == "caption":
        task_instruction = "Describe this satellite image in detail.\n"
    elif task == "grounding":
        task_instruction = "Identify the bounding box region for the requested object.\n"
    else:
        task_instruction = ""
        
    user_content = f"{image_tokens}{task_instruction}{evidence_text}{question}"
    
    return f"USER: {user_content.strip()}\nASSISTANT:"
