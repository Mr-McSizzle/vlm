VQA_PROMPT = """You are analyzing a satellite image.
Answer the user's question using only evidence visible in the image.
Question: {question}"""

CHANGE_VQA_PROMPT = """You are comparing two satellite images of the same area taken at
different times.
Image 1 is earlier and Image 2 is later.
Answer using only visual evidence available in the images.
Question: {question}"""

GROUNDING_PROMPT = """You are analyzing a satellite image.
Identify the location described by the user.
Provide a concise spatial description using image-relative terms
such as north, south, east, west, center, upper, lower, left, right.
Do not invent coordinates.
Question: {question}"""

CAPTIONING_PROMPT = """You are analyzing a satellite image. Describe it in detail.
Description:"""

EVIDENCE_PROMPT_ADDITION = """
Use the following additional evidence to help answer:
{evidence}
"""

def get_prompt(task_type: str, question: str = "", evidence: str = None) -> str:
    prompts = {
        "vqa": VQA_PROMPT,
        "captioning": CAPTIONING_PROMPT,
        "change_vqa": CHANGE_VQA_PROMPT,
        "grounding": GROUNDING_PROMPT,
    }
    
    base = prompts.get(task_type, VQA_PROMPT)
    if "{question}" in base:
        base = base.format(question=question)
        
    if evidence:
        base = EVIDENCE_PROMPT_ADDITION.format(evidence=evidence) + "\n" + base
        
    return base
