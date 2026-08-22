from vlm.prompts import get_prompt, VQA_PROMPT, EVIDENCE_PROMPT_ADDITION

def test_get_prompt_vqa():
    p = get_prompt("vqa", question="Is this a road?")
    assert "Is this a road?" in p
    assert "You are an expert" in p

def test_get_prompt_with_evidence():
    p = get_prompt("vqa", question="Is this a road?", evidence="Map data shows Highway 1")
    assert "Map data shows Highway 1" in p
    assert "Is this a road?" in p

def test_get_prompt_captioning():
    p = get_prompt("captioning")
    assert "Describe the provided image" in p
