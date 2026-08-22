import re

def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = str(text).lower().strip()
    text = re.sub(r'[^\w\s\d]', '', text)
    return text

def extract_number(text: str):
    matches = re.findall(r'\b\d+\b', str(text))
    if matches:
        return int(matches[0])
    return None

def determine_failure_category(expected: str, predicted: str, task: str) -> str:
    expected_norm = normalize_text(expected)
    pred_norm = normalize_text(predicted)
    
    if expected_norm in ['yes', 'no']:
        return "yes/no error"
    
    expected_num = extract_number(expected)
    if expected_num is not None:
        return "numeric counting error"
    
    if task == "change_vqa":
        return "temporal reasoning error"
        
    return "manual_review"

def score_prediction(expected: str, predicted: str, task: str) -> dict:
    """
    Returns a dict with:
    - correct: True, False, or None (if unscorable)
    - expected_normalized: str
    - predicted_normalized: str
    - failure_category: str or None
    """
    if not predicted or not expected:
        return {
            "correct": False,
            "expected_normalized": expected,
            "predicted_normalized": predicted,
            "failure_category": "empty prediction/target"
        }
        
    expected_norm = normalize_text(expected)
    pred_norm = normalize_text(predicted)
    
    # 1. Yes/No scoring
    if expected_norm in ['yes', 'no']:
        is_correct = (expected_norm == pred_norm)
        cat = None if is_correct else "yes/no error"
        return {"correct": is_correct, "expected_normalized": expected_norm, "predicted_normalized": pred_norm, "failure_category": cat}
        
    # 2. Numeric scoring
    expected_num = extract_number(expected)
    if expected_num is not None:
        pred_num = extract_number(predicted)
        is_correct = (expected_num == pred_num)
        cat = None if is_correct else "numeric counting error"
        return {"correct": is_correct, "expected_normalized": str(expected_num), "predicted_normalized": str(pred_num) if pred_num is not None else "", "failure_category": cat}

    # 3. Short categorical (if length is small, we assume exact match applies)
    if len(expected.split()) <= 3:
        is_correct = (expected_norm == pred_norm)
        cat = None if is_correct else determine_failure_category(expected, predicted, task)
        return {"correct": is_correct, "expected_normalized": expected_norm, "predicted_normalized": pred_norm, "failure_category": cat}
        
    # 4. Free-form / Caption (unscorable automatically by exact match)
    return {
        "correct": None,
        "expected_normalized": expected_norm,
        "predicted_normalized": pred_norm,
        "failure_category": "manual_review"
    }
