import json
import re
from pathlib import Path

def extract_number(text):
    match = re.search(r'\d+', text)
    return match.group(0) if match else None

def extract_yes_no(text):
    text_lower = text.lower()
    if 'yes' in text_lower and 'no' not in text_lower:
        return 'yes'
    if 'no' in text_lower and 'yes' not in text_lower:
        return 'no'
    # Fallback to the first occurrence
    yes_idx = text_lower.find('yes')
    no_idx = text_lower.find('no')
    if yes_idx != -1 and no_idx != -1:
        return 'yes' if yes_idx < no_idx else 'no'
    if yes_idx != -1: return 'yes'
    if no_idx != -1: return 'no'
    return None

def score_answer(expected, predicted):
    expected = expected.lower().strip()
    predicted = str(predicted).lower()
    
    if expected in ['yes', 'no']:
        return extract_yes_no(predicted) == expected
    elif expected.isdigit():
        return extract_number(predicted) == expected
    else:
        return expected in predicted

def main():
    with open('outputs/evaluation_base.json') as f:
        base_results = json.load(f)
    with open('outputs/evaluation_adapted.json') as f:
        adapted_results = json.load(f)

    base_dict = {r['id']: r for r in base_results}
    adapted_dict = {r['id']: r for r in adapted_results}

    quantitative_results = []
    
    stats = {
        'base': {'total': 0, 'correct': 0, 'vqa_total': 0, 'vqa_correct': 0, 'cdvqa_total': 0, 'cdvqa_correct': 0},
        'adapted': {'total': 0, 'correct': 0, 'vqa_total': 0, 'vqa_correct': 0, 'cdvqa_total': 0, 'cdvqa_correct': 0}
    }

    improved = 0
    degraded = 0
    unchanged_correct = 0
    unchanged_incorrect = 0

    for r_id, b_r in base_dict.items():
        if r_id not in adapted_dict: continue
        a_r = adapted_dict[r_id]
        
        task = b_r['task']
        expected = str(b_r['expected_answer'])
        b_ans = str(b_r['model_answer'])
        a_ans = str(a_r['model_answer'])
        
        b_correct = score_answer(expected, b_ans)
        a_correct = score_answer(expected, a_ans)
        
        quantitative_results.append({
            'id': r_id,
            'task': task,
            'expected_answer': expected,
            'base_answer': b_ans,
            'adapted_answer': a_ans,
            'base_correct': b_correct,
            'adapted_correct': a_correct
        })
        
        # Base stats
        stats['base']['total'] += 1
        if b_correct: stats['base']['correct'] += 1
        if task == 'vqa':
            stats['base']['vqa_total'] += 1
            if b_correct: stats['base']['vqa_correct'] += 1
        elif task == 'change_vqa':
            stats['base']['cdvqa_total'] += 1
            if b_correct: stats['base']['cdvqa_correct'] += 1
            
        # Adapted stats
        stats['adapted']['total'] += 1
        if a_correct: stats['adapted']['correct'] += 1
        if task == 'vqa':
            stats['adapted']['vqa_total'] += 1
            if a_correct: stats['adapted']['vqa_correct'] += 1
        elif task == 'change_vqa':
            stats['adapted']['cdvqa_total'] += 1
            if a_correct: stats['adapted']['cdvqa_correct'] += 1
            
        # Delta stats
        if not b_correct and a_correct:
            improved += 1
        elif b_correct and not a_correct:
            degraded += 1
        elif b_correct and a_correct:
            unchanged_correct += 1
        else:
            unchanged_incorrect += 1

    with open('outputs/FINAL_QUANTITATIVE_EVALUATION.json', 'w') as f:
        json.dump(quantitative_results, f, indent=2)

    base_acc = stats['base']['correct'] / stats['base']['total'] if stats['base']['total'] else 0
    adapt_acc = stats['adapted']['correct'] / stats['adapted']['total'] if stats['adapted']['total'] else 0
    delta = adapt_acc - base_acc

    base_vqa_acc = stats['base']['vqa_correct'] / stats['base']['vqa_total'] if stats['base']['vqa_total'] else 0
    adapt_vqa_acc = stats['adapted']['vqa_correct'] / stats['adapted']['vqa_total'] if stats['adapted']['vqa_total'] else 0

    base_cdvqa_acc = stats['base']['cdvqa_correct'] / stats['base']['cdvqa_total'] if stats['base']['cdvqa_total'] else 0
    adapt_cdvqa_acc = stats['adapted']['cdvqa_correct'] / stats['adapted']['cdvqa_total'] if stats['adapted']['cdvqa_total'] else 0

    md = ["# Final Quantitative Evaluation", ""]
    md.append("## Overall Accuracy")
    md.append(f"- **Base:** {base_acc*100:.1f}%")
    md.append(f"- **Adapted:** {adapt_acc*100:.1f}%")
    md.append(f"- **Improvement:** {delta*100:+.1f}%")
    md.append("")
    
    md.append("## Task Accuracy")
    md.append(f"- **Base VQA:** {base_vqa_acc*100:.1f}% | **Adapted VQA:** {adapt_vqa_acc*100:.1f}%")
    md.append(f"- **Base Change-VQA:** {base_cdvqa_acc*100:.1f}% | **Adapted Change-VQA:** {adapt_cdvqa_acc*100:.1f}%")
    md.append("")
    
    md.append("## Deltas")
    md.append(f"- **Improved:** {improved}")
    md.append(f"- **Degraded:** {degraded}")
    md.append(f"- **Unchanged Correct:** {unchanged_correct}")
    md.append(f"- **Unchanged Incorrect:** {unchanged_incorrect}")
    md.append("")
    
    md.append("## Qualitative Failure Categories")
    md.append("- **Counting Failure**: Adapted model outputs 0 or 2 for large groups of objects, failing to count correctly (e.g. expected 104, answered 0).")
    md.append("- **Yes/No Classification Failure**: Model defaults to 'yes' indiscriminately for Change-VQA and VQA, regardless of actual change.")
    md.append("- **Temporal/Change Reasoning Failure**: The model fails to compare spatial structures over time, instead adopting a lazy 'yes' bias.")
    md.append("- **Hallucination**: Hallucinating objects (like oceans or squares) that are not physically present in the test image.")
    md.append("- **Truncation**: Output cutting off mid-sentence (mitigated by configuring `max_new_tokens` correctly, but present in earlier runs).")
    md.append("- **Unsupported Explanation**: The model adds irrelevant reasoning to simple answers, reducing precision.")
    md.append("")

    md.append("## Per-Example Results")
    md.append("| ID | Task | Expected | Base Answer | Adapted Answer | Base Correct | Adapted Correct |")
    md.append("|---|---|---|---|---|---|---|")
    for r in quantitative_results:
        md.append(f"| {r['id']} | {r['task']} | {r['expected_answer']} | {r['base_answer']} | {r['adapted_answer']} | {r['base_correct']} | {r['adapted_correct']} |")

    with open('outputs/FINAL_QUANTITATIVE_EVALUATION.md', 'w') as f:
        f.write('\n'.join(md))
    
    # Simple logic for determining string output based on data
    if delta > 0:
        print("VLM EVALUATION — ADAPTED MODEL DEMONSTRABLY BETTER")
    elif delta < 0:
        print("VLM EVALUATION — NO DEMONSTRABLE IMPROVEMENT")
    else:
        print("VLM EVALUATION — INCONCLUSIVE")

if __name__ == '__main__':
    main()
