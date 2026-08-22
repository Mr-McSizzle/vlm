import json
import argparse
from pathlib import Path
from scoring import score_prediction

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=str, default="outputs/evaluation_base.json")
    parser.add_argument("--adapted", type=str, default="outputs/evaluation_adapted.json")
    args = parser.parse_args()
    
    with open(args.base) as f:
        base_data = {item["id"]: item for item in json.load(f)}
        
    with open(args.adapted) as f:
        adapted_data = {item["id"]: item for item in json.load(f)}
        
    comparisons = []
    
    summary = {
        "total_evaluated": 0,
        "base_success": 0,
        "adapted_success": 0,
        "scoring_coverage": 0,
        "base_correct": 0,
        "adapted_correct": 0,
        "adapted_improved": 0,
        "adapted_degraded": 0,
        "both_correct": 0,
        "both_incorrect": 0,
        "task_breakdown": {},
        "failures": []
    }
    
    for rid, adapted_record in adapted_data.items():
        if rid not in base_data:
            continue
            
        base_record = base_data[rid]
        task = adapted_record["task"]
        expected = adapted_record["expected_answer"]
        
        base_pred = base_record.get("prediction", {}).get("answer", "")
        adapted_pred = adapted_record.get("prediction", {}).get("answer", "")
        
        # Guard: check if the inference actually succeeded.
        b_success = base_record.get("success", False)
        a_success = adapted_record.get("success", False)
        
        summary["total_evaluated"] += 1
        if b_success: summary["base_success"] += 1
        if a_success: summary["adapted_success"] += 1
        
        # Scoring
        b_score = score_prediction(expected, base_pred, task) if b_success else {"correct": False, "failure_category": "inference failure"}
        a_score = score_prediction(expected, adapted_pred, task) if a_success else {"correct": False, "failure_category": "inference failure"}
        
        # Record failures for analysis
        if a_score.get("correct") is False:
            summary["failures"].append({
                "id": rid,
                "task": task,
                "category": a_score.get("failure_category", "unknown"),
                "expected": expected,
                "predicted": adapted_pred
            })
            
        b_c = b_score.get("correct")
        a_c = a_score.get("correct")
        
        changed = (b_pred != adapted_pred) if 'b_pred' in locals() else (base_pred != adapted_pred) # safe check
        
        category = "unscored"
        if b_c is not None and a_c is not None:
            summary["scoring_coverage"] += 1
            if a_c: summary["adapted_correct"] += 1
            if b_c: summary["base_correct"] += 1
            
            if a_c and not b_c:
                category = "adapted_improved"
                summary["adapted_improved"] += 1
            elif b_c and not a_c:
                category = "adapted_degraded"
                summary["adapted_degraded"] += 1
            elif b_c and a_c:
                category = "both_correct"
                summary["both_correct"] += 1
            else:
                category = "both_incorrect"
                summary["both_incorrect"] += 1
                
        if task not in summary["task_breakdown"]:
            summary["task_breakdown"][task] = {"total": 0, "base_correct": 0, "adapted_correct": 0, "scorable": 0}
            
        summary["task_breakdown"][task]["total"] += 1
        if a_c is not None:
            summary["task_breakdown"][task]["scorable"] += 1
            if b_c: summary["task_breakdown"][task]["base_correct"] += 1
            if a_c: summary["task_breakdown"][task]["adapted_correct"] += 1

        comparisons.append({
            "id": rid,
            "task": task,
            "expected": expected,
            "base_answer": base_pred,
            "adapted_answer": adapted_pred,
            "base_correct": b_c,
            "adapted_correct": a_c,
            "changed": (base_pred != adapted_pred),
            "category": category
        })
        
    Path("outputs").mkdir(exist_ok=True)
    with open("outputs/BASE_VS_ADAPTED_FINAL.json", 'w') as f:
        json.dump(comparisons, f, indent=2)
        
    # Write Summary Markdown
    with open("outputs/EVALUATION_SUMMARY.md", "w") as f:
        f.write("# VLM Evaluation Summary\n\n")
        f.write(f"- **Total Evaluated**: {summary['total_evaluated']}\n")
        f.write(f"- **Base Successes (Inference)**: {summary['base_success']}\n")
        f.write(f"- **Adapted Successes (Inference)**: {summary['adapted_success']}\n")
        f.write(f"- **Scorable Examples**: {summary['scoring_coverage']}\n\n")
        
        if summary['scoring_coverage'] > 0:
            b_acc = summary['base_correct'] / summary['scoring_coverage'] * 100
            a_acc = summary['adapted_correct'] / summary['scoring_coverage'] * 100
            f.write(f"## Measured Accuracy\n")
            f.write(f"- **Base Accuracy**: {b_acc:.1f}%\n")
            f.write(f"- **Adapted Accuracy**: {a_acc:.1f}%\n\n")
            f.write(f"## Comparison\n")
            f.write(f"- **Adapted Improved**: {summary['adapted_improved']}\n")
            f.write(f"- **Adapted Degraded**: {summary['adapted_degraded']}\n")
            f.write(f"- **Both Correct**: {summary['both_correct']}\n")
            f.write(f"- **Both Incorrect**: {summary['both_incorrect']}\n\n")
            
        f.write("## Task Breakdown\n")
        for t, stats in summary["task_breakdown"].items():
            scorable = stats["scorable"]
            if scorable > 0:
                b_a = stats["base_correct"] / scorable * 100
                a_a = stats["adapted_correct"] / scorable * 100
                f.write(f"- **{t}**: Base {b_a:.1f}% | Adapted {a_a:.1f}% ({scorable} scorable)\n")
            else:
                f.write(f"- **{t}**: {stats['total']} total (Unscorable qualitative)\n")
                
    # Write Failure Analysis Markdown
    with open("outputs/FAILURE_ANALYSIS.md", "w") as f:
        f.write("# Failure Analysis (Adapted Model)\n\n")
        categories = {}
        for fail in summary["failures"]:
            cat = fail["category"]
            categories.setdefault(cat, []).append(fail)
            
        for cat, fails in categories.items():
            f.write(f"## {cat.title()} ({len(fails)} cases)\n")
            # Only print first 5 of each type to prevent giant files
            for fail in fails[:5]:
                f.write(f"- **ID**: {fail['id']} | **Task**: {fail['task']}\n")
                f.write(f"  - **Expected**: {fail['expected']}\n")
                f.write(f"  - **Predicted**: {fail['predicted']}\n")
            if len(fails) > 5:
                f.write(f"  - *(...and {len(fails) - 5} more)*\n")
            f.write("\n")
            
    print("Comparison complete. Check outputs/EVALUATION_SUMMARY.md")

if __name__ == "__main__":
    main()
