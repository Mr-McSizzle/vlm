import json
import argparse
import subprocess
from datetime import datetime
from pathlib import Path
from scoring import score_prediction

def get_git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
    except Exception:
        return "unknown"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=str, default="outputs/evaluation_base.json")
    parser.add_argument("--adapted", type=str, default="outputs/evaluation_adapted.json")
    args = parser.parse_args()
    
    with open(args.base) as f:
        base_data = {item["id"]: item for item in json.load(f)}
        
    with open(args.adapted) as f:
        adapted_data = {item["id"]: item for item in json.load(f)}
        
    # Hard Validation 1: IDs must match
    base_ids = set(base_data.keys())
    adapted_ids = set(adapted_data.keys())
    if base_ids != adapted_ids:
        missing_in_adapted = base_ids - adapted_ids
        missing_in_base = adapted_ids - base_ids
        raise ValueError(f"ID mismatch between base and adapted evaluations. Missing in adapted: {missing_in_adapted}. Missing in base: {missing_in_base}")
        
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
    
    # Pre-open md file to overwrite completely
    md_path = Path("outputs/BASE_VS_ADAPTED_FINAL.md")
    md_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(md_path, "w") as md_f:
        # Provenance header
        md_f.write(f"# Base vs Adapted Comparison\n\n")
        md_f.write(f"**Evaluation generated from:**\n")
        md_f.write(f"- base_json = {args.base}\n")
        md_f.write(f"- adapted_json = {args.adapted}\n")
        md_f.write(f"- Git commit = {get_git_commit()}\n")
        md_f.write(f"- generated_at = {datetime.utcnow().isoformat()}Z\n\n")
    
        for rid in sorted(base_ids):
            base_record = base_data[rid]
            adapted_record = adapted_data[rid]
            
            task = adapted_record["task"]
            question = adapted_record["question"]
            expected = adapted_record["expected_answer"]
            
            base_pred = base_record.get("prediction", {}).get("answer", "")
            adapted_pred = adapted_record.get("prediction", {}).get("answer", "")
            
            b_success = base_record.get("success", False)
            a_success = adapted_record.get("success", False)
            
            summary["total_evaluated"] += 1
            if b_success: summary["base_success"] += 1
            if a_success: summary["adapted_success"] += 1
            
            b_score = score_prediction(expected, base_pred, task) if b_success else {"correct": False, "failure_category": "inference failure"}
            a_score = score_prediction(expected, adapted_pred, task) if a_success else {"correct": False, "failure_category": "inference failure"}
            
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
            
            # Write to markdown file exactly one section per ID
            md_f.write(f"## Record ID: {rid}\n")
            md_f.write(f"- **Task**: {task}\n")
            md_f.write(f"- **Question**: {question}\n")
            md_f.write(f"- **Expected Answer**: {expected}\n")
            md_f.write(f"- **Base Prediction**: {base_pred}\n")
            md_f.write(f"- **Adapted Prediction**: {adapted_pred}\n")
            md_f.write(f"- **Base Correctness**: {b_c}\n")
            md_f.write(f"- **Adapted Correctness**: {a_c}\n\n")
        
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
            
    print("Comparison complete. Check outputs/BASE_VS_ADAPTED_FINAL.md and outputs/EVALUATION_SUMMARY.md")

if __name__ == "__main__":
    main()
