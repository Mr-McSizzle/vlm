import json
import argparse
from pathlib import Path

def generate_comparison_report(base_json, adapted_json, out_file):
    with open(base_json) as f:
        base_res = json.load(f)
    with open(adapted_json) as f:
        adapt_res = json.load(f)
        
    base_dict = {r["id"]: r for r in base_res}
    adapt_dict = {r["id"]: r for r in adapt_res}
    
    report = ["# Base vs Adapted Final Evaluation\n"]
    
    for r_id, b_r in base_dict.items():
        if r_id not in adapt_dict:
            continue
        a_r = adapt_dict[r_id]
        
        report.append(f"## {r_id} ({b_r['task'].upper()})")
        report.append(f"**Question:** {b_r['question']}")
        report.append(f"**Expected Answer:** {b_r['expected_answer']}")
        report.append(f"\n**Base LLaVA-1.5 7B Output:**\n> {b_r['model_answer']}")
        report.append(f"\n**Adapted Model Output:**\n> {a_r['model_answer']}")
        report.append("\n---\n")
        
    with open(out_file, 'w') as f:
        f.write("\n".join(report))
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_json", type=str, required=True)
    parser.add_argument("--adapted_json", type=str, required=True)
    parser.add_argument("--output", type=str, default="outputs/BASE_VS_ADAPTED_FINAL.md")
    args = parser.parse_args()
    
    generate_comparison_report(args.base_json, args.adapted_json, args.output)
    print(f"Generated comparison report at {args.output}")
