import os
import argparse
import subprocess

def main():
    parser = argparse.ArgumentParser(description="GPU Integration Test for Authoritative Evaluation")
    parser.add_argument("--model", type=str, required=True, choices=["base", "adapted"])
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()
    
    print("WARNING: This test executes REAL evaluation on the GPU.")
    print("If you are on a 4GB VRAM machine (like an RTX 3050), this WILL fail with OOM.")
    print("This script is designed for the Colab T4 environment.\n")
    
    # We just run the authoritative evaluation script
    cmd = [
        "python", "evaluate_authoritative.py",
        "--model", args.model,
        "--output", f"outputs/evaluation_{args.model}.json"
    ]
    if args.model == "adapted":
        if not args.checkpoint:
            raise ValueError("Must provide --checkpoint for adapted model.")
        cmd.extend(["--checkpoint", args.checkpoint])
        
    print(f"Executing: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
