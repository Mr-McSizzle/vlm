# Training Feasibility Analysis

## Local Hardware
- **GPU:** NVIDIA RTX 3050 (4 GB VRAM)
- **Model:** LLaVA-1.5 7B

## Observations
- **Loading:** Loading the base model in 4-bit NormalFloat precision immediately consumes ~4 GB VRAM.
- **Inference:** Single-image inference pushes VRAM to ~4.2 GB, causing a slight spillover into shared system memory.
- **Training:** Adding a minimal LoRA adapter (r=8) adds ~8M trainable parameters. Even with a batch size of 1 and sequence length of ~600 tokens, the backward pass requires storing activations. This forces the GPU to swap heavily to shared memory, reducing throughput to less than 1 step per minute.

## Conclusion
**LOCAL TRAINING ONLY FOR SMOKE TEST**
**REMOTE GPU REQUIRED**

While the smoke test successfully proves the pipeline (dataloader, processor, forward/backward passes, LoRA injection), actual fine-tuning (e.g., thousands of steps on the VQA datasets) on this hardware is impossible within the hackathon time constraints due to aggressive OOM/swapping. A remote GPU with at least 16GB (preferably 24GB, e.g., RTX 3090/4090 or A10G) is required for the real training run.
