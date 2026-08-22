# Remote Training Plan

Since local training is unfeasible due to the 4GB VRAM limit on the RTX 3050, the real training run must be executed on a remote compute instance.

## 1. Environment and Hardware
- **Recommended GPU:** 1x NVIDIA RTX 3090 / 4090 (24GB VRAM) or A10G / A100.
- **OS:** Ubuntu 22.04 LTS
- **Python:** 3.10+
- **CUDA:** 11.8+ or 12.x compatible with PyTorch 2.x

## 2. Dependencies
- Install the exact same requirements as the local environment:
  ```bash
  pip install torch torchvision transformers datasets peft accelerate bitsandbytes pyyaml pillow
  ```

## 3. Dataset Manifest
Copy the unified dataset and corresponding image directories to the remote instance:
- `vlm/data/unified/train.jsonl`
- `vlm/data/unified/val.jsonl`
- `vlm/data/external/rsvqa/images/`
- `vlm/data/external/cdvqa/images/`
- (Note: BigEarthNet.txt requires no images as it provides domain adaptation via text)

## 4. Configuration
Modify `vlm/config.yaml` on the remote instance for actual training:
- **Batch Size:** Increase to 4 or 8 (depending on sequence lengths).
- **Gradient Accumulation:** Adjust to achieve an effective batch size of ~32-64.
- **Max Steps / Epochs:** Train for 1-3 epochs over the dataset.
- **Learning Rate:** 2e-4 with a cosine scheduler and warmup.

## 5. Training Command
Execute the training script (can use `train_lora.py` with standard Trainer integrations):
```bash
python vlm/train_lora.py
```

## 6. Output Location
The final adapter weights will be saved to `vlm/checkpoints/final_adapter`. These weights (approx. 32MB for r=8) can then be downloaded back to the local machine for inference!
