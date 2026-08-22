# T4 Real Training Configuration

The repository is now fully configured for the **REAL T4 TRAINING RUN** (Google Colab free-tier).
Smoke-test limitations have been removed.

## Expected Training Length

- **Training Records**: 80 examples
- **Batch Size**: 1
- **Batches per epoch**: 80
- **Gradient Accumulation Steps**: 8
- **Optimizer Updates per epoch**: 10  (80 / 8)
- **Total Epochs**: 3
- **Total Optimizer Updates**: 30  (10 updates/epoch * 3 epochs)

## Real Training Configuration (`config.yaml`)

```yaml
training:
  batch_size: 1
  gradient_accumulation_steps: 8
  learning_rate: 2e-4
  num_train_epochs: 3
  max_steps: -1
  save_steps: 50
  logging_steps: 5
  max_grad_norm: 1.0
  warmup_ratio: 0.03
  fp16: true
  output_dir: "checkpoints/final_adapter"
```

## Safety Assurances

1. **No Smoke-Test Logic**: `max_steps=-1` is now perfectly supported and relies entirely on `num_train_epochs`. 
2. **Gradient Accumulation Verified**: Loss is divided by `gradient_accumulation_steps`, and the optimizer is only stepped (and cleared) exactly every 8 batches or at the end of the epoch.
3. **Save and Resume Mechanism**: Checkpoints are saved every 50 global steps. Because there are 30 total steps, it will save once at the very end of training. If `save_steps` is lowered (e.g., to 10), it correctly saves intermediate states (including `optimizer.pt`) which can be resumed with the `--resume_from_checkpoint` flag.
4. **Learning Rate Management**: The `transformers` linear scheduler with warmup has been explicitly added to track learning rate correctly based on the dynamically computed 30 optimizer steps.
5. **No 4-step Limits**: All hard-coded `step >= steps: break` limits effectively shut off when `max_steps` is `-1`.

## Preflight Status

Validation against the configuration structure via `t4_preflight.py` is **PASSED**. No further step limits exist.
