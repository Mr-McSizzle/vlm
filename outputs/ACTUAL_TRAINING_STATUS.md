# Actual Training Status

**Status:** FULL TRAINING PENDING (REMOTE GPU REQUIRED)

## Current State
The project has successfully reached the final smoke-test phase on a local RTX 3050 (4GB VRAM) machine. 
- The unified dataset loads correctly.
- LLaVA-1.5 4-bit loading and LoRA injection are fully functional.
- The forward pass, backward pass, and checkpoint saving have all been proven to work.
- Due to the severe VRAM limitations of the local environment (causing heavy swapping to system RAM), full training is completely impractical.

## Clarification on `final_adapter`
The adapter weights currently stored in `vlm/checkpoints/final_adapter` were generated during a local 4-step proxy run. This artifact MUST be treated strictly as a **LOCAL SMOKE-TEST CHECKPOINT**. 
It is **NOT**:
- Remotely trained
- Fully fine-tuned
- A final benchmark checkpoint
- A10G-trained (or trained on any AWS instance)
- A final validated model

The actual remote training run is still pending execution by a teammate on a capable Linux GPU instance.
