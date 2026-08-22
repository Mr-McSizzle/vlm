# Checkpoint Provenance Audit

## Evidence
- **Directory Inspected:** `checkpoints/final_adapter/`
- **Files Present (Prior to Archive):** `adapter_config.json`, `adapter_model.safetensors`, `README.md`, `LOCAL_SMOKE_TEST_ONLY.txt`
- **Timestamps:** Files generated on 22-08-2026.
- **Marker Content:** The `LOCAL_SMOKE_TEST_ONLY.txt` explicitly states:
  > LOCAL_SMOKE_TEST_ONLY. This checkpoint was generated on a local RTX 3050 over just 4 steps to verify the training pipeline... It is NOT a fully trained model. It was NOT trained on AWS.

## Conclusion
The provenance of the checkpoint currently existing in `checkpoints/final_adapter/` is unambiguously **A. local smoke-test weights**.

Therefore, the adapter is **INVALID / AMBIGUOUS** for any claim of full training.

## Actions Taken
- The invalid smoke-test checkpoint has been physically moved to `checkpoints/archived_invalid_or_smoke/`.
- The `checkpoints/final_adapter/` directory is now empty and strictly reserved for a genuine T4-trained checkpoint.
