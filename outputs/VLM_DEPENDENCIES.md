# VLM Dependencies

This document tracks the explicit third-party packages required for the SatQuery VLM pipeline. Extraneous packages from the root environment (such as `airsim`, `xgboost`, `opencv`, etc.) have been strictly omitted to ensure seamless Colab deployment.

| Package | Version | Why it is needed | Which VLM component uses it |
| :--- | :--- | :--- | :--- |
| **torch** | `2.12.0+cu130` | Core deep learning framework for tensor operations, automatic differentiation, and model execution on GPUs. | All model scripts (`train_lora.py`, `dataset.py`, `inference.py`, `evaluate_final.py`). |
| **transformers** | `4.56.2` | Provides the base `LlavaForConditionalGeneration` architecture, `AutoProcessor`, and pre-trained weights. | `train_lora.py`, `inference.py`, `evaluate_baseline.py`, `evaluate_final.py`. |
| **peft** | `0.20.0` | Enables Parameter-Efficient Fine-Tuning (LoRA). Used to inject the adapter into the language model and freeze the vision encoder. | `train_lora.py`, `evaluate_final.py`, `tests/test_lora_setup.py`. |
| **accelerate** | `1.14.0` | Required for automatic device mapping (`device_map="auto"`) and gradient checkpointing hooks in 4-bit PEFT setups. | `train_lora.py`, implicit in `evaluate_baseline.py` for model loading. |
| **bitsandbytes** | `0.50.1` | Enables 4-bit NormalFloat quantization (`BitsAndBytesConfig`) to fit the 7B model on limited VRAM hardware. | `train_lora.py`, `evaluate_baseline.py`, `evaluate_final.py`. |
| **datasets** | `5.0.1` | Hugging Face datasets library used to download and stream remote-sensing dataset assets. | `download_subsets.py` |
| **pillow** | `11.3.0` | Standard image processing library (imported as `PIL`) used to load satellite JPG images into the VLM processor. | `dataset.py`, `evaluate_baseline.py`, `evaluate_final.py`. |
| **PyYAML** | `6.0.2` | Parses the `config.yaml` file to dynamically configure dataset splits, target ratios, and LoRA hyperparameters. | `train_lora.py`, `prepare_data.py`, `evaluate_final.py`. |
| **pydantic** | `2.11.9` | Provides strict schema validation for the VLM inference I/O, ensuring P3 Controller compatibility. | `schemas.py` |
| **pytest** | `9.0.2` | Test runner used to execute the quality gates, baseline tests, and smoke tests. | `tests/` directory (`test_dataset.py`, `test_lora_setup.py`, etc.). |
