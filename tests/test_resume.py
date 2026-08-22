import torch
from transformers import LlavaForConditionalGeneration, BitsAndBytesConfig
from peft import PeftModel

def test_resume():
    print("Testing checkpoint resume...")
    q_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    base_model = LlavaForConditionalGeneration.from_pretrained(
        'llava-hf/llava-1.5-7b-hf',
        quantization_config=q_config,
        device_map={"": 0}
    )
    
    from pathlib import Path
    vlm_dir = Path(__file__).parent.parent
    ckpt_dir = vlm_dir / "checkpoints" / "final_adapter"
    # Reload adapter with is_trainable=True to resume training
    model = PeftModel.from_pretrained(base_model, str(ckpt_dir), is_trainable=True)
    
    # Verify it requires gradients
    trainable_params, all_param = model.get_nb_trainable_parameters()
    print(f"Resumed Trainable: {trainable_params} / {all_param}")
    
    assert trainable_params > 0, "No trainable parameters on resume!"
    print("Resume test PASSED!")
    
if __name__ == "__main__":
    test_resume()
