import torch
from transformers import LlavaForConditionalGeneration, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
import yaml

def test_lora_trainable_parameters():
    with open('vlm/config.yaml') as f:
        config = yaml.safe_load(f)
        
    model_id = 'llava-hf/llava-1.5-7b-hf'
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4',
    )
    
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id, 
        quantization_config=quantization_config, 
        device_map={'': 0}
    )
    
    # We want to target only language model modules to keep vision encoder completely frozen!
    target_modules = config.get("lora", {}).get("target_modules", ".*language_model.*\.(q_proj|v_proj)$")
    
    peft_config = LoraConfig(
        r=config.get("lora", {}).get("r", 8),
        lora_alpha=config.get("lora", {}).get("alpha", 16),
        lora_dropout=config.get("lora", {}).get("dropout", 0.05),
        target_modules=target_modules,
        task_type="CAUSAL_LM"
    )
    
    peft_model = get_peft_model(model, peft_config)
    
    # Print trainable parameters
    trainable_params = 0
    all_param = 0
    for name, param in peft_model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            # Ensure it is a lora parameter
            assert "lora_" in name, f"Unexpected trainable parameter: {name}"
            
    # Verify vision encoder is frozen
    for name, param in peft_model.named_parameters():
        if "vision_tower" in name:
            assert not param.requires_grad, f"Vision tower parameter {name} is not frozen!"
            
    print(f"Total parameters: {all_param}")
    print(f"Trainable parameters: {trainable_params}")
    print(f"Trainable percentage: {100 * trainable_params / all_param:.4f}%")
    
    assert trainable_params > 0, "No trainable parameters found!"
    assert trainable_params < 0.1 * all_param, "Too many trainable parameters!"
    
if __name__ == "__main__":
    test_lora_trainable_parameters()
