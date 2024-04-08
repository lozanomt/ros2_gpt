# model_initializer.py
import os
import torch
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer, get_linear_schedule_with_warmup

def initialize_model():
    model_directory_path = os.path.join(os.path.dirname(__file__), '../..', 'share', 'ros2_gpt', 'models')
    config_path = os.path.join(model_directory_path, 'config.json')
    model_path = os.path.join(model_directory_path, 'model.safetensors')
        
    if os.path.exists(config_path) and os.path.exists(model_path):
        config = GPT2Config.from_json_file(config_path)
        model = GPT2LMHeadModel.from_pretrained(model_path, config=config)
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    else:
        model = GPT2LMHeadModel.from_pretrained('gpt2')
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    tokenizer.pad_token = tokenizer.eos_token

    # Define optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    #define scheduler
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=1)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    return model_path, model, tokenizer, optimizer, scheduler
