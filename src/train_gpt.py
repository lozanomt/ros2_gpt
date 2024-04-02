#!/usr/bin/env python3

import os
import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.utils.data import DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup

# Define your Dataset class if not already defined
class YourDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        # Add any additional initialization if needed

        # Load and preprocess data
        self.data = self.load_data()

    def load_data(self):
        # Implement your data loading logic here
        # This could involve reading files, processing them, etc.
        # Example: Read lines from a text file
        with open(self.data_path, 'r', encoding='utf-8') as file:
            data = file.readlines()

        # Preprocess data, tokenization, etc.
        preprocessed_data = [self.tokenizer.encode(text.strip(), max_length=self.max_length, truncation=True) for text in data]

        return preprocessed_data

    def __len__(self):
        # Return the total number of samples
        return len(self.data)

    def __getitem__(self, idx):
        # Return the item at the given index
        return torch.tensor(self.data[idx], dtype=torch.long)


def train_gpt(train_data_loader, num_epochs=3, model_name_or_path='gpt2', output_dir='src/gpt_training/models/'):
    # Load tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path)
    model = GPT2LMHeadModel.from_pretrained(model_name_or_path)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    # Define scheduler
    num_training_steps = len(train_data_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    # Define your training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for batch in train_data_loader:
            # Move batch to device
            batch = batch.to(device)

            # Forward pass
            outputs = model(input_ids=batch, labels=batch)
            loss = outputs.loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()  # Update learning rate

            # Update total loss
            total_loss += loss.item()

        # Calculate average loss for the epoch
        avg_loss = total_loss / len(train_data_loader)

        # Print loss for this epoch
        print(f"Epoch {epoch+1}/{num_epochs}, Avg Loss: {avg_loss:.4f}")

    # Save trained model
    model.save_pretrained(output_dir)


# Instantiate your Dataset
data_path = f"{ROOT}/data/test_data.txt"
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
max_length = 512

train_dataset = YourDataset(data_path=data_path, tokenizer=tokenizer, max_length=max_length)

# Set batch size
batch_size = 4

def collate_fn(batch):
    # Pad sequences to the maximum length in the batch
    max_len = max([seq.size(0) for seq in batch])
    if tokenizer.pad_token is None:
        pad_value = 0  # Use 0 as the padding value if pad_token is None
    else:
        pad_value = tokenizer.pad_token_id

    padded_seqs = [torch.cat([seq, torch.tensor([pad_value] * (max_len - seq.size(0)))]) for seq in batch]
    input_ids = torch.stack(padded_seqs).to(torch.long)  # Convert to long tensor
    return input_ids

# Create DataLoader with collate_fn
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# Set the number of epochs
num_epochs = 50

# Example usage:
train_gpt(train_loader, num_epochs=num_epochs)
