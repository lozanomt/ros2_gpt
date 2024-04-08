#!/usr/bin/env python3

import os
import torch
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer
from torch.utils.data import DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup
import threading
import time
import rclpy
import json
from rclpy.node import Node
from std_msgs.msg import String
from ros2_msgs.msg import Result
from ros2_msgs.srv import RequestResult
from ros2_gpt import model, tokenizer

class GPTNode(Node):

    def __init__(self):
        super().__init__('gpt_node')

        try:
            self.training_subscription = self.create_subscription(
                String,
                'training_input',
                self.training_callback,
                10)
        except Exception as ex:
            self.get_logger().error(f"Failed to subscribe to topic /training_input: {ex}")
        
        try:
            self.validation_subscription = self.create_subscription(
                String,
                'validation_input',
                self.validation_callback,
                10)
        except Exception as ex:
            self.get_logger().error(f"Failed to subscribe to topic /training_input: {ex}")
        
        try:
            self.inference_subscription = self.create_subscription(
                String,
                'inference_input',
                self.inference_callback,
                10)
        except Exception as ex:
            self.get_logger().error(f"Failed to subscribe to topic /training_input: {ex}")
        
        self.response_publisher = self.create_publisher(String, 'gpt_response', 10)
        self.training_results_publisher = self.create_publisher(Result, 'gpt_training_results', 10)
        self.validation_results_publisher = self.create_publisher(Result, 'gpt_validation_results', 10)

        self.training_clients = []

        self.model_path = os.path.join(os.path.dirname(__file__), '../..', 'share', 'ros2_gpt', 'models')
        self.model = None
        self.tokenizer = None
        self.max_length = 512
        self.best_val_loss = float('inf')
        self.load_model()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-5)

        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def load_model(self):
        config_path = os.path.join(self.model_path, 'config.json')
        model_path = os.path.join(self.model_path, 'model.safetensors')
        
        if os.path.exists(config_path) and os.path.exists(model_path):
            self.get_logger().info('Loading model from: %s' % self.model_path)
            config = GPT2Config.from_json_file(config_path)
            self.model = GPT2LMHeadModel.from_pretrained(self.model_path, config=config)
            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        else:
            self.get_logger().info('No model found. Loading default GPT2 model.')
            self.model = GPT2LMHeadModel.from_pretrained('gpt2')
            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

        # Add padding token
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    def init_training_client_callback(self, msg):
        self.training_clients.append(GPTTrainingClient(self))

    def training_callback(self, msg):
        self.get_logger().info('Received training data: %s' % msg.data)
        
        # Tokenize input text
        input_ids = self.tokenizer.encode(msg.data.strip(), max_length=self.max_length, truncation=True, return_tensors="pt")

        # Define dataset
        class InputDataset(Dataset):
            def __init__(self, inputs):
                self.inputs = inputs

            def __len__(self):
                return len(self.inputs)

            def __getitem__(self, idx):
                return self.inputs[idx]

        dataset = InputDataset(input_ids)

        # Set batch size
        batch_size = 1  # Since we're processing single input in this case

        # Create DataLoader
        data_loader = DataLoader(dataset, batch_size=batch_size)

        # Train the model
        result = Results()
        result.loss = self.train_model(data_loader)
        self.training_results_publisher.publish(result)

    def validation_callback(self, msg):
        self.get_logger().info('Received validation data: %s' % msg.data)
        
        # Tokenize input text
        input_ids = self.tokenizer.encode(msg.data.strip(), max_length=self.max_length, truncation=True, return_tensors="pt")

        # Define dataset
        class InputDataset(Dataset):
            def __init__(self, inputs):
                self.inputs = inputs

            def __len__(self):
                return len(self.inputs)

            def __getitem__(self, idx):
                return self.inputs[idx]

        dataset = InputDataset(input_ids)

        # Set batch size
        batch_size = 1  # Since we're processing single input in this case

        # Create DataLoader
        data_loader = DataLoader(dataset, batch_size=batch_size)

        # Evaluate the model
        result = Results()
        result.loss = self.evaluate_model(data_loader)
        self.validation_results_publisher.publish(result)

        # Save model if validation loss improved
        self.save_model()

    def inference_callback(self, msg):
        self.get_logger().info('Received inference input: %s' % msg.data)

        result = String()
        result.data = self.generate_response(msg.data)
        self.response_publisher.publish(result)

    def train_model(self, data_loader, num_epochs=10):
        # Define optimizer
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-5)

        # Define scheduler
        num_training_steps = len(data_loader) * num_epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

        # Define training loop
        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0.0
            total_correct = 0
            total_samples = 0

            for batch_idx, batch in enumerate(data_loader):
                batch = batch.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(input_ids=batch, labels=batch)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()

                # Calculate accuracy
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)
                labels = batch
                correct = (predictions == labels).sum().item()
                total_correct += correct
                total_samples += batch.size(0)

            # Calculate average loss for the epoch
            avg_loss = total_loss / len(data_loader)
            accuracy = total_correct / total_samples

            # Print loss for this epoch
            self.get_logger().info(f"Epoch {epoch+1}/{num_epochs}, Avg Loss: {avg_loss:.4f}")

        return avg_loss

    def evaluate_model(self, val_loader):
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(self.device)
                outputs = self.model(input_ids=batch, labels=batch)
                loss = outputs.loss
                total_loss += loss.item()
                
                # For accuracy calculation (assuming classification task)
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)
                labels = batch  # Assuming labels are included in the batch
                correct = (predictions == labels).sum().item()
                total_correct += correct
                total_samples += batch.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = total_correct / total_samples
        
        return avg_loss
    
    def generate_response(self, input_text):
        input_ids = self.tokenizer.encode(input_text, return_tensors='pt').to(self.device)
        
        with torch.no_grad():
            output = self.model.generate(input_ids, 
                                        attention_mask=torch.ones_like(input_ids),
                                        max_length=self.max_length, 
                                        num_return_sequences=1)
            
        response = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return response

    def save_model(self):
        # Saving model is not supported for this transformer version
        self.get_logger().info(f'Model saved to {self.model_path}.')
        self.model.save_pretrained(self.model_path)

    def publish_result(self, data):
        msg = Results()
        msg.loss = data
        self.results_publisher_.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    gpt_node = GPTNode()
    rclpy.spin(gpt_node)
    gpt_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
