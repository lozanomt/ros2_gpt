#!/usr/bin/env python3

import rclpy
import json
import time
import os
import torch
import threading
import traceback
from torch.utils.data import DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup
from rclpy.node import Node
from rclpy.parameter import Parameter
from std_msgs.msg import String
from ros2_msgs.msg import Result
from ros2_msgs.srv import RequestResult
from ros2_gpt import model_path, model, tokenizer, optimizer, scheduler

class GPTNode(Node):
    def __init__(self):
        super().__init__('gpt_node')

        self.GPTTrainingThread = threading.Thread(target=self.GPTTrainingThreadSub, name="GPTTrainingThread")
        self.GPTTrainingThread.start()
        self.GPTInferencingThread = threading.Thread(target=self.GPTInferencingThreadSub, name="GPTInferencingThread")
        self.GPTInferencingThread.start()
    
    def GPTTrainingThreadSub(self):
        self.client = self.create_client(RequestResult, '/data_acquisition/data')
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(f'Waiting for service: {r"/data_acquisition/data"}')
        self.get_logger().info(f'Connected to service: {r"/data_acquisition/data"}')
        
        # Timer to periodically make requests to data acquisition services
        self.request_timer_period = 0.2  # Adjust as needed in seconds
        self.request_timer = self.create_timer(self.request_timer_period, self.training_callback)   

        self.training_results_publisher = self.create_publisher(Result, 'gpt_training_results', 10)

    def GPTInferencingThreadSub(self):
        try:
            self.inference_subscription = self.create_subscription(
                String,
                'inference_input',
                self.inference_callback,
                10)
        except Exception as ex:
            self.get_logger().error(f"Failed to subscribe to topic /training_input: {ex}")       

        self.response_publisher = self.create_publisher(String, 'gpt_response', 10)

                                     
    def inference_callback(self, msg):
        self.get_logger().info('Received inference input: %s' % msg.data)

        result = String()
        result.data = self.generate_response(msg.data)
        self.response_publisher.publish(result)

    def generate_response(self, input_text):
        input_ids = tokenizer.encode(input_text, return_tensors='pt').to(model.device)
        
        with torch.no_grad():
            output = model.generate(input_ids, 
                                    attention_mask=torch.ones_like(input_ids),
                                    max_length=1024, 
                                    num_return_sequences=1)
            
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        return response

    def training_callback(self):
        # Make requests to each data acquisition service and process the responses
        request = RequestResult.Request()
        future = self.client.call_async(request)
        future.add_done_callback(self.handle_response)

    def handle_response(self, future):
        try:
            response = future.result()
            # Convert JSON string to dictionary
            data = json.loads(response.returnval)
            print(data)
        except Exception as ex:
            self.get_logger().info(f"Service call failed: {ex}")
        else:
            if response is not None:
                # Tokenize each field and encode batched data
                input_ids = tokenizer.encode(data, max_length=1024, truncation=True, return_tensors="pt")

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

                # Create DataLoader with batch size 1
                data_loader = DataLoader(dataset, batch_size=batch_size)

                # Define training loop
                model.train()
                total_loss = 0.0
                total_correct = 0
                total_samples = 0

                for batch_idx, batch in enumerate(data_loader):
                    batch = batch.to(model.device)

                    optimizer.zero_grad()
                    outputs = model(input_ids=batch, labels=batch)
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
                self.get_logger().info(f"Avg Loss: {avg_loss:.4f}")
                result = Result()
                result.loss = avg_loss
                self.training_results_publisher.publish(result)

                # Save model if validation loss improved
                model.save_pretrained(model_path)

    def shutdown(self):
        try:
            if self.GPTTrainingThread.is_alive():
                self.GPTTrainingThread.join()
            if self.GPTInferencingThread.is_alive():
                self.GPTInferencingThread.join()

        except Exception as ex:
            self.get_logger().error(f"{traceback.format_exc()}")

        finally:
            self.destroy_node()

def main(args=None):
    rclpy.init(args=args)
    gpt_node = GPTNode()
    rclpy.spin(gpt_node)
    gpt_node.shutdown()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
