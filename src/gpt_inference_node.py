#!/usr/bin/env python3

import os
import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

class GPTInferenceNode(Node):
    def __init__(self, model_path):
        super().__init__('gpt_inference_node')

        # Load tokenizer and model
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.model = GPT2LMHeadModel.from_pretrained(model_path)  # Update with the actual path

        # Subscribe to input topic
        self.subscription = self.create_subscription(
            String,
            'gpt_inference_input',
            self.input_callback,
            10
        )

        # Publisher for output topic
        self.pub = self.create_publisher(String, 'gpt_inference_output', 10)

    def input_callback(self, msg):
        # Perform inference
        input_text = msg.data
        response = self.generate_response(input_text)

        # Publish the response
        self.pub.publish(String(data=response))

    def generate_response(self, input_text):
        # Tokenize input text
        input_ids = self.tokenizer.encode(input_text, return_tensors='pt')

        # Generate response
        with torch.no_grad():
            output = self.model.generate(input_ids, max_length=100, num_return_sequences=1)

        # Decode and return response
        response = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return response

def main(model_path):
    rclpy.init()
    gpt_node = GPTInferenceNode(model_path)
    rclpy.spin(gpt_node)
    rclpy.shutdown()

if __name__ == "__main__":
    model_path = os.path.join(os.path.dirname(__file__), '../..', 'share', 'ros2_gpt', 'models')
    main(model_path)

import rclpy
from std_msgs.msg import String
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

class GPTInferenceNode:
    def __init__(self, model_path):
        # Initialize ROS node
        rclpy.init_node('gpt_inference_node')

        # Load tokenizer and model
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

        self.model = GPT2LMHeadModel.from_pretrained(model_path)  # Update with the actual path

        # Subscribe to input topic
        rclpy.Subscriber("gpt_inference_input", String, self.input_callback)

        # Publisher for output topic
        self.pub = rclpy.Publisher("gpt_inference_output", String, queue_size=10)

        rclpy.spin()

    def input_callback(self, msg):
        # Perform inference
        input_text = msg.data
        response = self.generate_response(input_text)

        # Publish the response
        self.pub.publish(response)

    def generate_response(self, input_text):
        # Tokenize input text
        input_ids = self.tokenizer.encode(input_text, return_tensors='pt')

        # Generate response
        with torch.no_grad():
            output = self.model.generate(input_ids, max_length=100, num_return_sequences=1)

        # Decode and return response
        response = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return response

if __name__ == "__main__":
    try:
        model_path = os.path.join(os.path.dirname(__file__), '../..', 'share', 'ros2_gpt', 'models')
        GPTInferenceNode(model_path)
    except rclpy.ROSInterruptException:
        pass
