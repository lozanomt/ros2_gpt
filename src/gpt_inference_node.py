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
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
import torch

class GPTInferenceNode(Node):
    def __init__(self):
        super().__init__('gpt_inference_node')

        # Load tokenizer and model
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.model_path = os.path.join(os.path.dirname(__file__), '../..', 'share', 'ros2_gpt', 'models')
        self.model = None
        self.load_model()

        # Subscribe to input topic
        self.subscription = self.create_subscription(
            String,
            'gpt_inference_input',
            self.input_callback,
            10
        )

        # Publisher for output topic
        self.pub = self.create_publisher(String, 'gpt_inference_output', 10)

    def load_model(self):
        config_path = os.path.join(self.model_path, 'config.json')
        model_path = os.path.join(self.model_path, 'model.safetensors')
        
        if os.path.exists(config_path) and os.path.exists(model_path):
            self.get_logger().info('Loading model from: %s' % self.model_path)
            config = GPT2Config.from_json_file(config_path)
            self.model = GPT2LMHeadModel.from_pretrained(self.model_path, config=config)
        else:
            self.get_logger().info('No model found. Loading default GPT2 model.')
            self.model = GPT2LMHeadModel.from_pretrained('gpt2')

    def input_callback(self, msg):
        # Perform inference
        input_text = msg.data
        response = self.generate_response(input_text)

        # Publish the response
        self.pub.publish(String(data=response))

    def generate_response(self, input_text):
        # Tokenize input text
        input_ids = self.tokenizer.encode(input_text, return_tensors='pt')

        # Generate attention mask
        attention_mask = torch.ones_like(input_ids)

        # Generate response
        with torch.no_grad():
            output = self.model.generate(input_ids, attention_mask=attention_mask, max_length=100, num_return_sequences=1)

        # Decode and return response
        response = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return response

def main(args=None):
    rclpy.init(args=args)
    gpt_node = GPTInferenceNode()
    rclpy.spin(gpt_node)
    rclpy.shutdown()

if __name__ == "__main__":
    main()
