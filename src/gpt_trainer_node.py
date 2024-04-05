#!/usr/bin/env python3

import os
import torch
from transformers import GPT2Config, GPT2LMHeadModel
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class GPTTrainerNode(Node):

    def __init__(self):
        super().__init__('gpt_trainer_node')
        self.subscription = self.create_subscription(
            String,
            'gpt_training_input',
            self.training_callback,
            10)
        self.publisher_ = self.create_publisher(String, 'gpt_results_output', 10)
        self.model_path = os.path.join(os.path.dirname(__file__), '../..', 'share', 'ros2_gpt', 'models')
        self.model = None
        self.load_model()

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

    def save_model(self):
        # Saving model is not supported for this transformer version
        self.get_logger().info('Model saving is not supported for this version.')

    def training_callback(self, msg):
        self.get_logger().info('Received training data: %s' % msg.data)
        
        # Placeholder for training process
        # Here, you can perform your training process using the received text data
        # Replace this with actual training process based on your requirement
        
        # For demonstration, let's just echo the received data
        result_data = "Processed: " + msg.data
        
        # Publish the processed data to gpt_results_output topic
        self.publish_result(result_data)

    def publish_result(self, data):
        msg = String()
        msg.data = data
        self.publisher_.publish(msg)
        self.get_logger().info('Published result: %s' % msg.data)

def main(args=None):
    rclpy.init(args=args)
    gpt_trainer_node = GPTTrainerNode()
    rclpy.spin(gpt_trainer_node)
    gpt_trainer_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
