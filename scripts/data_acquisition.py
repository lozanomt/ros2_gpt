#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from std_msgs.msg import String
import json
from datasets import load_dataset
from ros2_msgs.srv import RequestResult

class DataAcquisitionNode(Node):
    def __init__(self):
        super().__init__('data_acquisition')
        self.declare_parameters(namespace="",
                                parameters=[
                                    ('dataset_name', rclpy.Parameter.Type.STRING)
                                ])
        dataset_name = Parameter('dataset_name', Parameter.Type.STRING, 'wikipedia')

        self.set_parameters(parameter_list=[dataset_name])


        self.service_publisher = self.create_service(RequestResult, '/data_acquisition/data', self.callback)
        self.current_index = 0
        self.dataset = self.download()
        self.training_dataset = self.dataset['train']

    def callback(self, request, response):
        self.get_logger().info(f"Received request")
        try:
            returnval = json.dumps(self.training_dataset[self.current_index]['text'])
        except Exception as ex:
            self.get_logger().error(f"Error retrieving data: {ex}")
        response.success = True
        response.problem = ""
        response.returnval = returnval
        self.get_logger().info(f"Published batch data:\n\tSuccess: {response.success}\n\tProblem: {response.problem}\n\tReturnVal: {response.returnval}\n\t")
        self.current_index += 1
        return response

    def download(self):
        try:
            print(self.get_parameter('dataset_name').value)
            dataset = load_dataset(self.get_parameter('dataset_name').value, "20220301.en")
            return dataset
        except Exception as e:
            self.get_logger().error(f"Failed to download dataset: {e}")

def main(args=None):
    rclpy.init(args=args)
    data_acquisition_node = DataAcquisitionNode()
    rclpy.spin(data_acquisition_node)
    data_acquisition_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
