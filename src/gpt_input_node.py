#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class UserInputNode(Node):
    def __init__(self):
        super().__init__('user_input_node')
        self.publisher_ = self.create_publisher(String, 'user_input_topic', 10)
        self.timer_ = self.create_timer(0.5, self.publish_user_input)

    def publish_user_input(self):
        user_input = input("Enter your message: ")
        msg = String()
        msg.data = user_input
        self.publisher_.publish(msg)
        self.get_logger().info('Published: "%s"' % msg.data)

def main(args=None):
    rclpy.init(args=args)
    user_input_node = UserInputNode()
    rclpy.spin(user_input_node)
    user_input_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()