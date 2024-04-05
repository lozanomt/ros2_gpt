#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import requests
from bs4 import BeautifulSoup
import random
import time
import re

class WikipediaDownloader(Node):
    def __init__(self):
        super().__init__('wikipedia_downloader')
        self.publisher_ = self.create_publisher(String, 'gpt_results_output', 10)
        self.num_seconds = 60  # Default value for the interval
        self.timer_period = 1  # Check every second
        self.timer = self.create_timer(self.timer_period, self.timer_callback)

    def timer_callback(self):
        if self.timer_period % self.num_seconds == 0:
            self.download_wikipedia_page()
        self.timer_period += 1

    def download_wikipedia_page(self):
        url = "https://en.wikipedia.org/wiki/Special:Random"
        try:
            response = requests.get(url)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                # Find all paragraphs within the content text
                paragraphs = soup.find('div', {'id': 'mw-content-text'}).find_all('p')
                # Extract text from each paragraph and concatenate
                cleaned_text = "\n".join([paragraph.get_text() for paragraph in paragraphs])
                # Remove citations, references, and other elements
                cleaned_text = re.sub(r'\[.*?\]', '', cleaned_text)
                # Publish the cleaned text
                self.publish_text(cleaned_text)
        except Exception as e:
            self.get_logger().error(f"Failed to download Wikipedia page: {e}")


    def publish_text(self, text):
        msg = String()
        msg.data = text
        self.publisher_.publish(msg)
        self.get_logger().info(f"Published:\n{text}")

def main(args=None):
    rclpy.init(args=args)
    wikipedia_downloader = WikipediaDownloader()
    rclpy.spin(wikipedia_downloader)
    wikipedia_downloader.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
