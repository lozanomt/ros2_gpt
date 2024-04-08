#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include <iostream>


class SubscriberNode : public rclcpp::Node
{
public:
    SubscriberNode() : Node("output_subscriber")
    {
        subscription_inference_ = this->create_subscription<std_msgs::msg::String>(
            "gpt_response",
            10,
            std::bind(&SubscriberNode::Callback, this, std::placeholders::_1));
    }

private:
    void Callback(const std_msgs::msg::String::SharedPtr msg)
    {
        RCLCPP_INFO(this->get_logger(), "Inference Output: %s", msg->data.c_str());
    }

    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr subscription_inference_;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr subscription_results_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<SubscriberNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
