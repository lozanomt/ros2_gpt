#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"

class SubscriberNode : public rclcpp::Node
{
public:
    SubscriberNode() : Node("subscriber_node")
    {
        subscription_inference_ = this->create_subscription<std_msgs::msg::String>(
            "gpt_inference_output",
            10,
            std::bind(&SubscriberNode::inferenceCallback, this, std::placeholders::_1));

        subscription_results_ = this->create_subscription<std_msgs::msg::String>(
            "gpt_results_output",
            10,
            std::bind(&SubscriberNode::resultsCallback, this, std::placeholders::_1));
    }

private:
    void inferenceCallback(const std_msgs::msg::String::SharedPtr msg)
    {
        RCLCPP_INFO(this->get_logger(), "Inference Output: %s", msg->data.c_str());
    }

    void resultsCallback(const std_msgs::msg::String::SharedPtr msg)
    {
        RCLCPP_INFO(this->get_logger(), "Results Output: %s", msg->data.c_str());
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
