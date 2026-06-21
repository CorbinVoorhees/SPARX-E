#ifndef AUTO_PUBLISHER_NODE
#define AUTO_PUBLISHER_NODE

#include <std_msgs/msg/int32.hpp>
#include <geometry_msgs/msg/vector3.hpp>
#include <rclcpp/rclcpp.hpp>
/**
 * @brief AutoPublisher specifically targets nav_node during runtime to kill and restart the Kalman Filter
 * when state estimation has collapsed, or the nav_node as failed to start. This was a patch solution before
 * competition to restart the nav_node, and also the Kalman Filter, and SHOULD NOT BE USED ANYMORE.
 *
 * @deprecated
 * @author Krish Sridhar
 */
class AutoPublisher : public rclcpp::Node
{
private:
    // subscriber to the 'start_navigation' topic that is issued by the ground station
    // to hear kill instructions.
    rclcpp::Subscription<geometry_msgs::msg::Vector3>::SharedPtr start_nav_subscriber;

    // publisher to the 'nav_node_successful' topic that is issued by the rover on successful
    // nav_node instantiation.
    rclcpp::Publisher<std_msgs::msg::Int32>::SharedPtr revival_status_publisher;

    /**
     * @brief Callback for the start_navigation protocol.
     *
     * @param msg Calibration message that specifies rover starting point from rover.
     */
    void start_navigation_callback(const geometry_msgs::msg::Vector3::SharedPtr msg)
    {
        // if user inputs <0.0,0.0,0.0> (0 vector), then issue the kill signal
        if (pow(msg->x, 2) + pow(msg->y, 2) + pow(msg->z, 2) < 1e-3)
        {
            // kill the node internally using the pfkill command, and sending it to the system
            std::string cmd = "pkill -f nav_node";
            int status = std::system(cmd.c_str());
            std_msgs::msg::Int32 msg;

            // send kill status through publisher
            msg.data = WEXITSTATUS(status);
            this->revival_status_publisher->publish(msg);
        }
        else
        {
            // otherwise, we should restart nav_node with the following parameters
            // as a necesary initialization step.
            std::string cmd = "ros2 run nav_filters nav_node --ros-args "
                              "-p init_x:=" +
                              std::to_string(msg->x) + " "
                                                       "-p init_y:=" +
                              std::to_string(msg->y) + " "
                                                       "-p init_z:=" +
                              std::to_string(msg->z) + " &";

            // again, send the success through the topic.
            int status = std::system(cmd.c_str());
            std::cout << status << std::endl;
            std_msgs::msg::Int32 msg;
            msg.data = WEXITSTATUS(status);
            this->revival_status_publisher->publish(msg);
        }
    }

public:
    /**
     * @brief Construct a new Auto Publisher object.
     *
     * @deprecated
     * @warning DO NOT USE THIS NODE. USING A SERVICE, OR ALTERNATIVE NODE IS RECOMMENDED. OR FIX THE BUG!
     */
    AutoPublisher() : Node("auto_pub_node")
    {
        this->start_nav_subscriber = this->create_subscription<geometry_msgs::msg::Vector3>(
            "/start_navigation",
            10,
            std::bind(&AutoPublisher::start_navigation_callback, this, std::placeholders::_1));

        this->revival_status_publisher = this->create_publisher<std_msgs::msg::Int32>(
            "/nav_node_successful",
            10);
    }
};

#endif