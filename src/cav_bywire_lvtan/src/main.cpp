#include <memory>
#include <string>

#include <rclcpp/rclcpp.hpp>

#include "ros_cav.h"

int main(int argc, char* argv[])
{
    rclcpp::init(argc, argv);

    auto node = std::make_shared<ROSNode>();

    node->declare_parameter<int>("veh_id", 11710000);
    node->declare_parameter<int>("veh_type", 71);
    node->declare_parameter<std::string>("veh_name", "unspecified_name");
    node->declare_parameter<double>("move_x", 0.3);
    node->declare_parameter<double>("move_y", 0.3);
    node->declare_parameter<std::string>("can_control_mode", "direct_wheel_rpm");

    node->get_parameter("veh_id", node->veh_id);
    node->get_parameter("veh_type", node->veh_type);
    node->get_parameter("veh_name", node->veh_name);
    node->get_parameter("move_x", node->move_x);
    node->get_parameter("move_y", node->move_y);

    std::string can_control_mode;
    node->get_parameter("can_control_mode", can_control_mode);
    RCLCPP_INFO(node->get_logger(), "CAN 0x4C2 control mode parameter: %s", can_control_mode.c_str());

    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
