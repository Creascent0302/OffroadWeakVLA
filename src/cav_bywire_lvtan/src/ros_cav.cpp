#include "ros_cav.h"

#include <chrono>
#include <functional>

ROSNode::ROSNode()
: Node("lvtan_bywire_node")
{
    // Publishers are created before the timer starts.
    pub_vs = this->create_publisher<cav_msgs::msg::VehicleState>(
        "/vehicle/vehicle_state", 2);

    // Immediate wheel feedback published directly from CAN 0x4D1.
    // This avoids waiting for the RTK callback to republish VehicleState.
    pub_wheel_feedback = this->create_publisher<cav_msgs::msg::VehicleState>(
        "/vehicle/wheel_feedback", 20);

    pub_can_cmd = this->create_publisher<can_msgs::msg::Frame>(
        "/to_can_bus", 2);

    // Control command from upper controller.
    sub_control2bywire_msg = this->create_subscription<cav_msgs::msg::Control>(
        "/vehicle/control2bywire", 2,
        std::bind(&ROSNode::control2bywire_CB, this, std::placeholders::_1));

    // Legacy path-plan input, retained for the 0x4C1/0x4C3-0x4CF protocol.
    sub_traj_plan_msg = this->create_subscription<cav_msgs::msg::PlanedPath>(
        "/cav_path_plan/result", 2,
        std::bind(&ROSNode::trajPlan_CB, this, std::placeholders::_1));

    // CAN feedback from vehicle.
    sub_can_feedback_msg = this->create_subscription<can_msgs::msg::Frame>(
        "/from_can_bus", 20,
        std::bind(&ROSNode::can2pc_CB, this, std::placeholders::_1));

    // RTK data.
    sub_rtk = this->create_subscription<cav_msgs::msg::Gpybm>(
        "/car/gps", 10,
        std::bind(&ROSNode::rtk_CallBack, this, std::placeholders::_1));

    // Fixed-rate CAN transmission. This also makes loss_control_num effective.
    const auto period = std::chrono::milliseconds(1000 / FREQ);
    can_command_timer = this->create_wall_timer(
        period, std::bind(&ROSNode::publish_can_commands, this));
}
