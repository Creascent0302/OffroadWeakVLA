#ifndef ROSNODE_H
#define ROSNODE_H

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>

#include <can_msgs/msg/frame.hpp>
#include "cav_msgs/msg/control.hpp"
#include "cav_msgs/msg/gpybm.hpp"
#include "cav_msgs/msg/planed_path.hpp"
#include "cav_msgs/msg/vehicle_state.hpp"

#include "dataModel.hpp"

class ROSNode : public rclcpp::Node
{
public:
    ROSNode();
    ~ROSNode() override = default;

    // CAN data.
    CAN_INPUT_S can_input;
    CAN_OUTPUT_S can_output;

    // Receive control/path and publish CAN commands.
    rclcpp::Subscription<cav_msgs::msg::Control>::SharedPtr sub_control2bywire_msg;
    rclcpp::Subscription<cav_msgs::msg::PlanedPath>::SharedPtr sub_traj_plan_msg;
    rclcpp::Publisher<can_msgs::msg::Frame>::SharedPtr pub_can_cmd;
    rclcpp::TimerBase::SharedPtr can_command_timer;

    void control2bywire_CB(const cav_msgs::msg::Control::SharedPtr msg);
    void trajPlan_CB(const cav_msgs::msg::PlanedPath::SharedPtr msg);

    void can_cmd_safety_check();
    void encode_4C1();
    void encode_4C2();
    void encode_4C3();
    void encode_4C4_4CD();
    void encode_4CE_4CF();
    void encode_4C0();
    void set_cmd_can_msg();
    void publish_can_commands();

    // Receive RTK/CAN feedback and publish vehicle state.
    rclcpp::Subscription<can_msgs::msg::Frame>::SharedPtr sub_can_feedback_msg;
    rclcpp::Subscription<cav_msgs::msg::Gpybm>::SharedPtr sub_rtk;
    rclcpp::Publisher<cav_msgs::msg::VehicleState>::SharedPtr pub_vs;

    cav_msgs::msg::VehicleState vs_msg;
    std::vector<can_msgs::msg::Frame> can_cmd_msgs;

    bool have_wheel_feedback = false;

    void can2pc_CB(const can_msgs::msg::Frame::SharedPtr msg);
    void decode_can_0X4D1(const std::array<uint8_t, 8UL>& data);
    void decode_can_0X4D2(const std::array<uint8_t, 8UL>& data);
    void decode_can_0X4D3(const std::array<uint8_t, 8UL>& data);
    void decode_can_0X4D4(const std::array<uint8_t, 8UL>& data);

    void rtk_CallBack(const cav_msgs::msg::Gpybm::SharedPtr msg);

    // Vehicle settings.
    int veh_id = 11710000;
    int veh_type = 71;
    std::string veh_name = "unspecified_name";
    double move_x = 0.3;
    double move_y = 0.3;
};

#endif  // ROSNODE_H
