#include "dataModel.hpp"
#include "ros_cav.h"

#include <algorithm>
#include <cstdint>

namespace
{
uint16_t read_u16_le(const std::array<uint8_t, 8UL>& data, std::size_t offset)
{
    return static_cast<uint16_t>(
        static_cast<uint16_t>(data[offset]) |
        (static_cast<uint16_t>(data[offset + 1]) << 8U));
}
}  // namespace

void ROSNode::decode_can_0X4D1(const std::array<uint8_t, 8UL>& data)
{
    can_output.speed =
        (static_cast<double>(read_u16_le(data, 0)) - 300.0) * 0.01;

    can_output.left_rpm =
        (static_cast<double>(read_u16_le(data, 2)) - 1200.0) * 0.1;

    can_output.right_rpm =
        (static_cast<double>(read_u16_le(data, 4)) - 1200.0) * 0.1;
    // printf("can_output.left_rad = %.2f\n",
    //                 can_output.left_rpm*2*3.14/60.0);

    // printf("can_output.right_rad = %.2f\n",
    //                 can_output.right_rpm*2*3.14/60.0);
    can_output.soc = static_cast<double>(read_u16_le(data, 6));
    if (can_output.soc > 100.0)
    {
        RCLCPP_WARN_THROTTLE(
            this->get_logger(), *(this->get_clock()), 2000,
            "Received SOC %.1f > 100; clamped to 100", can_output.soc);
        can_output.soc = 100.0;
    }

    have_wheel_feedback = true;

    // Publish the same freshly decoded wheel feedback immediately.
    // The plotting/recording node subscribes to this topic, so feedback
    // no longer waits for the next RTK/GPS VehicleState publication.
    if (pub_wheel_feedback)
    {
        cav_msgs::msg::VehicleState wheel_msg;
        wheel_msg.timestamp = this->get_clock()->now().seconds();
        wheel_msg.left_drive_wheel_rpm = can_output.left_rpm;
        wheel_msg.right_drive_wheel_rpm = can_output.right_rpm;
        wheel_msg.left_drive_wheel_speed = can_output.left_rpm * 2.0 * PI / 60.0;
        wheel_msg.right_drive_wheel_speed = can_output.right_rpm * 2.0 * PI / 60.0;
        pub_wheel_feedback->publish(wheel_msg);
    }
}

void ROSNode::decode_can_0X4D2(const std::array<uint8_t, 8UL>& data)
{
    can_output.drive_state = static_cast<int>(data[0]);
    if (can_output.drive_state != 0 &&
        can_output.drive_state != 1 &&
        can_output.drive_state != 2)
    {
        RCLCPP_WARN_THROTTLE(
            this->get_logger(), *(this->get_clock()), 2000,
            "Invalid drive_state %d; using 0", can_output.drive_state);
        can_output.drive_state = 0;
    }

    can_output.track_angle = static_cast<double>(data[1]) - 90.0;
    if (can_output.track_angle < -50.0 || can_output.track_angle > 80.0)
    {
        RCLCPP_WARN_THROTTLE(
            this->get_logger(), *(this->get_clock()), 2000,
            "Track angle %.1f deg is outside [-50, 80]",
            can_output.track_angle);
    }

    can_output.contol_mode = static_cast<int>(data[2]);
    if (can_output.contol_mode != 0 && can_output.contol_mode != 1)
    {
        RCLCPP_WARN_THROTTLE(
            this->get_logger(), *(this->get_clock()), 2000,
            "Invalid control mode %d; using remote mode 0",
            can_output.contol_mode);
        can_output.contol_mode = 0;
    }

    can_output.yaw_rate =
        -(static_cast<double>(read_u16_le(data, 3)) - 1000.0) * 0.01;

    can_output.size_x = static_cast<double>(data[5]) * 0.01;
    can_output.size_y = static_cast<double>(data[6]) * 0.01;
    can_output.size_z = static_cast<double>(data[7]) * 0.01;
}

void ROSNode::decode_can_0X4D3(const std::array<uint8_t, 8UL>& data)
{
    can_output.bug_code = static_cast<int>(data[0]);
    can_output.left_motor_bug_code = static_cast<int>(data[1]);
    can_output.right_motor_bug_code = static_cast<int>(data[2]);

    if ((can_output.bug_code |
         can_output.left_motor_bug_code |
         can_output.right_motor_bug_code) != 0)
    {
        RCLCPP_WARN_THROTTLE(
            this->get_logger(), *(this->get_clock()), 2000,
            "Vehicle fault: vehicle=%d, left_motor=%d, right_motor=%d",
            can_output.bug_code,
            can_output.left_motor_bug_code,
            can_output.right_motor_bug_code);
    }
}

void ROSNode::decode_can_0X4D4(const std::array<uint8_t, 8UL>& data)
{
    can_output.steer_angle =
        (static_cast<double>(read_u16_le(data, 0)) - 1000.0) *
        0.1 * PI / 180.0;
}

void ROSNode::can2pc_CB(const can_msgs::msg::Frame::SharedPtr msg)
{
    if (msg->dlc < 8)
    {
        RCLCPP_WARN_THROTTLE(
            this->get_logger(), *(this->get_clock()), 2000,
            "Ignoring CAN ID 0x%X with DLC %u; expected 8",
            msg->id, static_cast<unsigned int>(msg->dlc));
        return;
    }

    switch (msg->id)
    {
        case 0x4D1:
            decode_can_0X4D1(msg->data);
            RCLCPP_DEBUG(
                this->get_logger(),
                "0x4D1 speed=%.2f m/s left=%.2f rpm right=%.2f rpm SOC=%.1f%%",
                can_output.speed,
                can_output.left_rpm,
                can_output.right_rpm,
                can_output.soc);
            break;

        case 0x4D2:
            decode_can_0X4D2(msg->data);
            RCLCPP_DEBUG(
                this->get_logger(),
                "0x4D2 state=%d track=%.1f deg mode=%d yaw_rate=%.2f rad/s",
                can_output.drive_state,
                can_output.track_angle,
                can_output.contol_mode,
                can_output.yaw_rate);
            break;

        case 0x4D3:
            decode_can_0X4D3(msg->data);
            RCLCPP_DEBUG(
                this->get_logger(),
                "0x4D3 fault=%d left=%d right=%d",
                can_output.bug_code,
                can_output.left_motor_bug_code,
                can_output.right_motor_bug_code);
            break;

        case 0x4D4:
            decode_can_0X4D4(msg->data);
            RCLCPP_DEBUG(
                this->get_logger(),
                "0x4D4 steer_angle=%.4f rad",
                can_output.steer_angle);
            break;

        default:
            break;
    }
}
