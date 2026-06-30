#include "UTM.h"
#include "ros_cav.h"

#include <cmath>

namespace
{
int driveMode2Gear(int drive_mode)
{
    switch (drive_mode)
    {
        case 0:
            return GEAR_DRIVE;
        case 1:
            return GEAR_PARK;
        case 2:
            return GEAR_REVERSE;
        default:
            return GEAR_NONE;
    }
}
}  // namespace

void ROSNode::rtk_CallBack(const cav_msgs::msg::Gpybm::SharedPtr msg)
{
    if (!std::isfinite(msg->latitude) ||
        !std::isfinite(msg->longitude) ||
        !std::isfinite(msg->height) ||
        !std::isfinite(msg->yaw) ||
        !std::isfinite(msg->pitch))
    {
        RCLCPP_WARN_THROTTLE(
            this->get_logger(), *(this->get_clock()), 2000,
            "Ignoring non-finite RTK data");
        return;
    }

    // Keep the original deployment-area latitude guard.
    if (msg->latitude < 30.0 || msg->latitude > 50.0 ||
        msg->longitude < -180.0 || msg->longitude > 180.0)
    {
        RCLCPP_WARN_THROTTLE(
            this->get_logger(), *(this->get_clock()), 2000,
            "Ignoring invalid/out-of-area RTK position: lat=%.8f lon=%.8f",
            msg->latitude, msg->longitude);
        return;
    }

    double veh_x = 0.0;
    double veh_y = 0.0;
    UTM::LLtoUTM(msg->latitude, msg->longitude, veh_y, veh_x);

    // Reject a sudden RTK jump after the first valid position.
    if (vs_msg.rtk_gps_utm_x > 100.0 &&
        (std::abs(veh_x - vs_msg.rtk_gps_utm_x) > 5.0 ||
         std::abs(veh_y - vs_msg.rtk_gps_utm_y) > 5.0))
    {
        RCLCPP_WARN_THROTTLE(
            this->get_logger(), *(this->get_clock()), 2000,
            "Ignoring RTK jump larger than 5 m");
        return;
    }

    vs_msg.timestamp = this->get_clock()->now().seconds();
    vs_msg.id = veh_id;
    vs_msg.name = veh_name;
    vs_msg.type = veh_type;

    // Current CAN state must be copied before vehicle_state is evaluated.
    vs_msg.battery_soc = can_output.soc;
    vs_msg.battery_volt = 0.0;
    vs_msg.vehicle_bug_code = can_output.bug_code;

    const bool has_fault =
        can_output.bug_code != 0 ||
        can_output.left_motor_bug_code != 0 ||
        can_output.right_motor_bug_code != 0;

    if (has_fault)
    {
        vs_msg.vehicle_state = 1;  // damage/fault
    }
    else if (vs_msg.battery_soc <= 30.0)
    {
        vs_msg.vehicle_state = 2;  // low battery
    }
    else
    {
        vs_msg.vehicle_state = 0;  // normal
    }

    // RTK state.
    vs_msg.rtk_state_string = "unknown";
    vs_msg.rtk_seq_num = 0;
    vs_msg.rtk_timestamp_sec = 0;
    vs_msg.rtk_timestamp_nsec = 0;
    vs_msg.rtk_gps_status = msg->status;
    vs_msg.rtk_gps_service = msg->status_vice;

    // RTK position.
    vs_msg.rtk_gps_longitude = msg->longitude;
    vs_msg.rtk_gps_latitude = msg->latitude;
    vs_msg.rtk_gps_altitude = msg->height;
    vs_msg.rtk_gps_utm_x = veh_x;
    vs_msg.rtk_gps_utm_y = veh_y;
    vs_msg.rtk_gps_utm_z = msg->height;

    // Key vehicle state.
    vs_msg.x = veh_x;
    vs_msg.y = veh_y;
    vs_msg.z = msg->height;

    vs_msg.speed_x = can_output.speed;
    vs_msg.speed_y = 0.0;
    vs_msg.speed_z = 0.0;

    vs_msg.acc_x = 0.0;
    vs_msg.acc_y = 0.0;
    vs_msg.acc_z = 0.0;

    vs_msg.heading = XM::Normalise_PI(msg->yaw);
    vs_msg.pitch = msg->pitch;
    vs_msg.roll = 0.0;
    vs_msg.yaw_rate = can_output.yaw_rate;

    // RTK velocities.
    vs_msg.rtk_linear_enu_vx = msg->east_vel;
    vs_msg.rtk_linear_enu_vy = msg->earth_vel;
    vs_msg.rtk_linear_enu_vz = 0.0;

    vs_msg.rtk_linear_vx = can_output.speed;
    vs_msg.rtk_linear_vy = 0.0;
    vs_msg.rtk_linear_vz = 0.0;

    vs_msg.rtk_angular_vx = 0.0;
    vs_msg.rtk_angular_vy = 0.0;
    vs_msg.rtk_angular_vz = can_output.yaw_rate;

    // Vehicle dimensions and steering.
    vs_msg.size_x = can_output.size_x;
    vs_msg.size_y = can_output.size_y;
    vs_msg.size_z = can_output.size_z;
    vs_msg.wheelbase = 0.5;
    vs_msg.steer_state_front_wheel = can_output.steer_angle;

    // Drive-wheel feedback.
    vs_msg.left_drive_wheel_rpm = can_output.left_rpm;
    vs_msg.right_drive_wheel_rpm = can_output.right_rpm;
    vs_msg.left_drive_wheel_speed = can_output.left_rpm * 2.0 * PI / 60.0;
    vs_msg.right_drive_wheel_speed = can_output.right_rpm * 2.0 * PI / 60.0;

    // Drive mode.
    vs_msg.gear_pos = driveMode2Gear(can_output.drive_state);
    vs_msg.by_wire_enabled = can_output.contol_mode;
    vs_msg.vehicle_mode = can_output.contol_mode;

    // Move reported sensor position to the configured vehicle reference point.
    if (vs_msg.size_x > 0.1)
    {
        const double dx = vs_msg.size_x / 2.0 - move_x;
        const double dy = move_y;

        vs_msg.x =
            vs_msg.x - std::cos(vs_msg.heading) * dx -
            std::sin(vs_msg.heading) * dy;
        vs_msg.y =
            vs_msg.y - std::sin(vs_msg.heading) * dx +
            std::cos(vs_msg.heading) * dy;
    }

    // Keep outgoing CAN pose fields current even when no new path message arrives.
    can_input.RTK_lat = vs_msg.rtk_gps_latitude;
    can_input.RTK_lon = vs_msg.rtk_gps_longitude;
    can_input.local_x = vs_msg.x;
    can_input.local_y = vs_msg.y;
    can_input.yaw = vs_msg.heading;

    pub_vs->publish(vs_msg);
}
