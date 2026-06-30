#include "dataModel.hpp"
#include "ros_cav.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace
{
enum class CanControlMode : uint8_t
{
    // 0x4C2 Byte 2~3: left wheel rpm * 10, signed int16, little-endian.
    // 0x4C2 Byte 4~5: right wheel rpm * 10, signed int16, little-endian.
    DirectWheelRpm = 0,

    // 0x4C2 Byte 2~3: equivalent vehicle speed, cm/s, uint16, little-endian.
    // 0x4C2 Byte 4~5: equivalent front-wheel angle,
    //                  raw = angle_deg * 10 + 1000, uint16, little-endian.
    EquivalentSpeedSteer = 1
};

/*
 * Select the 0x4C2 command format here.
 *
 * Direct left/right wheel RPM:
 *     CanControlMode::DirectWheelRpm
 *
 * Equivalent speed + front-wheel angle:
 *     CanControlMode::EquivalentSpeedSteer
 */
// constexpr CanControlMode kCanControlMode =
//     CanControlMode::DirectWheelRpm;

// 这里改模式
constexpr CanControlMode kCanControlMode =
    CanControlMode::DirectWheelRpm;

constexpr double kRpmToRadPerSec =
    (2.0 * PI) / 60.0;

constexpr double kRadPerSecToRpm =
    60.0 / (2.0 * PI);

constexpr double kRpmToRawScale = 10.0;

// Drive-wheel angular-speed software limit, rad/s.
constexpr double kMechanicalWheelSpeedMaxRadS = 47.7;

constexpr double kMaxWheelFeedbackDifferenceRadS = 9.4;
constexpr int kControlLossCycles = 30;
constexpr double kCurvatureEpsilon = 1.0e-6;

// Original equivalent-command protocol limits and scales.
constexpr double kEquivalentSpeedMaxCmS = 300.0;
constexpr double kEquivalentSteerMaxDeg = 90.0;
constexpr double kEquivalentSteerRawOffset = 1000.0;
constexpr double kEquivalentSteerRawScale = 10.0;
constexpr double kEquivalentSpeedEpsilonMps = 1.0e-4;

template <typename T>
T clamp_value(T value, T lo, T hi)
{
    return std::max(lo, std::min(hi, value));
}

int32_t clamp_to_int32(double value)
{
    if (!std::isfinite(value))
    {
        return 0;
    }

    const double rounded = std::round(value);
    const double lo = static_cast<double>(std::numeric_limits<int32_t>::min());
    const double hi = static_cast<double>(std::numeric_limits<int32_t>::max());
    return static_cast<int32_t>(clamp_value(rounded, lo, hi));
}

uint16_t clamp_to_uint16(double value)
{
    if (!std::isfinite(value))
    {
        return 0;
    }
    return static_cast<uint16_t>(
        clamp_value(std::round(value), 0.0, 65535.0));
}

int16_t clamp_to_int16(double value)
{
    if (!std::isfinite(value))
    {
        return 0;
    }

    return static_cast<int16_t>(
        clamp_value(
            std::round(value),
            static_cast<double>(std::numeric_limits<int16_t>::min()),
            static_cast<double>(std::numeric_limits<int16_t>::max())));
}

uint32_t clamp_to_uint24(double value)
{
    if (!std::isfinite(value))
    {
        return 0;
    }

    return static_cast<uint32_t>(
        clamp_value(std::round(value), 0.0, 16777215.0));
}

void write_u16_le(std::array<uint8_t, 8UL>& data, std::size_t offset, uint16_t value)
{
    data[offset] = static_cast<uint8_t>(value & 0xFFU);
    data[offset + 1] = static_cast<uint8_t>((value >> 8U) & 0xFFU);
}

void write_i16_le(std::array<uint8_t, 8UL>& data, std::size_t offset, int16_t value)
{
    write_u16_le(data, offset, static_cast<uint16_t>(value));
}

void write_i32_le(std::array<uint8_t, 8UL>& data, std::size_t offset, int32_t value)
{
    const uint32_t raw = static_cast<uint32_t>(value);
    data[offset] = static_cast<uint8_t>(raw & 0xFFU);
    data[offset + 1] = static_cast<uint8_t>((raw >> 8U) & 0xFFU);
    data[offset + 2] = static_cast<uint8_t>((raw >> 16U) & 0xFFU);
    data[offset + 3] = static_cast<uint8_t>((raw >> 24U) & 0xFFU);
}

void write_u24_le(std::array<uint8_t, 8UL>& data, std::size_t offset, uint32_t value)
{
    value &= 0x00FFFFFFU;
    data[offset] = static_cast<uint8_t>(value & 0xFFU);
    data[offset + 1] = static_cast<uint8_t>((value >> 8U) & 0xFFU);
    data[offset + 2] = static_cast<uint8_t>((value >> 16U) & 0xFFU);
}

int16_t wheel_rad_s_to_rpm_x10(double wheel_speed_rad_s)
{
    if (!std::isfinite(wheel_speed_rad_s))
    {
        return 0;
    }

    const double rpm =
        wheel_speed_rad_s * kRadPerSecToRpm;

    const long raw = std::lround(
        rpm * kRpmToRawScale);

    const long limited_raw = std::clamp(
        raw,
        static_cast<long>(
            std::numeric_limits<int16_t>::min()),
        static_cast<long>(
            std::numeric_limits<int16_t>::max()));

    return static_cast<int16_t>(limited_raw);
}


struct EquivalentVehicleCommand
{
    // Physical values used by the lower-controller protocol.
    double speed_cm_s = 0.0;
    double steer_deg = 0.0;
};

EquivalentVehicleCommand wheel_rad_s_to_equivalent_command(
    double left_wheel_rad_s,
    double right_wheel_rad_s)
{
    EquivalentVehicleCommand command;

    if (!std::isfinite(left_wheel_rad_s) ||
        !std::isfinite(right_wheel_rad_s))
    {
        return command;
    }

    // Drive-wheel angular speed, rad/s -> wheel linear speed, m/s.
    const double left_linear_mps =
        left_wheel_rad_s * LYNX_WHEEL_RADIUS;

    const double right_linear_mps =
        right_wheel_rad_s * LYNX_WHEEL_RADIUS;

    // Differential-drive equivalent longitudinal speed.
    const double speed_mps =
        0.5 * (left_linear_mps + right_linear_mps);

    // Differential-drive yaw rate.
    // Full track width = 2 * LYNX_HALF_TRACK.
    const double yaw_rate_rad_s =
        (right_linear_mps - left_linear_mps) /
        (2.0 * LYNX_HALF_TRACK);

    /*
     * The equivalent protocol has an unsigned forward-speed field.
     * Reverse motion and pivot turns cannot be represented accurately.
     * For those cases, output zero speed and zero steering.
     */
    if (!std::isfinite(speed_mps) ||
        !std::isfinite(yaw_rate_rad_s) ||
        speed_mps <= kEquivalentSpeedEpsilonMps)
    {
        return command;
    }

    /*
     * Equivalent bicycle-model steering:
     *
     * yaw_rate = speed * tan(delta) / wheelbase
     *
     * delta = atan(wheelbase * yaw_rate / speed)
     */
    const double steer_rad =
        std::atan(
            LYNX_L_WHEELS *
            yaw_rate_rad_s /
            speed_mps);

    command.speed_cm_s =
        clamp_value(
            speed_mps * 100.0,
            0.0,
            kEquivalentSpeedMaxCmS);

    command.steer_deg =
        clamp_value(
            steer_rad * 180.0 / PI,
            -kEquivalentSteerMaxDeg,
            kEquivalentSteerMaxDeg);

    return command;
}

void encode_direct_wheel_rpm_payload(
    std::array<uint8_t, 8UL>& data,
    double left_wheel_rad_s,
    double right_wheel_rad_s)
{
    const int16_t left_raw =
        wheel_rad_s_to_rpm_x10(
            left_wheel_rad_s);

    const int16_t right_raw =
        wheel_rad_s_to_rpm_x10(
            right_wheel_rad_s);

    // Byte 2~3: left wheel rpm*10, signed int16, little-endian.
    write_i16_le(data, 2, left_raw);

    // Byte 4~5: right wheel rpm*10, signed int16, little-endian.
    write_i16_le(data, 4, right_raw);
}

EquivalentVehicleCommand encode_equivalent_speed_steer_payload(
    std::array<uint8_t, 8UL>& data,
    double left_wheel_rad_s,
    double right_wheel_rad_s)
{
    const EquivalentVehicleCommand command =
        wheel_rad_s_to_equivalent_command(
            left_wheel_rad_s,
            right_wheel_rad_s);

    /*
     * Byte 2~3: speed in cm/s.
     *
     * Example:
     *     300 cm/s -> raw 300 -> 0x012C -> 2C 01
     */
    const uint16_t speed_raw =
        clamp_to_uint16(command.speed_cm_s);

    /*
     * Byte 4~5: front-wheel angle.
     *
     * Physical unit: degree.
     * Resolution: 0.1 degree/bit.
     * Offset: +1000.
     *
     * raw = steer_deg * 10 + 1000
     *
     *   0 deg -> 1000 -> 0x03E8
     * +12 deg -> 1120 -> 0x0460
     * -12 deg ->  880 -> 0x0370
     */
    const uint16_t steer_raw =
        clamp_to_uint16(
            command.steer_deg *
                kEquivalentSteerRawScale +
            kEquivalentSteerRawOffset);

    write_u16_le(data, 2, speed_raw);
    write_u16_le(data, 4, steer_raw);

    return command;
}


can_msgs::msg::Frame make_frame(uint32_t id, const std::array<uint8_t, 8UL>& data)
{
    can_msgs::msg::Frame frame;
    frame.id = id;
    frame.dlc = 8;
    frame.data = data;
    return frame;
}

double curvature_to_radius(double curvature)
{
    if (!std::isfinite(curvature) || std::abs(curvature) < kCurvatureEpsilon)
    {
        return 100.0;
    }

    return clamp_value(1.0 / curvature, -100.0, 100.0);
}
}  // namespace

int gear_from_standard_to_lynx(int gear_cmd)
{
    if (gear_cmd == GEAR_LOW)
    {
        return 1;
    }
    if (gear_cmd == GEAR_DRIVE)
    {
        return 0;
    }
    return 0;
}

void ROSNode::control2bywire_CB(
    const cav_msgs::msg::Control::SharedPtr msg)
{
    // A fresh upper-controller command has arrived.
    can_input.loss_control_num = 0;

    // The Python controller sets these fields to a stop state when:
    //   1. no valid reference path is available;
    //   2. no vehicle state has been received;
    //   3. controller calculation fails;
    //   4. by-wire control is disabled.
    const bool stop_requested =
        (msg->bywire_control_enable == 0) ||
        (msg->emerg_brake != 0) ||
        (msg->park_enable != 0);

    if (stop_requested)
    {
        can_input.Stop = 1;
        can_input.Go = 0;

        can_input.left_drive_wheel_speed_cmd = 0.0;
        can_input.right_drive_wheel_speed_cmd = 0.0;
        return;
    }

    // Valid path and valid controller output: allow vehicle motion.
    can_input.Stop = 0;
    can_input.Go = 1;

    // Direct left/right drive-wheel angular-speed commands, rad/s.
    can_input.left_drive_wheel_speed_cmd =
        msg->left_drive_wheel_speed_cmd;

    can_input.right_drive_wheel_speed_cmd =
        msg->right_drive_wheel_speed_cmd;
}

void ROSNode::trajPlan_CB(const cav_msgs::msg::PlanedPath::SharedPtr msg)
{
    // This legacy topic only supplies the auxiliary path data used by
    // 0x4C4-0x4CF. Stop/Go is controlled exclusively by Control.msg,
    // so an old or delayed PlanedPath message cannot overwrite the
    // safety state selected by the current Python controller.

    can_input.x_list.clear();
    can_input.y_list.clear();
    can_input.vd_list.clear();
    can_input.R_list.clear();

    can_input.x_list.reserve(10);
    can_input.y_list.reserve(10);
    can_input.vd_list.reserve(10);
    can_input.R_list.reserve(10);

    for (std::size_t i = 4; i < msg->planed_path.size() &&
                            can_input.x_list.size() < 10; i += 5)
    {
        const auto& point = msg->planed_path[i];
        can_input.x_list.push_back(point.x);
        can_input.y_list.push_back(point.y);
        can_input.vd_list.push_back(point.v);
        can_input.R_list.push_back(curvature_to_radius(point.cr));
    }
}

void ROSNode::encode_4C1()
{
    std::array<uint8_t, 8UL> data{};

    const int32_t lon_raw = clamp_to_int32(can_input.RTK_lon * 1.0e7);
    const int32_t lat_raw = clamp_to_int32(can_input.RTK_lat * 1.0e7);

    write_i32_le(data, 0, lon_raw);
    write_i32_le(data, 4, lat_raw);

    can_cmd_msgs.push_back(make_frame(0x4C1U, data));
}

void ROSNode::encode_4C2()
{
    std::array<uint8_t, 8UL> data{};

    // Byte 0: stop/brake level.
    data[0] =
        static_cast<uint8_t>(
            clamp_value(
                can_input.Stop,
                0,
                3));

    // Byte 1: start flag.
    data[1] =
        static_cast<uint8_t>(
            can_input.Go != 0);

    const bool motion_enabled =
        (can_input.Stop == 0) &&
        (can_input.Go != 0);

    if (motion_enabled)
    {
        switch (kCanControlMode)
        {
            case CanControlMode::DirectWheelRpm:
            {
                /*
                 * Direct-wheel mode:
                 *
                 * Byte 2~3: left wheel rpm*10
                 * Byte 4~5: right wheel rpm*10
                 */
                encode_direct_wheel_rpm_payload(
                    data,
                    can_input.left_drive_wheel_speed_can_cmd,
                    can_input.right_drive_wheel_speed_can_cmd);
                printf("can_input.left_drive_wheel_speed_can_cmd = %.2f\n",
                    can_input.left_drive_wheel_speed_can_cmd);

                printf("can_input.right_drive_wheel_speed_can_cmd = %.2f\n",
                    can_input.right_drive_wheel_speed_can_cmd);
                RCLCPP_INFO_THROTTLE(
                    this->get_logger(),
                    *(this->get_clock()),
                    1000,
                    "4C2 mode=DIRECT_RPM: "
                    "left=%.2f rad/s, right=%.2f rad/s",
                    can_input.left_drive_wheel_speed_can_cmd,
                    can_input.right_drive_wheel_speed_can_cmd);

                break;
            }

            case CanControlMode::EquivalentSpeedSteer:
            {
                /*
                 * Equivalent-command mode:
                 *
                 * Byte 2~3: equivalent speed, cm/s
                 * Byte 4~5: front-wheel angle,
                 *           raw = degree*10 + 1000
                 */
                const EquivalentVehicleCommand command =
                    encode_equivalent_speed_steer_payload(
                        data,
                        can_input.left_drive_wheel_speed_can_cmd,
                        can_input.right_drive_wheel_speed_can_cmd);

                RCLCPP_INFO_THROTTLE(
                    this->get_logger(),
                    *(this->get_clock()),
                    1000,
                    "4C2 mode=EQUIVALENT: "
                    "speed=%.1f cm/s, steer=%.2f deg, "
                    "source wheels=(%.2f, %.2f) rad/s",
                    command.speed_cm_s,
                    command.steer_deg,
                    can_input.left_drive_wheel_speed_can_cmd,
                    can_input.right_drive_wheel_speed_can_cmd);

                break;
            }

            default:
            {
                // Unknown mode: keep Byte 2~7 equal to zero.
                break;
            }
        }
    }

    /*
     * If Stop is active or Go is false, Byte 2~7 remain zero:
     *
     *     01 00 00 00 00 00 00 00
     */
    // Byte 6: fixed integer flag, downlink value = 1.
    data[6] = static_cast<uint8_t>(1);
    data[7] = 0;

    can_cmd_msgs.push_back(
        make_frame(0x4C2U, data));
}


void ROSNode::encode_4C3()
{
    std::array<uint8_t, 8UL> data{};

    const uint32_t x_raw = clamp_to_uint24(can_input.local_x * 100.0 + 8000000.0);
    const uint32_t y_raw = clamp_to_uint24(can_input.local_y * 100.0 + 8000000.0);
    const uint16_t yaw_raw = clamp_to_uint16(
        XM::Normalise_PI(can_input.yaw) * 1000.0 + 5000.0);

    write_u24_le(data, 0, x_raw);
    write_u24_le(data, 3, y_raw);
    write_u16_le(data, 6, yaw_raw);

    can_cmd_msgs.push_back(make_frame(0x4C3U, data));
}

void ROSNode::encode_4C4_4CD()
{
    for (std::size_t i = 0; i < 10; ++i)
    {
        std::array<uint8_t, 8UL> data{};

        if (i < can_input.x_list.size() &&
            i < can_input.y_list.size() &&
            i < can_input.vd_list.size())
        {
            const uint32_t x_raw =
                clamp_to_uint24(can_input.x_list[i] * 100.0 + 8000000.0);
            const uint32_t y_raw =
                clamp_to_uint24(can_input.y_list[i] * 100.0 + 8000000.0);
            const uint16_t speed_raw =
                clamp_to_uint16(can_input.vd_list[i] * 100.0);

            write_u24_le(data, 0, x_raw);
            write_u24_le(data, 3, y_raw);
            write_u16_le(data, 6, speed_raw);
        }

        can_cmd_msgs.push_back(
            make_frame(static_cast<uint32_t>(0x4C4U + i), data));
    }
}

void ROSNode::encode_4CE_4CF()
{
    std::array<uint8_t, 8UL> data_4ce{};
    std::array<uint8_t, 8UL> data_4cf{};

    for (std::size_t i = 0; i < 4; ++i)
    {
        if (i < can_input.R_list.size())
        {
            write_i16_le(
                data_4ce, 2 * i,
                clamp_to_int16(can_input.R_list[i] * 100.0));
        }

        const std::size_t second_index = i + 4;
        if (second_index < can_input.R_list.size())
        {
            write_i16_le(
                data_4cf, 2 * i,
                clamp_to_int16(can_input.R_list[second_index] * 100.0));
        }
    }

    can_cmd_msgs.push_back(make_frame(0x4CEU, data_4ce));
    can_cmd_msgs.push_back(make_frame(0x4CFU, data_4cf));
}

void ROSNode::encode_4C0()
{
    std::array<uint8_t, 8UL> data{};

    for (std::size_t i = 0; i < 2; ++i)
    {
        const std::size_t radius_index = i + 8;
        if (radius_index < can_input.R_list.size())
        {
            write_i16_le(
                data, 2 * i,
                clamp_to_int16(can_input.R_list[radius_index] * 100.0));
        }
    }

    // Correct ID: the original code accidentally sent another 0x4CE frame.
    can_cmd_msgs.push_back(make_frame(0x4C0U, data));
}

void ROSNode::set_cmd_can_msg()
{
    can_cmd_msgs.clear();
    can_cmd_msgs.reserve(16);

    encode_4C1();
    encode_4C2();
    encode_4C3();
    encode_4C4_4CD();
    encode_4CE_4CF();
    encode_4C0();
}

void ROSNode::can_cmd_safety_check()
{
    double left_target = can_input.left_drive_wheel_speed_cmd;
    double right_target = can_input.right_drive_wheel_speed_cmd;

    if (!std::isfinite(left_target))
    {
        left_target = 0.0;
    }
    if (!std::isfinite(right_target))
    {
        right_target = 0.0;
    }

    const double min_wheel_speed =
        -kMechanicalWheelSpeedMaxRadS;
    const double max_wheel_speed =
        kMechanicalWheelSpeedMaxRadS;

    const double left_before_clip = left_target;
    const double right_before_clip = right_target;

    left_target = clamp_value(
        left_target,
        min_wheel_speed,
        max_wheel_speed);
    right_target = clamp_value(
        right_target,
        min_wheel_speed,
        max_wheel_speed);

    if (left_before_clip != left_target ||
        right_before_clip != right_target)
    {
        RCLCPP_WARN_THROTTLE(
            this->get_logger(), *(this->get_clock()), 2000,
            "Wheel command clipped to mechanical range: "
            "left %.2f -> %.2f rad/s, right %.2f -> %.2f rad/s",
            left_before_clip, left_target,
            right_before_clip, right_target);
    }

    const bool control_timeout =
        can_input.loss_control_num > kControlLossCycles;

    if (control_timeout)
    {
        // No fresh Control message for about 0.6 s at 50 Hz.
        can_input.Stop = 1;
        can_input.Go = 0;

        RCLCPP_WARN_THROTTLE(
            this->get_logger(), *(this->get_clock()), 2000,
            "Control message timeout: Stop=1, Go=0, "
            "both wheel commands set to 0 rad/s");
    }

    const bool stop_active =
        control_timeout ||
        (can_input.Stop != 0) ||
        (can_input.Go == 0);

    if (stop_active)
    {
        // Important: do not run the feedback-difference limiter while
        // stopping. Otherwise a non-zero measured wheel speed could
        // change a zero stop target back into a non-zero CAN command.
        left_target = 0.0;
        right_target = 0.0;
    }
    else if (have_wheel_feedback)
    {
        // During normal motion, limit the target relative to measured
        // wheel speed to avoid an abrupt command step.
        const double left_feedback_rad_s =
            can_output.left_rpm * kRpmToRadPerSec;
        const double right_feedback_rad_s =
            can_output.right_rpm * kRpmToRadPerSec;

        left_target = clamp_value(
            left_target,
            left_feedback_rad_s -
                kMaxWheelFeedbackDifferenceRadS,
            left_feedback_rad_s +
                kMaxWheelFeedbackDifferenceRadS);

        right_target = clamp_value(
            right_target,
            right_feedback_rad_s -
                kMaxWheelFeedbackDifferenceRadS,
            right_feedback_rad_s +
                kMaxWheelFeedbackDifferenceRadS);
    }

    can_input.left_drive_wheel_speed_can_cmd =
        left_target;
    can_input.right_drive_wheel_speed_can_cmd =
        right_target;
}

void ROSNode::publish_can_commands()
{
    if (can_input.loss_control_num < std::numeric_limits<int>::max())
    {
        ++can_input.loss_control_num;
    }

    can_cmd_safety_check();
    set_cmd_can_msg();

    for (const auto& frame : can_cmd_msgs)
    {
        pub_can_cmd->publish(frame);
    }
}
