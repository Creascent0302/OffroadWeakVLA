#ifndef DATAMODEL_HPP
#define DATAMODEL_HPP

#include <cmath>
#include <vector>

// Node/CAN update frequency.
#define FREQ (50)

// Vehicle geometry.
#define LYNX_STEER_RATIO  (17.0)
#define LYNX_L_WHEELS     (1.94)   // m
#define LYNX_WHEEL_RADIUS (0.292)  // m
#define LYNX_HALF_TRACK   (0.557)  // m

#ifndef PI
#define PI (3.14159265358979323846)
#endif

#ifndef PI_L
#define PI_L (3141592L)
#endif

#ifndef RADIANS_PER_DEGREE
#define RADIANS_PER_DEGREE (PI / 180.0)
#endif

#ifndef DEGREES_PER_RADIAN
#define DEGREES_PER_RADIAN (180.0 / PI)
#endif

#ifndef DEG_TO_RAD
#define DEG_TO_RAD(x) (static_cast<double>(x) * PI / 180.0)
#endif

#ifndef RAD_TO_DEG
#define RAD_TO_DEG(x) (static_cast<double>(x) * 180.0 / PI)
#endif

// Define gears.
#define GEAR_NONE    (0)
#define GEAR_PARK    (1)
#define GEAR_REVERSE (2)  // 原地转向状态暂映射为 REVERSE
#define GEAR_NEUTRAL (3)
#define GEAR_DRIVE   (4)
#define GEAR_LOW     (5)

// Define turn signal.
#define TURN_SIGNAL_NONE  (0)
#define TURN_SIGNAL_LEFT  (1)
#define TURN_SIGNAL_RIGHT (2)

namespace XM
{
inline double Normalise_PI(double angle)
{
    if (!std::isfinite(angle))
    {
        return 0.0;
    }

    angle = std::fmod(angle + PI, 2.0 * PI);
    if (angle < 0.0)
    {
        angle += 2.0 * PI;
    }
    return angle - PI;
}

inline double Normalise_2PI(double angle)
{
    if (!std::isfinite(angle))
    {
        return 0.0;
    }

    angle = std::fmod(angle, 2.0 * PI);
    if (angle < 0.0)
    {
        angle += 2.0 * PI;
    }
    return angle;
}
}  // namespace XM

struct CAN_INPUT_S
{


    // Desired direct drive-wheel angular speed from the controller.
    double left_drive_wheel_speed_cmd = 0.0;   // rad/s
    double right_drive_wheel_speed_cmd = 0.0;  // rad/s

    // Safety-checked command actually encoded to CAN.
    double left_drive_wheel_speed_can_cmd = 0.0;   // rad/s
    double right_drive_wheel_speed_can_cmd = 0.0;  // rad/s

    int Stop = 1;
    int Go = 0;

    double RTK_lat = 0.0;  // degree
    double RTK_lon = 0.0;  // degree

    double local_x = 0.0;  // m
    double local_y = 0.0;  // m
    double yaw = 0.0;      // rad

    // Sampled trajectory points.
    std::vector<double> x_list;
    std::vector<double> y_list;
    std::vector<double> vd_list;
    std::vector<double> R_list;

    double ey = 0.0;
    double ephi = 0.0;

    // Incremented by the 50 Hz CAN timer and reset by each control callback.
    int loss_control_num = 0;
};

struct CAN_OUTPUT_S
{
    double speed = 0.0;     // m/s
    double yaw_rate = 0.0;  // rad/s

    double size_x = 0.0;  // m
    double size_y = 0.0;  // m
    double size_z = 0.0;  // m

    double track_angle = 0.0;  // degree, according to CAN protocol
    double steer_angle = 0.0;  // rad

    double left_rpm = 0.0;   // rpm
    double right_rpm = 0.0;  // rpm
    double soc = 0.0;        // %

    int drive_state = 0;  // 0: differential drive, 1: brake, 2: pivot turn

    // Keep the original misspelled member name for compatibility with existing code.
    int contol_mode = 0;  // 1: auto, 0: remote

    int bug_code = 0;
    int left_motor_bug_code = 0;
    int right_motor_bug_code = 0;
};

#endif  // DATAMODEL_HPP
