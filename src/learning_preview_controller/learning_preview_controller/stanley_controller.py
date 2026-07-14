import json
import math
import traceback

import numpy as np
import rclpy
from rclpy.node import Node

from cav_msgs.msg import Control, VehicleState
from std_msgs.msg import String

from .path_helper import PathOnlyHelper, load_yaml_config
from .start_stop_panel import StartStopPanel


class StanleyControllerNode(Node):
    """Standalone Stanley controller.

    Config file:
        learning_preview_controller/config/stanley_params.yaml

    The Stanley config only needs:
        path:
        stanley:

    EquivalentSpeedSteer mode reuses the existing two command fields:

        left_drive_wheel_speed_cmd  -> desired speed, cm/s
        right_drive_wheel_speed_cmd -> front-wheel steering angle, deg
    """

    def __init__(self):
        super().__init__("stanley_controller")

        self.declare_parameter("vehicle_state_topic", "/vehicle/vehicle_state")
        self.declare_parameter("control_topic", "/vehicle/control2bywire")
        self.declare_parameter("wheel_feedback_topic", "/vehicle/wheel_feedback")
        self.declare_parameter("control_period", 0.05)
        self.declare_parameter("config_file", "")
        self.declare_parameter("gear_cmd", 4)
        self.declare_parameter("vehicle_mode", 1)
        self.declare_parameter("enable_bywire", True)
        self.declare_parameter("enable_start_panel", True)
        self.declare_parameter("start_panel_update_period", 0.20)
        self.declare_parameter("calculate_control_on_start", False)
        self.declare_parameter("publish_plot_samples", True)
        self.declare_parameter("plot_sample_topic", "/learning_preview/plot_sample")

        vehicle_state_topic = str(self.get_parameter("vehicle_state_topic").value)
        control_topic = str(self.get_parameter("control_topic").value)
        wheel_feedback_topic = str(self.get_parameter("wheel_feedback_topic").value)
        self.control_period = float(self.get_parameter("control_period").value)

        config_file = str(self.get_parameter("config_file").value).strip()
        cfg, self.config_file = load_yaml_config(
            config_file if config_file else None,
            default_config_name="stanley_params.yaml",
        )

        path_cfg = cfg.get("path", {}) or {}
        stanley_cfg = cfg.get("stanley", {}) or {}

        self.helper = PathOnlyHelper(path_cfg)
        self.ref_path = self.helper.ref_path
        self.path_cfg = path_cfg

        self.origin_utm_x = float(path_cfg["origin_utm_x"])
        self.origin_utm_y = float(path_cfg["origin_utm_y"])
        self.origin_heading_rad = float(path_cfg["origin_heading_rad"])
        self.use_start_frame_as_origin = self._cfg_bool(
            path_cfg,
            "use_start_frame_as_origin",
            False,
        )
        self.start_origin_capture_pending = False
        self.start_origin_capture_min_sequence = 0
        self.start_origin_has_been_captured = False

        self.k_gain = float(stanley_cfg.get("k_gain", 0.8))
        self.speed_epsilon_mps = abs(float(stanley_cfg.get("speed_epsilon_mps", 0.1)))
        self.max_steer_deg = abs(float(stanley_cfg.get("max_steer_deg", 28.0)))
        self.cross_track_sign = float(stanley_cfg.get("cross_track_sign", 1.0))

        if "target_speed_cm_s" in stanley_cfg:
            self.target_speed_cm_s = float(stanley_cfg["target_speed_cm_s"])
        elif "target_speed_mps" in stanley_cfg:
            self.target_speed_cm_s = float(stanley_cfg["target_speed_mps"]) * 100.0
        else:
            self.target_speed_cm_s = float(path_cfg.get("v_ref", 2.0)) * 100.0

        self.target_speed_cm_s = max(0.0, self.target_speed_cm_s)
        self.target_speed_mps = self.target_speed_cm_s / 100.0

        self.state = None
        self.state_sequence = 0
        self.count = 0
        self.last_stop_reason = ""

        self.latest_left_wheel_feedback = math.nan
        self.latest_right_wheel_feedback = math.nan
        self.plot_start_time_sec = self.get_clock().now().nanoseconds * 1.0e-9
        self.calculate_control_enabled = bool(
            self.get_parameter("calculate_control_on_start").value
        )
        self.publish_plot_samples = bool(
            self.get_parameter("publish_plot_samples").value
        )

        plot_sample_topic = str(self.get_parameter("plot_sample_topic").value)
        self.pub_plot_sample = self.create_publisher(String, plot_sample_topic, 50)
        self.pub_cmd = self.create_publisher(Control, control_topic, 2)
        self.sub_state = self.create_subscription(
            VehicleState,
            vehicle_state_topic,
            self.vehicle_state_callback,
            10,
        )
        self.sub_wheel_feedback = self.create_subscription(
            VehicleState,
            wheel_feedback_topic,
            self.wheel_feedback_callback,
            20,
        )

        self.start_panel = None
        if bool(self.get_parameter("enable_start_panel").value):
            try:
                self.start_panel = StartStopPanel(
                    initial_enabled=self.calculate_control_enabled,
                    on_toggle=self.set_calculate_control_enabled,
                    title="Stanley Controller Start / Stop",
                )
            except Exception:
                self.start_panel = None
                self.get_logger().error(
                    "Failed to initialize Stanley Start/Stop panel.\n"
                    + traceback.format_exc()
                )

        self.timer = self.create_timer(self.control_period, self.control_loop)
        self.start_panel_timer = None
        if self.start_panel is not None:
            self.start_panel_timer = self.create_timer(
                max(float(self.get_parameter("start_panel_update_period").value), 0.05),
                self.update_start_panel,
            )

        if self.calculate_control_enabled and self.use_start_frame_as_origin:
            self.request_start_frame_origin_capture()

        self.get_logger().warn(
            "Stanley controller started with Stanley-only config: "
            f"{self.config_file}. "
            "For the by-wire node use can_control_mode=equivalent_speed_steer. "
            "Command mapping: left=speed_cm_s, right=steer_deg. "
            f"target_speed={self.target_speed_cm_s:.1f} cm/s, "
            f"k={self.k_gain:.3f}, max_steer={self.max_steer_deg:.1f} deg."
        )

    @staticmethod
    def _finite(value, default=0.0):
        try:
            value = float(value)
        except (TypeError, ValueError):
            return default
        return value if math.isfinite(value) else default

    @staticmethod
    def _wrap_to_pi(angle):
        return math.atan2(math.sin(angle), math.cos(angle))

    @staticmethod
    def _cfg_bool(config, key, default=False):
        value = config.get(key, default)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return bool(value)

    @staticmethod
    def _json_safe(value):
        if isinstance(value, np.ndarray):
            return StanleyControllerNode._json_safe(value.tolist())
        if isinstance(value, (list, tuple)):
            return [StanleyControllerNode._json_safe(item) for item in value]
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
        return value if math.isfinite(value) else None

    def global_to_path_frame(self, global_x, global_y, global_heading):
        dx = global_x - self.origin_utm_x
        dy = global_y - self.origin_utm_y
        c = math.cos(self.origin_heading_rad)
        s = math.sin(self.origin_heading_rad)
        local_x = c * dx + s * dy
        local_y = -s * dx + c * dy
        local_heading = self._wrap_to_pi(global_heading - self.origin_heading_rad)
        return local_x, local_y, local_heading

    def request_start_frame_origin_capture(self):
        self.start_origin_capture_pending = True
        self.start_origin_capture_min_sequence = self.state_sequence
        self.start_origin_has_been_captured = False
        self.get_logger().info(
            "Stanley start-frame origin capture armed. The next fresh "
            "VehicleState frame becomes local (0, 0, 0)."
        )

    def capture_start_frame_origin(self, global_x, global_y, global_heading):
        self.origin_utm_x = float(global_x)
        self.origin_utm_y = float(global_y)
        self.origin_heading_rad = self._wrap_to_pi(float(global_heading))
        self.start_origin_capture_pending = False
        self.start_origin_has_been_captured = True
        self.helper.ID_last = 0
        self.plot_start_time_sec = self.get_clock().now().nanoseconds * 1.0e-9
        self.publish_plot_reset()
        self.get_logger().warn(
            "Stanley start-frame origin captured: "
            f"UTM x={self.origin_utm_x:.3f}, y={self.origin_utm_y:.3f}, "
            f"heading={self.origin_heading_rad:.6f}."
        )

    def set_calculate_control_enabled(self, enabled):
        enabled = bool(enabled)
        if enabled == self.calculate_control_enabled:
            return
        self.calculate_control_enabled = enabled
        if enabled:
            if self.use_start_frame_as_origin:
                self.request_start_frame_origin_capture()
            self.plot_start_time_sec = self.get_clock().now().nanoseconds * 1.0e-9
            # Clear immediately on Start. Dynamic-origin mode sends a second
            # reset once the first fresh VehicleState frame is captured.
            self.publish_plot_reset()
            self.last_stop_reason = ""
            self.get_logger().warn("Stanley control ENABLED from Start panel.")
        else:
            self.start_origin_capture_pending = False
            self.get_logger().warn("Stanley control DISABLED; publishing safe stop.")
            self.publish_stop("Stanley control calculation is disabled from UI")
        if self.start_panel is not None:
            self.start_panel.set_enabled(self.calculate_control_enabled)
        self.publish_plot_event("control_enabled", enabled=self.calculate_control_enabled)

    def vehicle_state_callback(self, msg):
        self.state = msg
        self.state_sequence += 1

    def wheel_feedback_callback(self, msg):
        left = self._finite(msg.left_drive_wheel_speed, math.nan)
        right = self._finite(msg.right_drive_wheel_speed, math.nan)
        if math.isfinite(left):
            self.latest_left_wheel_feedback = left
        if math.isfinite(right):
            self.latest_right_wheel_feedback = right

    def make_safe_zero_cmd(self):
        cmd = Control()
        cmd.timestamp = self.get_clock().now().nanoseconds * 1.0e-9
        cmd.count = self.count
        cmd.left_drive_wheel_speed_cmd = 0.0
        cmd.right_drive_wheel_speed_cmd = 0.0
        cmd.brake_cmd = 0.0
        cmd.throttle_cmd = 0.0
        cmd.acceleration_cmd = 0.0
        cmd.gear_cmd = int(self.get_parameter("gear_cmd").value)
        cmd.turn_signal_cmd = 0
        cmd.bywire_control_enable = 0
        cmd.emerg_brake = 1
        cmd.park_enable = 1
        cmd.front_light = 0
        cmd.engine_enable = 1
        cmd.vehicle_mode = int(self.get_parameter("vehicle_mode").value)
        cmd.yaw_speed_cmd = 0.0
        cmd.ey_out = 0.0
        cmd.ephi_out = 0.0
        return cmd

    def publish_stop(self, reason):
        if reason != self.last_stop_reason:
            self.get_logger().warn(f"Publishing stop command: {reason}")
            self.last_stop_reason = reason
        self.pub_cmd.publish(self.make_safe_zero_cmd())

    def compute_stanley(self, x, y, psi, measured_speed_mps):
        _, nearest_id = self.helper.FindNearestPoint(
            self.ref_path,
            x,
            y,
            psi,
            int(getattr(self.helper, "ID_last", 0)),
        )
        lateral_error, heading_error, curvature = self.helper.CalErr(
            x,
            y,
            psi,
            nearest_id,
            self.ref_path,
        )
        self.helper.ID_last = nearest_id

        # Basic Stanley:
        #   delta = heading_error + atan(k * lateral_error / speed)
        speed_for_stanley = abs(float(measured_speed_mps))
        if speed_for_stanley < self.speed_epsilon_mps:
            speed_for_stanley = max(self.target_speed_mps, self.speed_epsilon_mps)

        cross_track_term = math.atan2(
            self.k_gain * self.cross_track_sign * float(lateral_error),
            speed_for_stanley,
        )
        steer_rad = float(heading_error) + cross_track_term
        steer_deg = math.degrees(steer_rad)
        steer_deg = max(-self.max_steer_deg, min(self.max_steer_deg, steer_deg))
        steer_rad = math.radians(steer_deg)

        return {
            "nearest_id": nearest_id,
            "lateral_error": float(lateral_error),
            "heading_error": float(heading_error),
            "curvature": float(curvature),
            "speed_cm_s": float(self.target_speed_cm_s),
            "steer_deg": float(steer_deg),
            "steer_rad": float(steer_rad),
            "speed_for_stanley_mps": float(speed_for_stanley),
        }

    def publish_plot_reset(self):
        """Clear plot history and synchronize Stanley path/origin."""
        path_type = str(self.path_cfg.get("type", "straight_circle")).strip().lower()
        self.publish_plot_event(
            "reset",
            origin_utm_x=self._json_safe(self.origin_utm_x),
            origin_utm_y=self._json_safe(self.origin_utm_y),
            origin_heading_rad=self._json_safe(self.origin_heading_rad),
            ref_path=self._json_safe(self.ref_path),
            position_equal_aspect=path_type not in ("sine", "sin"),
        )

    def publish_plot_event(self, event, **extra):
        if not self.publish_plot_samples:
            return
        payload = {
            "event": str(event),
            "ros_time_sec": self.get_clock().now().nanoseconds * 1.0e-9,
            "controller_type": "stanley",
            "config_file": self.config_file,
        }
        payload.update(extra)
        msg = String()
        msg.data = json.dumps(payload, separators=(",", ":"))
        self.pub_plot_sample.publish(msg)

    def append_plot_sample(self, x, y, result, vehicle_speed_mps):
        if not self.publish_plot_samples:
            return
        now_sec = self.get_clock().now().nanoseconds * 1.0e-9
        payload = {
            "event": "sample",
            "controller_type": "stanley",
            "config_file": self.config_file,
            "time_sec": self._json_safe(now_sec - self.plot_start_time_sec),
            "local_x": self._json_safe(x),
            "local_y": self._json_safe(y),
            "lateral_error": self._json_safe(result.get("lateral_error")),
            "heading_error": self._json_safe(result.get("heading_error")),

            # Stanley mode command channels:
            #   left_wheel_cmd  = desired speed, cm/s
            #   right_wheel_cmd = front-wheel angle, deg
            "left_wheel_cmd": self._json_safe(result.get("speed_cm_s")),
            "right_wheel_cmd": self._json_safe(result.get("steer_deg")),
            "left_wheel_feedback": self._json_safe(self.latest_left_wheel_feedback),
            "right_wheel_feedback": self._json_safe(self.latest_right_wheel_feedback),
            "matrix_a": None,
            "matrix_b": None,
            "vehicle_speed": self._json_safe(vehicle_speed_mps),
            "nonlinear_observed_u": None,
            "nonlinear_observed_r": None,
            "nonlinear_estimated_u": self._json_safe(result.get("speed_cm_s")),
            "nonlinear_estimated_r": self._json_safe(result.get("steer_rad")),
            "control_enabled": bool(self.calculate_control_enabled),
        }
        msg = String()
        msg.data = json.dumps(payload, separators=(",", ":"))
        self.pub_plot_sample.publish(msg)

    def control_loop(self):
        self.count += 1
        if not bool(self.get_parameter("enable_bywire").value):
            self.publish_stop("enable_bywire is false")
            return
        if self.state is None:
            self.publish_stop("vehicle state has not been received")
            return
        if not self.calculate_control_enabled:
            self.publish_stop("Stanley control calculation is disabled from UI")
            return

        state = self.state
        global_x = self._finite(state.x)
        global_y = self._finite(state.y)
        global_heading = self._finite(state.heading)

        if self.use_start_frame_as_origin and self.start_origin_capture_pending:
            if self.state_sequence <= self.start_origin_capture_min_sequence:
                self.publish_stop("waiting for first vehicle state after Start to reset origin")
                return
            self.capture_start_frame_origin(global_x, global_y, global_heading)

        x, y, psi = self.global_to_path_frame(global_x, global_y, global_heading)
        speed_x = self._finite(state.speed_x)

        try:
            result = self.compute_stanley(x, y, psi, speed_x)
        except Exception:
            self.get_logger().error("Stanley calculation failed.\n" + traceback.format_exc())
            self.publish_stop("Stanley calculation failed")
            return

        cmd = Control()
        cmd.timestamp = self.get_clock().now().nanoseconds * 1.0e-9
        cmd.count = self.count

        # EquivalentSpeedSteer mode mapping:
        #   left field  -> desired speed, cm/s
        #   right field -> front-wheel steering angle, deg
        cmd.left_drive_wheel_speed_cmd = result["speed_cm_s"]
        cmd.right_drive_wheel_speed_cmd = result["steer_deg"]

        cmd.brake_cmd = 0.0
        cmd.throttle_cmd = 0.0
        cmd.acceleration_cmd = 0.0
        cmd.gear_cmd = int(self.get_parameter("gear_cmd").value)
        cmd.turn_signal_cmd = 0
        cmd.bywire_control_enable = 1
        cmd.emerg_brake = 0
        cmd.park_enable = 0
        cmd.front_light = 0
        cmd.engine_enable = 1
        cmd.vehicle_mode = int(self.get_parameter("vehicle_mode").value)
        cmd.yaw_speed_cmd = result["steer_rad"]
        cmd.ey_out = result["lateral_error"]
        cmd.ephi_out = result["heading_error"]

        self.pub_cmd.publish(cmd)
        self.append_plot_sample(x, y, result, speed_x)
        self.last_stop_reason = ""

    def update_start_panel(self):
        if self.start_panel is None:
            return
        try:
            self.start_panel.update()
        except Exception:
            try:
                self.start_panel.close()
            except Exception:
                pass
            self.start_panel = None


def main(args=None):
    rclpy.init(args=args)
    node = StanleyControllerNode()
    try:
        rclpy.spin(node)
    finally:
        if node.start_panel is not None:
            try:
                node.start_panel.close()
            except Exception:
                pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
