import json
import math
import traceback

import numpy as np
import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory

from cav_msgs.msg import Control, VehicleState
from std_msgs.msg import String

from .controller import Learning_preview_controller
from .output_recorder import OutputExcelRecorder
from .start_stop_panel import StartStopPanel


class LearningPreviewControllerNode(Node):
    """ROS2 wrapper for Learning_preview_controller.

    Input:
        /vehicle/vehicle_state      cav_msgs/msg/VehicleState

    Output:
        /vehicle/control2bywire     cav_msgs/msg/Control

    Safety policy:
        A normal motion command is published only when:
            1. the configured reference path is valid;
            2. a vehicle-state message has been received;
            3. controller calculation succeeds;
            4. enable_bywire is true.

        Otherwise an explicit stop command is published:
            bywire_control_enable = 0
            emerg_brake = 1
            park_enable = 1
            left/right wheel command = 0

        The C++ by-wire node converts that stop request into:
            4C2 Byte 0 Stop = 1
            4C2 Byte 1 Go   = 0
            wheel bytes     = 0
    """

    def __init__(self):
        super().__init__("learning_preview_controller")

        self.declare_parameter(
            "vehicle_state_topic",
            "/vehicle/vehicle_state",
        )
        self.declare_parameter(
            "control_topic",
            "/vehicle/control2bywire",
        )
        self.declare_parameter(
            "wheel_feedback_topic",
            "/vehicle/wheel_feedback",
        )
        self.declare_parameter("control_period", 0.05)
        self.declare_parameter("config_file", "")
        self.declare_parameter("gear_cmd", 4)
        self.declare_parameter("vehicle_mode", 1)
        self.declare_parameter("enable_bywire", True)
        # Start/Stop remains inside this control node.  Plotting is now
        # a separate node that subscribes to plot_sample_topic.
        self.declare_parameter("enable_start_panel", True)
        self.declare_parameter("start_panel_update_period", 0.20)
        self.declare_parameter("calculate_control_on_start", False)
        self.declare_parameter("publish_plot_samples", True)
        self.declare_parameter("plot_sample_topic", "/learning_preview/plot_sample")
        self.declare_parameter("plot_time_window_sec", 60.0)
        self.declare_parameter("plot_history_points", 2000)
        self.declare_parameter(
            "record_dir",
            "/home/yhs/lv_tan_cyh/record",
        )

        vehicle_state_topic = str(
            self.get_parameter(
                "vehicle_state_topic"
            ).value
        )
        control_topic = str(
            self.get_parameter("control_topic").value
        )
        wheel_feedback_topic = str(
            self.get_parameter(
                "wheel_feedback_topic"
            ).value
        )
        control_period = float(
            self.get_parameter("control_period").value
        )
        config_file = str(
            self.get_parameter("config_file").value
        )

        if not config_file:
            pkg_share = get_package_share_directory(
                "learning_preview_controller"
            )
            config_file = (
                f"{pkg_share}/config/preview_params.yaml"
            )

        self.controller = Learning_preview_controller(
            config_file
        )

        self.get_logger().info(
            "Selected controller algorithm: "
            f"{getattr(self.controller, 'controller_type', 'preview')}"
        )

        # Fixed path origin in the global UTM frame.
        path_cfg = self.controller.path_cfg

        self.origin_utm_x = float(
            path_cfg["origin_utm_x"]
        )
        self.origin_utm_y = float(
            path_cfg["origin_utm_y"]
        )
        self.origin_heading_rad = float(
            path_cfg["origin_heading_rad"]
        )

        origin_values = (
            self.origin_utm_x,
            self.origin_utm_y,
            self.origin_heading_rad,
        )
        if not all(
            math.isfinite(value)
            for value in origin_values
        ):
            raise ValueError(
                "Path origin must contain finite values: "
                "origin_utm_x, origin_utm_y, "
                "origin_heading_rad"
            )

        # Optional dynamic origin mode.  When enabled, every time
        # controller.output() is started, the first fresh VehicleState
        # frame after Start becomes the path origin.  Therefore that
        # frame maps to local x=0, y=0, heading=0.
        self.use_start_frame_as_origin = self._cfg_bool(
            path_cfg,
            "use_start_frame_as_origin",
            False,
        )
        self.start_origin_capture_pending = False
        self.start_origin_capture_min_sequence = 0
        self.start_origin_has_been_captured = False

        self.state = None
        self.state_sequence = 0
        self.count = 0
        self.last_stop_reason = ""

        self.last_u = 0.0
        self.last_r = 0.0
        self.last_w_l = 0.0
        self.last_w_r = 0.0

        self.latest_left_wheel_feedback = math.nan
        self.latest_right_wheel_feedback = math.nan
        self.latest_left_wheel_feedback_rpm = math.nan
        self.latest_right_wheel_feedback_rpm = math.nan
        self.latest_wheel_feedback_time = math.nan

        self.output_recorder = OutputExcelRecorder(
            record_dir=str(
                self.get_parameter("record_dir").value
            ),
            logger=self.get_logger(),
        )

        # Control calculation is gated by the UI toggle.  The node can
        # still receive vehicle state and refresh the plot while this is
        # false, but controller.output(...) will not be called.
        self.calculate_control_enabled = bool(
            self.get_parameter(
                "calculate_control_on_start"
            ).value
        )

        # Plotting is decoupled: this node only publishes samples.
        self.plot_start_time_sec = (
            self.get_clock().now().nanoseconds * 1.0e-9
        )
        self.publish_plot_samples = bool(
            self.get_parameter("publish_plot_samples").value
        )
        plot_sample_topic = str(
            self.get_parameter("plot_sample_topic").value
        )
        self.pub_plot_sample = self.create_publisher(
            String,
            plot_sample_topic,
            50,
        )

        # Start/Stop is still owned by the controller process.
        self.start_panel = None
        if bool(self.get_parameter("enable_start_panel").value):
            try:
                self.start_panel = StartStopPanel(
                    initial_enabled=self.calculate_control_enabled,
                    on_toggle=self.set_calculate_control_enabled,
                    title="Learning Preview Controller Start / Stop",
                )
                self.get_logger().info(
                    "Start/Stop panel is enabled in the controller node. "
                    "Plotting is handled by the separate plot_node."
                )
            except Exception:
                self.start_panel = None
                self.get_logger().error(
                    "Failed to initialize Start/Stop panel. "
                    "The controller will still run; set "
                    "calculate_control_on_start=true for headless startup.\n"
                    + traceback.format_exc()
                )

        self.pub_cmd = self.create_publisher(
            Control,
            control_topic,
            2,
        )

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

        self.timer = self.create_timer(
            control_period,
            self.control_loop,
        )

        self.start_panel_timer = None
        if self.start_panel is not None:
            start_panel_update_period = max(
                float(
                    self.get_parameter(
                        "start_panel_update_period"
                    ).value
                ),
                0.05,
            )
            self.start_panel_timer = self.create_timer(
                start_panel_update_period,
                self.update_start_panel,
            )

        self.get_logger().info(
            "Learning preview controller started. "
            f"state='{vehicle_state_topic}', "
            f"wheel_feedback='{wheel_feedback_topic}', "
            f"cmd='{control_topic}', "
            f"period={control_period:.3f}s, "
            f"config='{config_file}'"
        )

        self.get_logger().info(
            "Initial path origin: "
            f"UTM x={self.origin_utm_x:.3f} m, "
            f"UTM y={self.origin_utm_y:.3f} m, "
            f"heading={self.origin_heading_rad:.6f} rad. "
            "This pose maps to local (0, 0, 0). "
            "use_start_frame_as_origin="
            f"{self.use_start_frame_as_origin}."
        )

        self.get_logger().warn(
            "Controller output calculation is "
            + ("ENABLED" if self.calculate_control_enabled else "DISABLED")
            + " on startup. "
            "When disabled, vehicle state is still published for plotting but only "
            "safe stop commands are published."
        )

        if self.calculate_control_enabled:
            if self.use_start_frame_as_origin:
                self.request_start_frame_origin_capture()
                self.get_logger().warn(
                    "Controller output calculation is enabled on startup. "
                    "Waiting for the first VehicleState frame to reset "
                    "the path origin to local (0, 0, 0)."
                )
            else:
                self.output_recorder.start(
                    self.get_clock().now().nanoseconds * 1.0e-9
                )

        if self.has_valid_reference_path():
            path = np.asarray(
                self.controller.ref_path,
                dtype=float,
            )
            self.get_logger().info(
                "Reference path is valid: "
                f"{path.shape[0]} points."
            )
        else:
            self.get_logger().error(
                "Reference path is missing or invalid. "
                "The node will publish stop commands."
            )

    @staticmethod
    def _finite(value, default=0.0):
        try:
            value = float(value)
        except (TypeError, ValueError):
            return default

        return (
            value
            if math.isfinite(value)
            else default
        )

    @staticmethod
    def _wrap_to_pi(angle):
        """Normalize an angle to [-pi, pi]."""
        return math.atan2(
            math.sin(angle),
            math.cos(angle),
        )

    @staticmethod
    def _cfg_bool(config, key, default=False):
        value = config.get(key, default)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in (
                "1",
                "true",
                "yes",
                "on",
            )
        return bool(value)

    def has_valid_reference_path(self):
        """Return True only for a usable finite reference path."""

        ref_path = getattr(
            self.controller,
            "ref_path",
            None,
        )

        if ref_path is None:
            return False

        try:
            path = np.asarray(
                ref_path,
                dtype=float,
            )
        except (TypeError, ValueError):
            return False

        if path.ndim != 2:
            return False
        if path.shape[0] < 2:
            return False
        if path.shape[1] < 4:
            return False

        return bool(
            np.all(np.isfinite(path[:, :4]))
        )

    def global_to_path_frame(
        self,
        global_x,
        global_y,
        global_heading,
    ):
        """Convert absolute UTM pose to the fixed path-local frame."""

        dx = global_x - self.origin_utm_x
        dy = global_y - self.origin_utm_y

        c = math.cos(self.origin_heading_rad)
        s = math.sin(self.origin_heading_rad)

        local_x = c * dx + s * dy
        local_y = -s * dx + c * dy

        local_heading = self._wrap_to_pi(
            global_heading -
            self.origin_heading_rad
        )

        return local_x, local_y, local_heading

    def request_start_frame_origin_capture(self):
        """Capture the first fresh VehicleState frame after Start."""

        self.start_origin_capture_pending = True
        self.start_origin_capture_min_sequence = self.state_sequence
        self.start_origin_has_been_captured = False
        self.get_logger().info(
            "Start-frame origin capture armed. "
            "The next fresh VehicleState frame will become "
            "local (0, 0, 0)."
        )

    def capture_start_frame_origin(
        self,
        global_x,
        global_y,
        global_heading,
    ):
        """Set current global pose as the origin for the path frame."""

        self.origin_utm_x = float(global_x)
        self.origin_utm_y = float(global_y)
        self.origin_heading_rad = self._wrap_to_pi(
            float(global_heading)
        )

        self.start_origin_capture_pending = False
        self.start_origin_has_been_captured = True

        # Reset controller state that depends on path progress or previous
        # samples so the run starts cleanly from the new local frame.
        if hasattr(self.controller, "ID_last"):
            self.controller.ID_last = 0
        if hasattr(self.controller, "lateral_err"):
            self.controller.lateral_err = 0.0
        if hasattr(self.controller, "ResetLateralBiasIntegral"):
            self.controller.ResetLateralBiasIntegral()

        self.last_u = 0.0
        self.last_r = 0.0
        self.last_w_l = 0.0
        self.last_w_r = 0.0
        self.last_stop_reason = ""

        now_sec = (
            self.get_clock().now().nanoseconds * 1.0e-9
        )
        self.plot_start_time_sec = now_sec

        self.publish_plot_reset()

        if self.output_recorder.is_recording:
            saved_path = self.output_recorder.stop()
            self.save_plot_figures(saved_path)
        self.output_recorder.start(now_sec)

        self.get_logger().warn(
            "Start-frame origin captured: "
            f"UTM x={self.origin_utm_x:.3f} m, "
            f"UTM y={self.origin_utm_y:.3f} m, "
            f"heading={self.origin_heading_rad:.6f} rad. "
            "This frame now maps to local x=0, y=0, heading=0."
        )


    def compute_tracking_errors(self, x, y, psi):
        """Compute e_y and e_psi without running controller.output()."""
        try:
            ref_path = np.asarray(self.controller.ref_path, dtype=float)
            _, nearest_id = self.controller.FindNearestPoint(
                ref_path,
                x,
                y,
                psi,
                int(getattr(self.controller, "ID_last", 0)),
            )
            lateral_error, heading_error, _ = self.controller.CalErr(
                x,
                y,
                psi,
                nearest_id,
                ref_path,
            )
            return (
                self._finite(lateral_error, math.nan),
                self._finite(heading_error, math.nan),
            )
        except Exception:
            return math.nan, math.nan

    def vehicle_state_callback(
        self,
        msg: VehicleState,
    ):
        self.state = msg
        self.state_sequence += 1

    def wheel_feedback_callback(
        self,
        msg: VehicleState,
    ):
        """Store the freshest CAN wheel-speed feedback sample."""

        left_speed = self._finite(
            msg.left_drive_wheel_speed,
            math.nan,
        )
        right_speed = self._finite(
            msg.right_drive_wheel_speed,
            math.nan,
        )
        left_rpm = self._finite(
            msg.left_drive_wheel_rpm,
            math.nan,
        )
        right_rpm = self._finite(
            msg.right_drive_wheel_rpm,
            math.nan,
        )

        if math.isfinite(left_speed):
            self.latest_left_wheel_feedback = left_speed
        if math.isfinite(right_speed):
            self.latest_right_wheel_feedback = right_speed
        if math.isfinite(left_rpm):
            self.latest_left_wheel_feedback_rpm = left_rpm
        if math.isfinite(right_rpm):
            self.latest_right_wheel_feedback_rpm = right_rpm

        stamp = self._finite(msg.timestamp, math.nan)
        if not math.isfinite(stamp):
            stamp = self.get_clock().now().nanoseconds * 1.0e-9
        self.latest_wheel_feedback_time = stamp

    def get_current_wheel_feedback(self, state=None):
        """Return latest left/right wheel feedback, with VehicleState fallback."""

        left_speed = self.latest_left_wheel_feedback
        right_speed = self.latest_right_wheel_feedback
        left_rpm = self.latest_left_wheel_feedback_rpm
        right_rpm = self.latest_right_wheel_feedback_rpm

        if state is not None:
            if not math.isfinite(left_speed):
                left_speed = self._finite(
                    state.left_drive_wheel_speed,
                    math.nan,
                )
            if not math.isfinite(right_speed):
                right_speed = self._finite(
                    state.right_drive_wheel_speed,
                    math.nan,
                )
            if not math.isfinite(left_rpm):
                left_rpm = self._finite(
                    state.left_drive_wheel_rpm,
                    math.nan,
                )
            if not math.isfinite(right_rpm):
                right_rpm = self._finite(
                    state.right_drive_wheel_rpm,
                    math.nan,
                )

        return left_speed, right_speed, left_rpm, right_rpm

    def save_plot_figures(self, xlsx_path):
        # Plot figures are now owned by plot_node, not by the controller.
        return

    def set_calculate_control_enabled(self, enabled):
        """Callback used by the controller-owned Start/Stop panel.

        This only gates the expensive/active controller.output(...) call.
        Vehicle-state subscription, coordinate conversion, plot sample
        publishing, and safe-stop publishing continue to run.
        """

        enabled = bool(enabled)
        if enabled == self.calculate_control_enabled:
            return

        self.calculate_control_enabled = enabled

        if enabled:
            self.plot_start_time_sec = (
                self.get_clock().now().nanoseconds * 1.0e-9
            )
            if hasattr(self.controller, "ResetLateralBiasIntegral"):
                self.controller.ResetLateralBiasIntegral()

            # Clear the independent plot immediately when Start is clicked.
            # If dynamic-origin mode is enabled, another reset is sent after
            # the first fresh VehicleState frame captures the exact origin.
            self.publish_plot_reset()
            if self.use_start_frame_as_origin:
                self.request_start_frame_origin_capture()
                self.get_logger().warn(
                    "UI enabled controller.output() calculation. "
                    "Waiting for the first VehicleState frame after "
                    "Start to reset local origin and heading."
                )
            else:
                self.output_recorder.start(
                    self.get_clock().now().nanoseconds * 1.0e-9
                )
                self.get_logger().warn(
                    "UI enabled controller.output() calculation. "
                    "Normal control commands may be published."
                )
            self.last_stop_reason = ""
        else:
            self.start_origin_capture_pending = False
            if hasattr(self.controller, "ResetLateralBiasIntegral"):
                self.controller.ResetLateralBiasIntegral()
            saved_path = self.output_recorder.stop()
            self.save_plot_figures(saved_path)
            self.get_logger().warn(
                "UI disabled controller.output() calculation. "
                "Publishing safe stop commands only."
            )
            if saved_path is not None:
                self.get_logger().info(
                    f"Output record saved to: {saved_path}"
                )
            self.publish_stop(
                "controller output calculation is disabled from UI"
            )

        if self.start_panel is not None:
            self.start_panel.set_enabled(self.calculate_control_enabled)

        self.publish_plot_event(
            "control_enabled",
            enabled=self.calculate_control_enabled,
        )

    @staticmethod
    def _json_safe(value):
        if isinstance(value, np.ndarray):
            return LearningPreviewControllerNode._json_safe(
                value.tolist()
            )
        if isinstance(value, (list, tuple)):
            return [
                LearningPreviewControllerNode._json_safe(item)
                for item in value
            ]
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
        return value if math.isfinite(value) else None

    def publish_plot_reset(self):
        """Clear plot history and synchronize path/origin with this controller."""
        ref_path = np.asarray(self.controller.ref_path, dtype=float)
        path_type = str(self.controller.path_cfg.get("type", "straight_circle")).strip().lower()
        self.publish_plot_event(
            "reset",
            controller_type=str(getattr(self.controller, "controller_type", "preview")),
            origin_utm_x=self._json_safe(self.origin_utm_x),
            origin_utm_y=self._json_safe(self.origin_utm_y),
            origin_heading_rad=self._json_safe(self.origin_heading_rad),
            ref_path=self._json_safe(ref_path),
            position_equal_aspect=path_type not in ("sine", "sin"),
        )

    def publish_plot_event(self, event, **extra):
        if not self.publish_plot_samples:
            return
        payload = {
            "event": str(event),
            "ros_time_sec": self.get_clock().now().nanoseconds * 1.0e-9,
        }
        payload.update(extra)
        msg = String()
        msg.data = json.dumps(payload, separators=(",", ":"))
        self.pub_plot_sample.publish(msg)

    def append_plot_sample(
        self,
        local_x,
        local_y,
        lateral_error=math.nan,
        heading_error=math.nan,
        left_wheel_cmd=0.0,
        right_wheel_cmd=0.0,
        left_wheel_feedback=math.nan,
        right_wheel_feedback=math.nan,
        vehicle_speed=0.0,
        nonlinear_observed_u=math.nan,
        nonlinear_observed_r=math.nan,
        nonlinear_estimated_u=math.nan,
        nonlinear_estimated_r=math.nan,
    ):
        if not self.publish_plot_samples:
            return

        now_sec = (
            self.get_clock().now().nanoseconds * 1.0e-9
        )
        elapsed_sec = now_sec - self.plot_start_time_sec

        payload = {
            "event": "sample",
            "time_sec": self._json_safe(elapsed_sec),
            "local_x": self._json_safe(local_x),
            "local_y": self._json_safe(local_y),
            "lateral_error": self._json_safe(lateral_error),
            "heading_error": self._json_safe(heading_error),
            "left_wheel_cmd": self._json_safe(left_wheel_cmd),
            "right_wheel_cmd": self._json_safe(right_wheel_cmd),
            "left_wheel_feedback": self._json_safe(left_wheel_feedback),
            "right_wheel_feedback": self._json_safe(right_wheel_feedback),
            "matrix_a": self._json_safe(getattr(self.controller, "A", None)),
            "matrix_b": self._json_safe(getattr(self.controller, "B", None)),
            "vehicle_speed": self._json_safe(vehicle_speed),
            "nonlinear_observed_u": self._json_safe(nonlinear_observed_u),
            "nonlinear_observed_r": self._json_safe(nonlinear_observed_r),
            "nonlinear_estimated_u": self._json_safe(nonlinear_estimated_u),
            "nonlinear_estimated_r": self._json_safe(nonlinear_estimated_r),
            "control_enabled": bool(self.calculate_control_enabled),
        }

        msg = String()
        msg.data = json.dumps(payload, separators=(",", ":"))
        self.pub_plot_sample.publish(msg)

    def update_start_panel(self):
        if self.start_panel is None:
            return
        try:
            self.start_panel.update()
        except Exception:
            self.get_logger().error(
                "Start/Stop panel update failed.\n"
                + traceback.format_exc()
            )
            try:
                self.start_panel.close()
            except Exception:
                pass
            self.start_panel = None

    def make_safe_zero_cmd(self):
        """Create an explicit vehicle-stop request."""

        cmd = Control()

        cmd.timestamp = (
            self.get_clock().now().nanoseconds
            * 1.0e-9
        )
        cmd.count = self.count

        cmd.left_drive_wheel_speed_cmd = 0.0
        cmd.right_drive_wheel_speed_cmd = 0.0

        cmd.brake_cmd = 0.0
        cmd.throttle_cmd = 0.0
        cmd.acceleration_cmd = 0.0

        cmd.gear_cmd = int(
            self.get_parameter("gear_cmd").value
        )
        cmd.turn_signal_cmd = 0

        # These three fields are interpreted by the C++ by-wire node.
        cmd.bywire_control_enable = 0
        cmd.emerg_brake = 1
        cmd.park_enable = 1

        cmd.front_light = 0
        cmd.engine_enable = 1
        cmd.vehicle_mode = int(
            self.get_parameter(
                "vehicle_mode"
            ).value
        )

        cmd.ey_out = 0.0
        cmd.ephi_out = 0.0

        return cmd

    def publish_stop(self, reason):
        """Publish stop and log only when the stop reason changes."""

        if reason != self.last_stop_reason:
            self.get_logger().warn(
                f"Publishing stop command: {reason}"
            )
            self.last_stop_reason = reason

        self.pub_cmd.publish(
            self.make_safe_zero_cmd()
        )

        self.last_w_l = 0.0
        self.last_w_r = 0.0

    def control_loop(self):
        self.count += 1

        if not bool(
            self.get_parameter(
                "enable_bywire"
            ).value
        ):
            self.publish_stop(
                "enable_bywire is false"
            )
            return

        # No generated/configured path means no motion command.
        if not self.has_valid_reference_path():
            self.publish_stop(
                "reference path is missing or invalid"
            )
            return

        # A controller must not move without a current vehicle pose.
        if self.state is None:
            self.publish_stop(
                "vehicle state has not been received"
            )
            return

        state = self.state

        global_x = self._finite(state.x)
        global_y = self._finite(state.y)
        global_heading = self._finite(
            state.heading
        )

        if (
            self.calculate_control_enabled
            and self.use_start_frame_as_origin
            and self.start_origin_capture_pending
        ):
            if (
                self.state_sequence
                <= self.start_origin_capture_min_sequence
            ):
                self.publish_stop(
                    "waiting for first vehicle state after Start "
                    "to reset origin"
                )
                return

            self.capture_start_frame_origin(
                global_x,
                global_y,
                global_heading,
            )

        x, y, psi = self.global_to_path_frame(
            global_x,
            global_y,
            global_heading,
        )

        r = self._finite(state.yaw_rate)
        u = self._finite(state.speed_x)

        (
            left_wheel_feedback,
            right_wheel_feedback,
            left_wheel_feedback_rpm,
            right_wheel_feedback_rpm,
        ) = self.get_current_wheel_feedback(state)

        if not self.calculate_control_enabled:
            # Keep the real-time state and tracking-error display alive
            # without executing controller.output(...).
            ey, ephi = self.compute_tracking_errors(x, y, psi)
            self.append_plot_sample(
                local_x=x,
                local_y=y,
                lateral_error=ey,
                heading_error=ephi,
                left_wheel_cmd=0.0,
                right_wheel_cmd=0.0,
                left_wheel_feedback=left_wheel_feedback,
                right_wheel_feedback=right_wheel_feedback,
                vehicle_speed=u,
            )
            self.last_u = u
            self.last_r = r
            self.publish_stop(
                "controller output calculation is disabled from UI"
            )
            return

        state_in = [
            x,
            y,
            psi,
            r,
            u,
            self.last_r,
            self.last_u,
            self.last_w_l,
            self.last_w_r,
        ]

        try:
            out = self.controller.output(state_in)

            if len(out) <= 13:
                raise ValueError(
                    "Controller output does not contain "
                    "left/right wheel commands"
                )

            raw_w_l = float(out[12])
            raw_w_r = float(out[13])

            if not (
                math.isfinite(raw_w_l) and
                math.isfinite(raw_w_r)
            ):
                raise ValueError(
                    "Controller returned a non-finite "
                    "wheel command"
                )

            w_l = raw_w_l
            w_r = raw_w_r

            ey = (
                self._finite(out[17], math.nan)
                if len(out) > 17
                else math.nan
            )

            ref_psi = (
                self._finite(out[16], math.nan)
                if len(out) > 16
                else math.nan
            )
            ephi = (
                self._wrap_to_pi(psi - ref_psi)
                if math.isfinite(ref_psi)
                else math.nan
            )

            nonlinear_estimated_u = (
                self._finite(out[0], math.nan)
                if len(out) > 0
                else math.nan
            )
            nonlinear_estimated_r = (
                self._finite(out[1], math.nan)
                if len(out) > 1
                else math.nan
            )
            nonlinear_observed_u = (
                self._finite(out[2], math.nan)
                if len(out) > 2
                else math.nan
            )
            nonlinear_observed_r = (
                self._finite(out[3], math.nan)
                if len(out) > 3
                else math.nan
            )
        except Exception:
            self.get_logger().error(
                "Controller output failed.\n"
                + traceback.format_exc()
            )
            ey, ephi = self.compute_tracking_errors(x, y, psi)
            self.append_plot_sample(
                local_x=x,
                local_y=y,
                lateral_error=ey,
                heading_error=ephi,
                left_wheel_cmd=0.0,
                right_wheel_cmd=0.0,
                left_wheel_feedback=left_wheel_feedback,
                right_wheel_feedback=right_wheel_feedback,
                vehicle_speed=u,
            )
            self.publish_stop(
                "controller calculation failed"
            )
            return

        cmd = Control()

        cmd.timestamp = (
            self.get_clock().now().nanoseconds
            * 1.0e-9
        )
        cmd.count = self.count

        cmd.left_drive_wheel_speed_cmd = w_l
        cmd.right_drive_wheel_speed_cmd = w_r

        cmd.brake_cmd = 0.0
        cmd.throttle_cmd = 0.0
        cmd.acceleration_cmd = 0.0

        cmd.gear_cmd = int(
            self.get_parameter("gear_cmd").value
        )
        cmd.turn_signal_cmd = 0

        # Normal running state.
        cmd.bywire_control_enable = 1
        cmd.emerg_brake = 0
        cmd.park_enable = 0

        cmd.front_light = 0
        cmd.engine_enable = 1
        cmd.vehicle_mode = int(
            self.get_parameter(
                "vehicle_mode"
            ).value
        )

        cmd.ey_out = ey
        cmd.ephi_out = ephi

        self.output_recorder.append_output_sample(
            ros_time_sec=cmd.timestamp,
            loop_count=self.count,
            local_x=x,
            local_y=y,
            local_heading=psi,
            yaw_rate=r,
            speed_x=u,
            last_yaw_rate=self.last_r,
            last_speed_x=self.last_u,
            last_left_wheel_cmd=self.last_w_l,
            last_right_wheel_cmd=self.last_w_r,
            output_values=out,
            left_wheel_feedback=left_wheel_feedback,
            right_wheel_feedback=right_wheel_feedback,
            left_wheel_feedback_rpm=left_wheel_feedback_rpm,
            right_wheel_feedback_rpm=right_wheel_feedback_rpm,
        )

        # Record the local vehicle position, the commands that are about
        # to be published, and the current learned A/B matrices.
        self.append_plot_sample(
            local_x=x,
            local_y=y,
            lateral_error=ey,
            heading_error=ephi,
            left_wheel_cmd=w_l,
            right_wheel_cmd=w_r,
            left_wheel_feedback=left_wheel_feedback,
            right_wheel_feedback=right_wheel_feedback,
            vehicle_speed=u,
            nonlinear_observed_u=nonlinear_observed_u,
            nonlinear_observed_r=nonlinear_observed_r,
            nonlinear_estimated_u=nonlinear_estimated_u,
            nonlinear_estimated_r=nonlinear_estimated_r,
        )

        self.pub_cmd.publish(cmd)

        self.last_stop_reason = ""
        self.last_u = u
        self.last_r = r
        self.last_w_l = w_l
        self.last_w_r = w_r

    def close_output_recorder(self):
        saved_path = self.output_recorder.close()
        self.save_plot_figures(saved_path)
        if saved_path is not None:
            self.get_logger().info(
                f"Output record saved to: {saved_path}"
            )


def main(args=None):
    rclpy.init(args=args)

    node = LearningPreviewControllerNode()

    try:
        rclpy.spin(node)
    finally:
        try:
            node.close_output_recorder()
        except Exception:
            pass
        if node.start_panel is not None:
            try:
                node.start_panel.close()
            except Exception:
                pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
