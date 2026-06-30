import math
import traceback

import numpy as np
import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory

from cav_msgs.msg import Control, VehicleState

from .controller import Learning_preview_controller
from .output_recorder import OutputExcelRecorder
from .realtime_plot import RealtimeControllerPlot


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
        self.declare_parameter("control_period", 0.05)
        self.declare_parameter("config_file", "")
        self.declare_parameter("gear_cmd", 4)
        self.declare_parameter("vehicle_mode", 1)
        self.declare_parameter("enable_bywire", True)
        self.declare_parameter("enable_plot", True)
        self.declare_parameter("calculate_control_on_start", False)
        self.declare_parameter("plot_update_period", 0.20)
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

        self.state = None
        self.count = 0
        self.last_stop_reason = ""

        self.last_u = 0.0
        self.last_r = 0.0
        self.last_w_l = 0.0
        self.last_w_r = 0.0

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

        # Real-time plotting state.
        self.plotter = None
        self.plot_start_time_sec = (
            self.get_clock().now().nanoseconds * 1.0e-9
        )

        if bool(self.get_parameter("enable_plot").value):
            try:
                self.plotter = RealtimeControllerPlot(
                    ref_path=self.controller.ref_path,
                    initial_calculate_enabled=self.calculate_control_enabled,
                    history_points=int(
                        self.get_parameter(
                            "plot_history_points"
                        ).value
                    ),
                    time_window_sec=float(
                        self.get_parameter(
                            "plot_time_window_sec"
                        ).value
                    ),
                    desired_speed=getattr(
                        self.controller,
                        "u_r",
                        None,
                    ),
                )
                self.plotter.set_calculate_enabled_callback(
                    self.set_calculate_control_enabled
                )
                self.get_logger().info(
                    "Real-time plotting is enabled. "
                    "Use the dashboard button to start/stop "
                    "controller.output() calculation."
                )
            except Exception:
                self.plotter = None
                self.get_logger().error(
                    "Failed to initialize real-time plotting. "
                    "The controller will continue without plots.\n"
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

        self.timer = self.create_timer(
            control_period,
            self.control_loop,
        )

        self.plot_timer = None
        if self.plotter is not None:
            plot_update_period = max(
                float(
                    self.get_parameter(
                        "plot_update_period"
                    ).value
                ),
                0.05,
            )
            self.plot_timer = self.create_timer(
                plot_update_period,
                self.update_plot,
            )

        self.get_logger().info(
            "Learning preview controller started. "
            f"state='{vehicle_state_topic}', "
            f"cmd='{control_topic}', "
            f"period={control_period:.3f}s, "
            f"config='{config_file}'"
        )

        self.get_logger().info(
            "Fixed path origin: "
            f"UTM x={self.origin_utm_x:.3f} m, "
            f"UTM y={self.origin_utm_y:.3f} m, "
            f"heading={self.origin_heading_rad:.6f} rad. "
            "This pose maps to local (0, 0, 0)."
        )

        self.get_logger().warn(
            "Controller output calculation is "
            + ("ENABLED" if self.calculate_control_enabled else "DISABLED")
            + " on startup. "
            "When disabled, vehicle state is still plotted but only "
            "safe stop commands are published."
        )

        if self.calculate_control_enabled:
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

    def set_calculate_control_enabled(self, enabled):
        """Callback used by the Matplotlib UI button.

        This only gates the expensive/active controller.output(...) call.
        Vehicle-state subscription, coordinate conversion, plot refresh, and
        safe-stop publishing continue to run.
        """

        enabled = bool(enabled)
        if enabled == self.calculate_control_enabled:
            return

        self.calculate_control_enabled = enabled

        if enabled:
            self.output_recorder.start(
                self.get_clock().now().nanoseconds * 1.0e-9
            )
            self.get_logger().warn(
                "UI enabled controller.output() calculation. "
                "Normal control commands may be published."
            )
            self.last_stop_reason = ""
        else:
            saved_path = self.output_recorder.stop()
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

    def append_plot_sample(
        self,
        local_x,
        local_y,
        lateral_error=math.nan,
        heading_error=math.nan,
        left_wheel_cmd=0.0,
        right_wheel_cmd=0.0,
        vehicle_speed=0.0,
        nonlinear_observed_u=math.nan,
        nonlinear_observed_r=math.nan,
        nonlinear_estimated_u=math.nan,
        nonlinear_estimated_r=math.nan,
    ):
        if self.plotter is None:
            return

        now_sec = (
            self.get_clock().now().nanoseconds * 1.0e-9
        )
        elapsed_sec = (
            now_sec - self.plot_start_time_sec
        )

        self.plotter.append_sample(
            time_sec=elapsed_sec,
            local_x=local_x,
            local_y=local_y,
            lateral_error=lateral_error,
            heading_error=heading_error,
            left_wheel_cmd=left_wheel_cmd,
            right_wheel_cmd=right_wheel_cmd,
            matrix_a=self.controller.A,
            matrix_b=self.controller.B,
            vehicle_speed=vehicle_speed,
            nonlinear_observed_u=nonlinear_observed_u,
            nonlinear_observed_r=nonlinear_observed_r,
            nonlinear_estimated_u=nonlinear_estimated_u,
            nonlinear_estimated_r=nonlinear_estimated_r,
        )

    def update_plot(self):
        if self.plotter is None:
            return

        try:
            self.plotter.update()
        except Exception:
            self.get_logger().error(
                "Real-time plot update failed. "
                "Plotting has been disabled.\n"
                + traceback.format_exc()
            )
            try:
                self.plotter.close()
            except Exception:
                pass
            self.plotter = None

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

        x, y, psi = self.global_to_path_frame(
            global_x,
            global_y,
            global_heading,
        )

        r = self._finite(state.yaw_rate)
        u = self._finite(state.speed_x)

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
        if node.plotter is not None:
            try:
                node.plotter.close()
            except Exception:
                pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
