import json
import math
import traceback

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from learning_preview_controller.path_helper import PathOnlyHelper, load_yaml_config

from .realtime_plot import RealtimeControllerPlot


class LearningPreviewPlotNode(Node):
    """Independent plot-only node.

    Run with:
        ros2 run learning_preview_plot plot_node

    It only subscribes to /learning_preview/plot_sample and draws figures.
    It does not own Start/Stop and does not publish control commands.

    Default path config:
        preview_params.yaml

    To draw the Stanley reference path:
        ros2 run learning_preview_plot plot_node --ros-args -p config_name:=stanley_params.yaml
    """

    def __init__(self):
        super().__init__("learning_preview_plot")

        self.declare_parameter("plot_sample_topic", "/learning_preview/plot_sample")
        self.declare_parameter("config_file", "")
        self.declare_parameter("config_name", "preview_params.yaml")
        self.declare_parameter("plot_update_period", 0.20)
        self.declare_parameter("plot_time_window_sec", 60.0)
        self.declare_parameter("plot_history_points", 2000)
        self.declare_parameter("plot_position_window_m", 35.0)
        self.declare_parameter(
            "plot_position_aspect_ratio_threshold",
            4.0,
        )

        config_file = str(self.get_parameter("config_file").value).strip()
        config_name = str(self.get_parameter("config_name").value).strip()
        if not config_name:
            config_name = "preview_params.yaml"

        cfg, resolved_config_file = load_yaml_config(
            config_file if config_file else None,
            default_config_name=config_name,
        )
        self.path_helper = PathOnlyHelper(cfg.get("path", {}) or {})
        path_cfg = self.path_helper.path_cfg

        desired_speed = path_cfg.get("v_ref", None)
        stanley_cfg = cfg.get("stanley", {}) or {}
        if "target_speed_cm_s" in stanley_cfg:
            desired_speed = float(stanley_cfg["target_speed_cm_s"]) / 100.0

        self.plotter = RealtimeControllerPlot(
            ref_path=self.path_helper.ref_path,
            initial_calculate_enabled=False,
            history_points=int(self.get_parameter("plot_history_points").value),
            time_window_sec=float(self.get_parameter("plot_time_window_sec").value),
            desired_speed=desired_speed,
            position_equal_aspect=str(
                path_cfg.get("type", "straight_circle")
            ).strip().lower() not in ("sine", "sin"),
            position_follow_window_m=float(
                self.get_parameter("plot_position_window_m").value
            ),
            position_aspect_ratio_threshold=float(
                self.get_parameter(
                    "plot_position_aspect_ratio_threshold"
                ).value
            ),
            show_control_button=False,
        )

        topic = str(self.get_parameter("plot_sample_topic").value)
        self.sub_sample = self.create_subscription(
            String,
            topic,
            self.sample_callback,
            200,
        )

        update_period = max(
            float(self.get_parameter("plot_update_period").value),
            0.05,
        )
        self.timer = self.create_timer(update_period, self.update_plot)

        self.get_logger().info(
            f"Independent plot node started. Listening on '{topic}'. "
            f"Reference path config: {resolved_config_file}. "
            "Start/Stop remains in learning_preview_controller."
        )

    @staticmethod
    def _num(value, default=math.nan):
        if value is None:
            return default
        try:
            value = float(value)
        except (TypeError, ValueError):
            return default
        return value if math.isfinite(value) else default

    def sample_callback(self, msg):
        try:
            payload = json.loads(msg.data)
        except Exception:
            self.get_logger().warn("Ignored invalid plot sample JSON.")
            return

        event = str(payload.get("event", "sample"))
        if event in ("clear", "reset"):
            ref_path = payload.get("ref_path")
            if ref_path is not None:
                try:
                    self.plotter.set_reference_path(
                        ref_path,
                        position_equal_aspect=payload.get(
                            "position_equal_aspect",
                            None,
                        ),
                    )
                except Exception as exc:
                    self.get_logger().warn(
                        f"Ignored invalid reference path from controller: {exc}"
                    )
            self.plotter.clear_history()
            self.get_logger().info(
                "Plot history cleared for a new control run. "
                f"Controller origin: x={payload.get('origin_utm_x')}, "
                f"y={payload.get('origin_utm_y')}, "
                f"heading={payload.get('origin_heading_rad')}"
            )
            return
        if event != "sample":
            return

        self.plotter.append_sample(
            time_sec=self._num(payload.get("time_sec"), 0.0),
            local_x=self._num(payload.get("local_x"), math.nan),
            local_y=self._num(payload.get("local_y"), math.nan),
            lateral_error=self._num(payload.get("lateral_error")),
            heading_error=self._num(payload.get("heading_error")),
            left_wheel_cmd=self._num(payload.get("left_wheel_cmd"), 0.0),
            right_wheel_cmd=self._num(payload.get("right_wheel_cmd"), 0.0),
            left_wheel_feedback=self._num(payload.get("left_wheel_feedback")),
            right_wheel_feedback=self._num(payload.get("right_wheel_feedback")),
            matrix_a=payload.get("matrix_a"),
            matrix_b=payload.get("matrix_b"),
            vehicle_speed=self._num(payload.get("vehicle_speed"), 0.0),
            nonlinear_observed_u=self._num(payload.get("nonlinear_observed_u")),
            nonlinear_observed_r=self._num(payload.get("nonlinear_observed_r")),
            nonlinear_estimated_u=self._num(payload.get("nonlinear_estimated_u")),
            nonlinear_estimated_r=self._num(payload.get("nonlinear_estimated_r")),
        )

    def update_plot(self):
        try:
            self.plotter.update()
        except Exception:
            self.get_logger().error("Plot update failed.\n" + traceback.format_exc())
            try:
                self.plotter.close()
            except Exception:
                pass
            raise


def main(args=None):
    rclpy.init(args=args)
    node = LearningPreviewPlotNode()
    try:
        rclpy.spin(node)
    finally:
        try:
            node.plotter.close()
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
