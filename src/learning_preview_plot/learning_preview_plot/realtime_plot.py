from collections import deque
import math

import numpy as np


PLOT_LAYOUT_VERSION = "position_adaptive_follow_window_v8"


class RealtimeControllerPlot:
    """Non-blocking Matplotlib dashboard for the ROS2 controller.

    Window layout:
        1. Position/trajectory, lateral error, and heading error
        2. Control input, vehicle speed, learned matrix values,
           wheel-speed feedback, and nonlinear functions in one figure

    The nonlinear function channels are separated:
        - f_U observed vs estimated
        - f_R observed vs estimated
    """

    def __init__(
        self,
        ref_path,
        initial_calculate_enabled=False,
        history_points=2000,
        time_window_sec=60.0,
        desired_speed=None,
        position_equal_aspect=True,
        position_follow_window_m=35.0,
        position_aspect_ratio_threshold=4.0,
        show_control_button=True,
    ):
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button

        print(f"[RealtimeControllerPlot] PLOT_LAYOUT_VERSION = {PLOT_LAYOUT_VERSION}")

        self.plt = plt
        self.enabled = True
        self.motion_visible = True
        self.estimation_visible = True
        self.calculate_enabled = bool(initial_calculate_enabled)
        self.calculate_enabled_callback = None
        self.show_control_button = bool(show_control_button)
        self.time_window_sec = max(float(time_window_sec), 1.0)
        self.history_points = max(int(history_points), 100)
        self.desired_speed = self._to_finite_or_none(desired_speed)
        # Position view is adaptive in two ways:
        #   1) long/slender paths automatically use independent X/Y scaling;
        #   2) after samples arrive, the axes follow a local path window
        #      instead of always displaying the complete reference path.
        self.position_equal_aspect_requested = bool(position_equal_aspect)
        self.position_equal_aspect = bool(position_equal_aspect)
        self.position_follow_window_m = max(float(position_follow_window_m), 5.0)
        self.position_aspect_ratio_threshold = max(
            float(position_aspect_ratio_threshold),
            1.0,
        )

        ref = np.asarray(ref_path, dtype=float)
        if ref.ndim != 2 or ref.shape[0] < 2 or ref.shape[1] < 2:
            raise ValueError(
                "ref_path must be an NxM array with at least x/y columns"
            )

        self.ref_x = ref[:, 0].copy()
        self.ref_y = ref[:, 1].copy()
        self._rebuild_reference_geometry()
        self._refresh_position_aspect()

        self.time = deque(maxlen=self.history_points)
        self.actual_x = deque(maxlen=self.history_points)
        self.actual_y = deque(maxlen=self.history_points)
        self.lateral_error = deque(maxlen=self.history_points)
        self.heading_error = deque(maxlen=self.history_points)
        self.left_cmd = deque(maxlen=self.history_points)
        self.right_cmd = deque(maxlen=self.history_points)
        self.left_feedback = deque(maxlen=self.history_points)
        self.right_feedback = deque(maxlen=self.history_points)
        self.vehicle_speed = deque(maxlen=self.history_points)

        self.nonlinear_observed_u = deque(maxlen=self.history_points)
        self.nonlinear_observed_r = deque(maxlen=self.history_points)
        self.nonlinear_estimated_u = deque(maxlen=self.history_points)
        self.nonlinear_estimated_r = deque(maxlen=self.history_points)

        self.a00_history = deque(maxlen=self.history_points)
        self.a11_history = deque(maxlen=self.history_points)
        self.b00_history = deque(maxlen=self.history_points)
        self.b11_history = deque(maxlen=self.history_points)

        self.plt.ion()

        # Figure 1: trajectory and tracking errors.
        self.fig_path, self.path_axes = self.plt.subplots(
            3,
            1,
            figsize=(12, 10),
            num="Position / Trajectory and Tracking Errors",
            gridspec_kw={"height_ratios": [2.4, 1.0, 1.0]},
        )
        self.fig_path.subplots_adjust(
            hspace=0.48,
            bottom=0.08,
            top=0.95,
            left=0.09,
            right=0.97,
        )

        self.ax_path = self.path_axes[0]
        self.ax_lateral_error = self.path_axes[1]
        self.ax_heading_error = self.path_axes[2]

        (self.reference_path_line,) = self.ax_path.plot(
            self.ref_x,
            self.ref_y,
            linestyle="--",
            linewidth=1.8,
            label="Desired path",
        )
        (self.actual_path_line,) = self.ax_path.plot(
            [],
            [],
            linewidth=2.5,
            label="Actual trajectory",
        )
        (self.current_position_marker,) = self.ax_path.plot(
            [],
            [],
            marker="o",
            linestyle="None",
            markersize=8,
            label="Current position",
        )

        self.ax_path.set_title(
            "Desired Path and Actual Vehicle Trajectory"
        )
        self.ax_path.set_xlabel("Local X (m)")
        self.ax_path.set_ylabel("Local Y (m)")
        self.ax_path.grid(True)
        self.ax_path.legend(loc="best")
        self._apply_position_aspect()
        self._set_position_limits()

        (self.lateral_error_line,) = self.ax_lateral_error.plot(
            [],
            [],
            label="Lateral error",
        )
        self.ax_lateral_error.axhline(
            0.0,
            linewidth=1.0,
            linestyle="--",
        )
        self.ax_lateral_error.set_title("Lateral Error")
        self.ax_lateral_error.set_xlabel("Time (s)")
        self.ax_lateral_error.set_ylabel("e_y (m)")
        self.ax_lateral_error.grid(True)
        self.ax_lateral_error.legend(loc="best")

        (self.heading_error_line,) = self.ax_heading_error.plot(
            [],
            [],
            label="Heading error",
        )
        self.ax_heading_error.axhline(
            0.0,
            linewidth=1.0,
            linestyle="--",
        )
        self.ax_heading_error.set_title("Heading Error")
        self.ax_heading_error.set_xlabel("Time (s)")
        self.ax_heading_error.set_ylabel("e_psi (rad)")
        self.ax_heading_error.grid(True)
        self.ax_heading_error.legend(loc="best")

        # Figure 2: combine the old motion figure and estimation figure.
        self.fig_motion, combined_axes = self.plt.subplots(
            3,
            2,
            figsize=(14, 10),
            num="Control, Speed, Matrix, Feedback and Nonlinear Estimates",
        )
        self.fig_motion.subplots_adjust(
            hspace=0.58,
            wspace=0.30,
            bottom=0.13,
            top=0.94,
            left=0.07,
            right=0.98,
        )
        self.motion_axes = combined_axes.ravel()
        # Keep fig_estimation as an alias so older close/update logic cannot
        # accidentally reference a missing attribute.
        self.fig_estimation = self.fig_motion

        self.ax_control = self.motion_axes[0]
        self.ax_speed = self.motion_axes[1]
        self.ax_a_params = self.motion_axes[2]
        self.ax_b_params = self.motion_axes[3]
        self.ax_nonlinear_u = self.motion_axes[4]
        self.ax_nonlinear_r = self.motion_axes[5]

        (self.left_cmd_line,) = self.ax_control.plot(
            [],
            [],
            label="Left wheel command",
        )
        (self.right_cmd_line,) = self.ax_control.plot(
            [],
            [],
            label="Right wheel command",
        )
        (self.left_feedback_line,) = self.ax_control.plot(
            [],
            [],
            linestyle=":",
            label="Left wheel feedback",
        )
        (self.right_feedback_line,) = self.ax_control.plot(
            [],
            [],
            linestyle=":",
            label="Right wheel feedback",
        )
        self.ax_control.set_title("Wheel Command and Actual Feedback")
        self.ax_control.set_xlabel("Time (s)")
        self.ax_control.set_ylabel("Wheel speed (rad/s)")
        self.ax_control.grid(True)
        self.ax_control.legend(loc="best")

        (self.vehicle_speed_line,) = self.ax_speed.plot(
            [],
            [],
            label="Measured vehicle speed",
        )
        if self.desired_speed is not None:
            self.reference_speed_line = self.ax_speed.axhline(
                self.desired_speed,
                linestyle="--",
                linewidth=1.2,
                label="Reference speed",
            )
        else:
            self.reference_speed_line = None
        self.ax_speed.set_title("Vehicle Speed")
        self.ax_speed.set_xlabel("Time (s)")
        self.ax_speed.set_ylabel("Speed (m/s)")
        self.ax_speed.grid(True)
        self.ax_speed.legend(loc="best")

        (self.a00_line,) = self.ax_a_params.plot(
            [],
            [],
            label="A00",
        )
        (self.a11_line,) = self.ax_a_params.plot(
            [],
            [],
            label="A11",
        )
        self.ax_a_params.set_title("Realtime Matrix A Parameters")
        self.ax_a_params.set_xlabel("Time (s)")
        self.ax_a_params.set_ylabel("A value")
        self.ax_a_params.grid(True)
        self.ax_a_params.legend(loc="best")

        (self.b00_line,) = self.ax_b_params.plot(
            [],
            [],
            label="B00",
        )
        (self.b11_line,) = self.ax_b_params.plot(
            [],
            [],
            label="B11",
        )
        self.ax_b_params.set_title("Realtime Matrix B Parameters")
        self.ax_b_params.set_xlabel("Time (s)")
        self.ax_b_params.set_ylabel("B value")
        self.ax_b_params.grid(True)
        self.ax_b_params.legend(loc="best")

        self.status_text = self.fig_motion.text(
            0.5,
            0.070,
            "",
            ha="center",
            va="center",
            fontsize=10,
        )
        self.button_axis = None
        self.calculate_button = None
        if self.show_control_button:
            self.button_axis = self.fig_motion.add_axes(
                [0.36, 0.020, 0.28, 0.040]
            )
            self.calculate_button = Button(
                self.button_axis,
                "",
            )
            self.calculate_button.on_clicked(
                self._on_calculate_button_clicked
            )

        (self.obs_u_line,) = self.ax_nonlinear_u.plot(
            [],
            [],
            label="Observed f_U",
        )
        (self.est_u_line,) = self.ax_nonlinear_u.plot(
            [],
            [],
            linestyle="--",
            label="Estimated f_U",
        )
        self.ax_nonlinear_u.set_title(
            "Nonlinear Function f_U: Observed vs Estimated"
        )
        self.ax_nonlinear_u.set_xlabel("Time (s)")
        self.ax_nonlinear_u.set_ylabel("Function value")
        self.ax_nonlinear_u.grid(True)
        self.ax_nonlinear_u.legend(loc="best")

        (self.obs_r_line,) = self.ax_nonlinear_r.plot(
            [],
            [],
            label="Observed f_R",
        )
        (self.est_r_line,) = self.ax_nonlinear_r.plot(
            [],
            [],
            linestyle="--",
            label="Estimated f_R",
        )
        self.ax_nonlinear_r.set_title(
            "Nonlinear Function f_R: Observed vs Estimated"
        )
        self.ax_nonlinear_r.set_xlabel("Time (s)")
        self.ax_nonlinear_r.set_ylabel("Function value")
        self.ax_nonlinear_r.grid(True)
        self.ax_nonlinear_r.legend(loc="best")

        self.fig_path.canvas.mpl_connect(
            "close_event",
            self._on_main_close,
        )
        self.fig_motion.canvas.mpl_connect(
            "close_event",
            self._on_motion_close,
        )

        self._refresh_control_ui()

        for fig in (
            self.fig_path,
            self.fig_motion,
        ):
            fig.canvas.draw_idle()
            fig.canvas.flush_events()

        self.plt.show(block=False)

    def _on_main_close(self, _event):
        self.enabled = False

    def _on_motion_close(self, _event):
        self.motion_visible = False
        self.estimation_visible = False

    def _on_estimation_close(self, _event):
        self.motion_visible = False
        self.estimation_visible = False

    def set_calculate_enabled_callback(self, callback):
        self.calculate_enabled_callback = callback

    def _on_calculate_button_clicked(self, _event):
        self.calculate_enabled = not self.calculate_enabled

        if self.calculate_enabled_callback is not None:
            try:
                self.calculate_enabled_callback(
                    self.calculate_enabled
                )
            except Exception as exc:
                print(
                    "Failed to update controller calculation state:",
                    exc,
                )

        self._refresh_control_ui()
        self.fig_motion.canvas.draw_idle()
        self.fig_motion.canvas.flush_events()

    def _refresh_control_ui(self):
        if self.calculate_enabled:
            button_text = "Stop output()"
            status_text = (
                "Control calculation: ON - "
                "controller.output() is running."
            )
        else:
            button_text = "Start output()"
            status_text = (
                "Control calculation: OFF - "
                "real-time state is still displayed; "
                "safe stop commands are published."
            )

        if self.calculate_button is not None:
            self.calculate_button.label.set_text(button_text)
            self.status_text.set_text(status_text)
        else:
            self.status_text.set_text(
                "Plot-only node: Start/Stop is controlled by the controller node."
            )

    def _rebuild_reference_geometry(self):
        """Cache cumulative path length for local-window selection."""
        if self.ref_x.size < 2:
            self.ref_s = np.zeros_like(self.ref_x, dtype=float)
            return

        dx = np.diff(self.ref_x)
        dy = np.diff(self.ref_y)
        ds = np.hypot(dx, dy)
        ds[~np.isfinite(ds)] = 0.0
        self.ref_s = np.concatenate(([0.0], np.cumsum(ds)))

    def _refresh_position_aspect(self):
        """Disable equal aspect automatically for long, slender paths."""
        finite = np.isfinite(self.ref_x) & np.isfinite(self.ref_y)
        if not np.any(finite):
            self.position_equal_aspect = self.position_equal_aspect_requested
            return

        x = self.ref_x[finite]
        y = self.ref_y[finite]
        x_span = max(float(np.ptp(x)), 1.0e-6)
        y_span = max(float(np.ptp(y)), 1.0e-6)
        slenderness = max(x_span, y_span) / min(x_span, y_span)

        self.position_equal_aspect = (
            self.position_equal_aspect_requested
            and slenderness <= self.position_aspect_ratio_threshold
        )

    def _apply_position_aspect(self):
        self.ax_path.set_aspect(
            "equal" if self.position_equal_aspect else "auto",
            adjustable="box",
        )

    def _local_reference_segment(self, current_x, current_y):
        """Return the nearby reference-path section around the vehicle."""
        if self.ref_x.size == 0:
            return self.ref_x, self.ref_y

        distance_sq = (
            (self.ref_x - current_x) ** 2
            + (self.ref_y - current_y) ** 2
        )
        finite = np.isfinite(distance_sq)
        if not np.any(finite):
            return self.ref_x, self.ref_y

        safe_distance = np.where(finite, distance_sq, np.inf)
        nearest_index = int(np.argmin(safe_distance))
        nearest_s = float(self.ref_s[nearest_index])

        # Show more road in front of the vehicle than behind it.
        back_distance = 0.30 * self.position_follow_window_m
        ahead_distance = 0.70 * self.position_follow_window_m
        mask = (
            (self.ref_s >= nearest_s - back_distance)
            & (self.ref_s <= nearest_s + ahead_distance)
        )

        if np.count_nonzero(mask) < 2:
            start = max(0, nearest_index - 1)
            stop = min(self.ref_x.size, nearest_index + 2)
            return self.ref_x[start:stop], self.ref_y[start:stop]

        return self.ref_x[mask], self.ref_y[mask]

    def _recent_actual_segment(self, actual_x, actual_y):
        """Keep only the recent travelled section used for position limits."""
        x = np.asarray(actual_x, dtype=float)
        y = np.asarray(actual_y, dtype=float)
        finite = np.isfinite(x) & np.isfinite(y)
        x = x[finite]
        y = y[finite]
        if x.size <= 1:
            return x, y

        segment_lengths = np.hypot(np.diff(x), np.diff(y))
        segment_lengths[~np.isfinite(segment_lengths)] = 0.0
        travelled_from_end = np.concatenate(
            ([0.0], np.cumsum(segment_lengths[::-1]))
        )
        keep_count = int(
            np.searchsorted(
                travelled_from_end,
                self.position_follow_window_m,
                side="right",
            )
        )
        keep_count = min(max(keep_count, 2), x.size)
        return x[-keep_count:], y[-keep_count:]

    def _set_position_limits(self, actual_x=None, actual_y=None):
        has_actual = (
            actual_x is not None
            and actual_y is not None
            and len(actual_x) > 0
            and len(actual_y) > 0
        )

        if has_actual:
            recent_x, recent_y = self._recent_actual_segment(
                actual_x,
                actual_y,
            )
            if recent_x.size > 0:
                ref_x, ref_y = self._local_reference_segment(
                    float(recent_x[-1]),
                    float(recent_y[-1]),
                )
                x_values = np.concatenate([ref_x, recent_x])
                y_values = np.concatenate([ref_y, recent_y])
            else:
                x_values = self.ref_x
                y_values = self.ref_y
        else:
            x_values = self.ref_x
            y_values = self.ref_y

        finite_mask = np.isfinite(x_values) & np.isfinite(y_values)
        if not np.any(finite_mask):
            return

        x_values = x_values[finite_mask]
        y_values = y_values[finite_mask]

        x_min = float(np.min(x_values))
        x_max = float(np.max(x_values))
        y_min = float(np.min(y_values))
        y_max = float(np.max(y_values))

        if self.position_equal_aspect:
            x_span = max(x_max - x_min, 1.0)
            y_span = max(y_max - y_min, 1.0)
            span = max(x_span, y_span)

            x_center = 0.5 * (x_min + x_max)
            y_center = 0.5 * (y_min + y_max)

            self.ax_path.set_xlim(
                x_center - 0.55 * span,
                x_center + 0.55 * span,
            )
            self.ax_path.set_ylim(
                y_center - 0.55 * span,
                y_center + 0.55 * span,
            )
            return

        # Independent limits make long double-lane-change/sine paths readable.
        x_span = max(x_max - x_min, 1.0)
        y_span = max(y_max - y_min, 1.0)
        x_margin = max(0.08 * x_span, 0.5)
        y_margin = max(0.15 * y_span, 0.2)

        self.ax_path.set_xlim(x_min - x_margin, x_max + x_margin)
        self.ax_path.set_ylim(y_min - y_margin, y_max + y_margin)

    @staticmethod
    def _matrix_value(matrix_data, row, col):
        try:
            matrix = np.asarray(matrix_data, dtype=float)
        except (TypeError, ValueError):
            return math.nan

        if matrix.ndim != 2:
            return math.nan
        if matrix.shape[0] <= row or matrix.shape[1] <= col:
            return math.nan

        value = float(matrix[row, col])
        return value if math.isfinite(value) else math.nan

    @staticmethod
    def _to_finite_or_nan(value):
        try:
            value = float(value)
        except (TypeError, ValueError):
            return math.nan

        return value if math.isfinite(value) else math.nan

    @staticmethod
    def _to_finite_or_none(value):
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None

        return value if math.isfinite(value) else None

    def set_reference_path(self, ref_path, position_equal_aspect=None):
        """Replace the displayed reference path with the controller's path."""
        ref = np.asarray(ref_path, dtype=float)
        if ref.ndim != 2 or ref.shape[0] < 2 or ref.shape[1] < 2:
            raise ValueError("ref_path must contain at least two finite x/y points")
        if not np.all(np.isfinite(ref[:, :2])):
            raise ValueError("ref_path x/y values must be finite")

        self.ref_x = ref[:, 0].copy()
        self.ref_y = ref[:, 1].copy()
        self.reference_path_line.set_data(self.ref_x, self.ref_y)
        self._rebuild_reference_geometry()

        if position_equal_aspect is not None:
            self.position_equal_aspect_requested = bool(
                position_equal_aspect
            )
        self._refresh_position_aspect()
        self._apply_position_aspect()
        self._set_position_limits()

    def clear_history(self):
        """Clear all history and immediately remove every old plot trace."""

        for history in (
            self.time,
            self.actual_x,
            self.actual_y,
            self.lateral_error,
            self.heading_error,
            self.left_cmd,
            self.right_cmd,
            self.left_feedback,
            self.right_feedback,
            self.vehicle_speed,
            self.nonlinear_observed_u,
            self.nonlinear_observed_r,
            self.nonlinear_estimated_u,
            self.nonlinear_estimated_r,
            self.a00_history,
            self.a11_history,
            self.b00_history,
            self.b11_history,
        ):
            history.clear()

        for line in (
            self.actual_path_line,
            self.current_position_marker,
            self.lateral_error_line,
            self.heading_error_line,
            self.left_cmd_line,
            self.right_cmd_line,
            self.left_feedback_line,
            self.right_feedback_line,
            self.vehicle_speed_line,
            self.a00_line,
            self.a11_line,
            self.b00_line,
            self.b11_line,
            self.obs_u_line,
            self.est_u_line,
            self.obs_r_line,
            self.est_r_line,
        ):
            line.set_data([], [])

        self._set_position_limits()
        self.ax_lateral_error.set_xlim(0.0, 1.0)
        self.ax_lateral_error.set_ylim(-0.05, 0.05)
        self.ax_heading_error.set_xlim(0.0, 1.0)
        self.ax_heading_error.set_ylim(-0.05, 0.05)
        for axis in (
            self.ax_control,
            self.ax_speed,
            self.ax_a_params,
            self.ax_b_params,
            self.ax_nonlinear_u,
            self.ax_nonlinear_r,
        ):
            axis.set_xlim(0.0, 1.0)
            axis.relim()
            axis.autoscale_view(scalex=False, scaley=True)

        self.fig_path.canvas.draw_idle()
        self.fig_path.canvas.flush_events()
        self.fig_motion.canvas.draw_idle()
        self.fig_motion.canvas.flush_events()
        self.plt.pause(0.001)

    def append_sample(
        self,
        time_sec,
        local_x,
        local_y,
        lateral_error=math.nan,
        heading_error=math.nan,
        left_wheel_cmd=0.0,
        right_wheel_cmd=0.0,
        left_wheel_feedback=math.nan,
        right_wheel_feedback=math.nan,
        matrix_a=None,
        matrix_b=None,
        vehicle_speed=0.0,
        nonlinear_observed_u=math.nan,
        nonlinear_observed_r=math.nan,
        nonlinear_estimated_u=math.nan,
        nonlinear_estimated_r=math.nan,
    ):
        if not self.enabled:
            return

        values = (
            time_sec,
            local_x,
            local_y,
            left_wheel_cmd,
            right_wheel_cmd,
            vehicle_speed,
        )
        try:
            if not all(math.isfinite(float(value)) for value in values):
                return
        except (TypeError, ValueError):
            return

        self.time.append(float(time_sec))
        self.actual_x.append(float(local_x))
        self.actual_y.append(float(local_y))
        self.lateral_error.append(
            self._to_finite_or_nan(lateral_error)
        )
        self.heading_error.append(
            self._to_finite_or_nan(heading_error)
        )
        self.left_cmd.append(float(left_wheel_cmd))
        self.right_cmd.append(float(right_wheel_cmd))
        self.left_feedback.append(
            self._to_finite_or_nan(left_wheel_feedback)
        )
        self.right_feedback.append(
            self._to_finite_or_nan(right_wheel_feedback)
        )
        self.vehicle_speed.append(float(vehicle_speed))

        self.nonlinear_observed_u.append(
            self._to_finite_or_nan(nonlinear_observed_u)
        )
        self.nonlinear_observed_r.append(
            self._to_finite_or_nan(nonlinear_observed_r)
        )
        self.nonlinear_estimated_u.append(
            self._to_finite_or_nan(nonlinear_estimated_u)
        )
        self.nonlinear_estimated_r.append(
            self._to_finite_or_nan(nonlinear_estimated_r)
        )

        self.a00_history.append(self._matrix_value(matrix_a, 0, 0))
        self.a11_history.append(self._matrix_value(matrix_a, 1, 1))
        self.b00_history.append(self._matrix_value(matrix_b, 0, 0))
        self.b11_history.append(self._matrix_value(matrix_b, 1, 1))

    def update(self):
        if not self.enabled or not self.time:
            return

        path_exists = self.plt.fignum_exists(self.fig_path.number)
        motion_exists = self.plt.fignum_exists(self.fig_motion.number)
        estimation_exists = self.plt.fignum_exists(
            self.fig_estimation.number
        )

        if not path_exists:
            self.enabled = False
            return

        self.motion_visible = self.motion_visible and motion_exists
        self.estimation_visible = (
            self.estimation_visible and estimation_exists
        )

        time_values = np.asarray(self.time, dtype=float)
        x_values = np.asarray(self.actual_x, dtype=float)
        y_values = np.asarray(self.actual_y, dtype=float)
        lateral_error_values = np.asarray(
            self.lateral_error,
            dtype=float,
        )
        heading_error_values = np.asarray(
            self.heading_error,
            dtype=float,
        )
        left_values = np.asarray(self.left_cmd, dtype=float)
        right_values = np.asarray(self.right_cmd, dtype=float)
        left_feedback_values = np.asarray(self.left_feedback, dtype=float)
        right_feedback_values = np.asarray(self.right_feedback, dtype=float)
        speed_values = np.asarray(self.vehicle_speed, dtype=float)
        obs_u_values = np.asarray(self.nonlinear_observed_u, dtype=float)
        obs_r_values = np.asarray(self.nonlinear_observed_r, dtype=float)
        est_u_values = np.asarray(self.nonlinear_estimated_u, dtype=float)
        est_r_values = np.asarray(self.nonlinear_estimated_r, dtype=float)
        a00_values = np.asarray(self.a00_history, dtype=float)
        a11_values = np.asarray(self.a11_history, dtype=float)
        b00_values = np.asarray(self.b00_history, dtype=float)
        b11_values = np.asarray(self.b11_history, dtype=float)

        self.actual_path_line.set_data(x_values, y_values)
        self.current_position_marker.set_data(
            [x_values[-1]],
            [y_values[-1]],
        )
        self._set_position_limits(x_values, y_values)

        self.lateral_error_line.set_data(
            time_values,
            lateral_error_values,
        )
        self._autoscale_error_axis(
            self.ax_lateral_error,
            time_values,
            lateral_error_values,
            default_half_span=0.05,
        )

        self.heading_error_line.set_data(
            time_values,
            heading_error_values,
        )
        self._autoscale_error_axis(
            self.ax_heading_error,
            time_values,
            heading_error_values,
            default_half_span=0.05,
        )

        self.fig_path.canvas.draw_idle()
        self.fig_path.canvas.flush_events()

        combined_needs_draw = False

        if self.motion_visible:
            self.left_cmd_line.set_data(time_values, left_values)
            self.right_cmd_line.set_data(time_values, right_values)
            self.left_feedback_line.set_data(time_values, left_feedback_values)
            self.right_feedback_line.set_data(time_values, right_feedback_values)
            self._autoscale_time_axis(self.ax_control, time_values)

            self.vehicle_speed_line.set_data(time_values, speed_values)
            self._autoscale_time_axis(self.ax_speed, time_values)
            combined_needs_draw = True

        if self.estimation_visible:
            self.a00_line.set_data(time_values, a00_values)
            self.a11_line.set_data(time_values, a11_values)
            self._autoscale_time_axis(self.ax_a_params, time_values)

            self.b00_line.set_data(time_values, b00_values)
            self.b11_line.set_data(time_values, b11_values)
            self._autoscale_time_axis(self.ax_b_params, time_values)

            self.obs_u_line.set_data(time_values, obs_u_values)
            self.est_u_line.set_data(time_values, est_u_values)
            self._autoscale_time_axis(self.ax_nonlinear_u, time_values)

            self.obs_r_line.set_data(time_values, obs_r_values)
            self.est_r_line.set_data(time_values, est_r_values)
            self._autoscale_time_axis(self.ax_nonlinear_r, time_values)
            combined_needs_draw = True

        if combined_needs_draw:
            self.fig_motion.canvas.draw_idle()
            self.fig_motion.canvas.flush_events()

        self.plt.pause(0.001)

    def _set_time_axis(self, axis, time_values):
        time_end = float(time_values[-1])
        time_start = max(
            float(time_values[0]),
            time_end - self.time_window_sec,
        )

        if time_end <= time_start:
            time_end = time_start + 1.0

        axis.set_xlim(time_start, time_end)

    def _autoscale_time_axis(self, axis, time_values):
        self._set_time_axis(axis, time_values)
        axis.relim()
        axis.autoscale_view(
            scalex=False,
            scaley=True,
        )


    def _autoscale_error_axis(
        self,
        axis,
        time_values,
        error_values,
        default_half_span=0.05,
    ):
        self._set_time_axis(axis, time_values)

        finite_values = np.asarray(error_values, dtype=float)
        finite_values = finite_values[np.isfinite(finite_values)]

        if finite_values.size == 0:
            axis.set_ylim(-default_half_span, default_half_span)
            return

        value_min = float(np.min(finite_values))
        value_max = float(np.max(finite_values))

        if value_max <= value_min:
            margin = max(abs(value_min) * 0.10, default_half_span)
        else:
            margin = max(0.10 * (value_max - value_min), 1.0e-4)

        axis.set_ylim(value_min - margin, value_max + margin)

    def save_figures(self, output_base_path):
        """Save current dashboard figures next to the output workbook."""
        from pathlib import Path

        base = Path(output_base_path).expanduser()
        if base.suffix:
            base = base.with_suffix("")

        saved_paths = []
        figures = (
            (self.fig_path, "figure1_path_tracking"),
            (self.fig_motion, "figure2_combined_control_estimation"),
        )

        for fig, suffix in figures:
            try:
                if self.plt.fignum_exists(fig.number):
                    path = base.with_name(f"{base.name}_{suffix}.png")
                    fig.savefig(path, dpi=150, bbox_inches="tight")
                    saved_paths.append(path)
            except Exception as exc:
                print(f"Failed to save figure {suffix}: {exc}")

        return saved_paths

    def close(self):
        self.enabled = False
        for fig in (
            self.fig_path,
            self.fig_motion,
        ):
            try:
                if self.plt.fignum_exists(fig.number):
                    self.plt.close(fig)
            except Exception:
                pass
