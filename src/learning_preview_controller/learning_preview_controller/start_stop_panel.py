class StartStopPanel:
    """Small non-blocking Matplotlib panel owned by the control node."""

    def __init__(
        self,
        initial_enabled=False,
        on_toggle=None,
        title="Controller Start / Stop",
    ):
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button

        self.plt = plt
        self.on_toggle = on_toggle
        self.enabled = True
        self.calculate_enabled = bool(initial_enabled)

        self.plt.ion()
        self.fig = self.plt.figure(
            num=title,
            figsize=(5.2, 1.8),
        )
        self.fig.subplots_adjust(
            left=0.04,
            right=0.96,
            bottom=0.12,
            top=0.90,
        )
        self.fig.canvas.mpl_connect(
            "close_event",
            self._on_close,
        )

        self.status_text = self.fig.text(
            0.5,
            0.72,
            "",
            ha="center",
            va="center",
            fontsize=11,
        )
        self.hint_text = self.fig.text(
            0.5,
            0.48,
            "Vehicle state is still received while stopped; only safe stop commands are published.",
            ha="center",
            va="center",
            fontsize=8,
        )

        self.button_axis = self.fig.add_axes(
            [0.30, 0.13, 0.40, 0.24]
        )
        self.button = Button(
            self.button_axis,
            "",
        )
        self.button.on_clicked(self._on_clicked)

        self._refresh()
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        self.plt.show(block=False)

    def _on_close(self, _event):
        self.enabled = False

    def _on_clicked(self, _event):
        desired_state = not self.calculate_enabled
        if self.on_toggle is not None:
            try:
                self.on_toggle(desired_state)
            except Exception as exc:
                print(
                    "Failed to update controller calculation state:",
                    exc,
                )
                return
        else:
            self.set_enabled(desired_state)

    def set_enabled(self, enabled):
        self.calculate_enabled = bool(enabled)
        self._refresh()

    def _refresh(self):
        if self.calculate_enabled:
            button_text = "Stop output()"
            status = "Control calculation: ON"
        else:
            button_text = "Start output()"
            status = "Control calculation: OFF"

        self.button.label.set_text(button_text)
        self.status_text.set_text(status)

    def update(self):
        if not self.enabled:
            return
        try:
            if not self.plt.fignum_exists(self.fig.number):
                self.enabled = False
                return
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
            self.plt.pause(0.001)
        except Exception:
            self.enabled = False

    def close(self):
        self.enabled = False
        try:
            if self.plt.fignum_exists(self.fig.number):
                self.plt.close(self.fig)
        except Exception:
            pass
