import numpy as np


class PathMixin:
    def NormalizeRefPathStart(self, ref_path):
        """Transform a generated reference path so its first pose is (0, 0, 0).

        Expected columns:
            0: x, m
            1: y, m
            2: heading, rad
            3: curvature, 1/m

        Curvature and any additional columns are preserved.
        """

        ref_path = np.asarray(
            ref_path,
            dtype=float,
        ).copy()

        if ref_path.ndim != 2:
            raise ValueError(
                "ref_path must be a 2-D array"
            )

        if ref_path.shape[0] == 0:
            raise ValueError(
                "ref_path must not be empty"
            )

        if ref_path.shape[1] < 4:
            raise ValueError(
                "ref_path must contain at least "
                "[x, y, heading, curvature]"
            )

        x0 = float(ref_path[0, 0])
        y0 = float(ref_path[0, 1])
        heading0 = float(ref_path[0, 2])

        dx = ref_path[:, 0] - x0
        dy = ref_path[:, 1] - y0

        c = np.cos(heading0)
        s = np.sin(heading0)

        # Rotate by -heading0.
        ref_path[:, 0] = c * dx + s * dy
        ref_path[:, 1] = -s * dx + c * dy

        heading_local = ref_path[:, 2] - heading0
        ref_path[:, 2] = np.arctan2(
            np.sin(heading_local),
            np.cos(heading_local),
        )

        return ref_path

    def GenerateDoubleLaneChangeRef(self):
            path = self.path_cfg

            X_start = float(path["X_start"])
            X_end = float(path["X_end"])

            # 原始密集采样间隔，用来计算弧长，建议比目标 ds 小很多
            dX_dense = float(path.get("dX_dense", 0.01))

            # 期望车速和采样周期
            v_ref = float(path["v_ref"])   # m/s
            Ts = float(path["Ts"])         # s

            ds = v_ref * Ts                # 每个参考点的期望路径长度间隔

            shape = float(path["shape"])
            dx1 = float(path["dx1"])
            dx2 = float(path["dx2"])
            dy1 = float(path["dy1"])
            dy2 = float(path["dy2"])
            Xs1 = float(path["Xs1"])
            Xs2 = float(path["Xs2"])

            def eval_path(X_ref):
                z1 = shape / dx1 * (X_ref - Xs1) - shape / 2.0
                z2 = shape / dx2 * (X_ref - Xs2) - shape / 2.0

                sech1_sq = 1.0 / np.cosh(z1) ** 2
                sech2_sq = 1.0 / np.cosh(z2) ** 2

                Y_ref = (
                    dy1 / 2.0 * (1.0 + np.tanh(z1))
                    - dy2 / 2.0 * (1.0 + np.tanh(z2))
                )

                dYdX = (
                    dy1 / 2.0 * sech1_sq * shape / dx1
                    - dy2 / 2.0 * sech2_sq * shape / dx2
                )

                phi_ref = np.arctan(dYdX)

                d2YdX2 = (
                    -dy1 * sech1_sq * np.tanh(z1) * (shape / dx1) ** 2
                    + dy2 * sech2_sq * np.tanh(z2) * (shape / dx2) ** 2
                )

                kappa_ref = d2YdX2 / (1.0 + dYdX ** 2) ** 1.5

                return Y_ref, phi_ref, kappa_ref, dYdX

            # 1. 先生成密集 X
            X_dense = np.arange(X_start, X_end + dX_dense, dX_dense)

            # 2. 计算密集路径上的 Y、航向角、曲率
            Y_dense, phi_dense, kappa_dense, dYdX_dense = eval_path(X_dense)

            # 3. 计算相邻点之间的弧长
            dS = np.sqrt(np.diff(X_dense) ** 2 + np.diff(Y_dense) ** 2)

            # 4. 累计弧长
            S_dense = np.concatenate(([0.0], np.cumsum(dS)))

            # 5. 按固定弧长 ds 采样
            S_sample = np.arange(0.0, S_dense[-1], ds)

            # 如果想保证终点一定包含进去
            if S_sample[-1] < S_dense[-1]:
                S_sample = np.append(S_sample, S_dense[-1])

            # 6. 根据弧长反查对应的 X
            X_ref = np.interp(S_sample, S_dense, X_dense)

            # 7. 用解析公式重新计算 Y、phi、kappa
            Y_ref, phi_ref, kappa_ref, _ = eval_path(X_ref)

            ref_path = np.column_stack((X_ref, Y_ref, phi_ref, kappa_ref))

            return ref_path

    def GenerateSineRef(self):
            path = self.path_cfg

            # =========================
            # 基本参数
            # =========================
            v_ref = float(path["v_ref"])      # 期望车速, m/s
            Ts = float(path["Ts"])            # 采样时间, s

            ds = v_ref * Ts                   # 定长采样间隔, m

            if ds <= 0.0:
                raise ValueError("ds = v_ref * Ts must be positive")

            # 前后直线，可选
            L_straight = float(path["L_straight"])   # 正弦段前直线长度, m
            L_end = float(path["L_end"])             # 正弦段后直线长度, m

            # 正弦路径参数
            L_sine = float(path["L_sine"])                    # 正弦段在 X 方向的长度, m
            A = float(path["A"])                     # 正弦幅值, m
            wave_length = float(path["wave_length"])  # 正弦波长, m

            phase_deg = float(path["phase_deg"])     # 相位角, deg
            phase = np.deg2rad(phase_deg)

            y0 = 0.0                   # 整体 Y 偏移

            turn = path.get("turn", "left")                   # "left" 或 "right"

            if turn == "left":
                direction = 1.0
            elif turn == "right":
                direction = -1.0
            else:
                raise ValueError("turn must be 'left' or 'right'")

            if L_sine <= 0.0:
                raise ValueError("L_sine must be positive")

            if wave_length <= 0.0:
                raise ValueError("wave_length must be positive")

            # 是否让正弦段起点的 Y 与前段直线连续
            # True: 令正弦段起点 Y = y0
            zero_start = bool(path["zero_start"])

            omega = 2.0 * np.pi / wave_length

            # =========================
            # 正弦函数及其导数
            # =========================
            def sine_y(x):
                if zero_start:
                    y_shift = np.sin(phase)
                else:
                    y_shift = 0.0

                return y0 + direction * A * (
                    np.sin(omega * x + phase) - y_shift
                )

            def sine_dy_dx(x):
                return direction * A * omega * np.cos(omega * x + phase)

            def sine_d2y_dx2(x):
                return -direction * A * omega ** 2 * np.sin(omega * x + phase)

            # =========================
            # 建立 正弦段 x -> 弧长 s 的映射
            # 用于实现按弧长定长采样
            # =========================
            dense_num = int(path["dense_num"])
            dense_num = max(dense_num, 1000)

            x_dense = np.linspace(0.0, L_sine, dense_num)
            y_dense = sine_y(x_dense)

            dx_dense = np.diff(x_dense)
            dy_dense = np.diff(y_dense)

            ds_dense = np.sqrt(dx_dense ** 2 + dy_dense ** 2)

            s_dense = np.zeros_like(x_dense)
            s_dense[1:] = np.cumsum(ds_dense)

            L_sine_arc = s_dense[-1]   # 正弦段真实弧长

            # =========================
            # 路径总长度
            # =========================
            L_total = L_straight + L_sine_arc + L_end

            s_ref = np.arange(0.0, L_total + 1e-9, ds)

            # 保证终点被包含
            if s_ref[-1] < L_total:
                s_ref = np.append(s_ref, L_total)

            # =========================
            # 初始化
            # =========================
            X_ref = np.zeros_like(s_ref)
            Y_ref = np.zeros_like(s_ref)
            phi_ref = np.zeros_like(s_ref)
            kappa_ref = np.zeros_like(s_ref)

            # =========================
            # 第一段：前直线
            # 起点: (0, y0)
            # 方向: 沿 X 正方向
            # =========================
            idx_straight = s_ref <= L_straight

            X_ref[idx_straight] = s_ref[idx_straight]
            Y_ref[idx_straight] = y0
            phi_ref[idx_straight] = 0.0
            kappa_ref[idx_straight] = 0.0

            # =========================
            # 第二段：正弦路径
            # 按弧长 s 反查对应的 x
            # =========================
            idx_sine = (s_ref > L_straight) & (s_ref <= L_straight + L_sine_arc)

            s_sine = s_ref[idx_sine] - L_straight

            # 由弧长 s 插值得到对应的 x
            x_sine = np.interp(s_sine, s_dense, x_dense)

            y_sine = sine_y(x_sine)
            dy_dx = sine_dy_dx(x_sine)
            d2y_dx2 = sine_d2y_dx2(x_sine)

            X_ref[idx_sine] = L_straight + x_sine
            Y_ref[idx_sine] = y_sine

            # 航向角 phi = atan(dy/dx)
            phi_ref[idx_sine] = np.arctan2(dy_dx, 1.0)

            # 曲率 kappa = y'' / (1 + y'^2)^(3/2)
            kappa_ref[idx_sine] = d2y_dx2 / ((1.0 + dy_dx ** 2) ** 1.5)

            # =========================
            # 第三段：正弦后的直线，可选
            # 沿正弦末端切线方向继续走
            # =========================
            idx_end = s_ref > L_straight + L_sine_arc

            if np.any(idx_end):
                s_end = s_ref[idx_end] - L_straight - L_sine_arc

                x_end = L_sine
                y_end = sine_y(x_end)

                dy_dx_end = sine_dy_dx(x_end)
                phi_end = np.arctan2(dy_dx_end, 1.0)

                X_sine_end = L_straight + x_end
                Y_sine_end = y_end

                X_ref[idx_end] = X_sine_end + s_end * np.cos(phi_end)
                Y_ref[idx_end] = Y_sine_end + s_end * np.sin(phi_end)

                phi_ref[idx_end] = phi_end
                kappa_ref[idx_end] = 0.0

            ref_path = np.column_stack((X_ref, Y_ref, phi_ref, kappa_ref))

            return ref_path

    def GenerateStraightCircleRef(self):
            path = self.path_cfg

            # =========================
            # 基本参数
            # =========================
            v_ref = float(path["v_ref"])      # 期望车速, m/s
            Ts = float(path["Ts"])            # 采样时间, s

            ds = v_ref * Ts                   # 定长采样间隔, m

            L_straight = float(path["L_straight"])      # 前段直线长度, m
            L_end = float(path.get("L_end", 0.0))       # 圆弧后的直线长度, m，可选

            R = float(path.get("R", 100.0))             # 圆弧半径, 默认 100 m

            arc_angle_deg = float(path["arc_angle_deg"])    # 圆弧角度, deg
            arc_angle = np.deg2rad(abs(arc_angle_deg))       # 转成 rad

            turn = path.get("turn", "left")             # "left" 或 "right"

            if turn == "left":
                direction = 1.0
            elif turn == "right":
                direction = -1.0
            else:
                raise ValueError("turn must be 'left' or 'right'")

            if ds <= 0.0:
                raise ValueError("ds = v_ref * Ts must be positive")

            # =========================
            # 路径总长度
            # =========================
            L_arc = R * arc_angle
            L_total = L_straight + L_arc + L_end

            # 按弧长定长采样
            s_ref = np.arange(0.0, L_total + 1e-9, ds)

            # 保证终点被包含
            if s_ref[-1] < L_total:
                s_ref = np.append(s_ref, L_total)

            # =========================
            # 初始化
            # =========================
            X_ref = np.zeros_like(s_ref)
            Y_ref = np.zeros_like(s_ref)
            phi_ref = np.zeros_like(s_ref)
            kappa_ref = np.zeros_like(s_ref)

            # =========================
            # 第一段：直线
            # 起点: (0, 0)
            # 方向: 沿 X 正方向
            # =========================
            idx_straight = s_ref <= L_straight

            X_ref[idx_straight] = s_ref[idx_straight]
            Y_ref[idx_straight] = 0.0
            phi_ref[idx_straight] = 0.0
            kappa_ref[idx_straight] = 0.0

            # =========================
            # 第二段：圆弧
            # 圆弧半径 R = 100 m
            # 曲率 kappa = +/- 1/R
            # =========================
            idx_arc = (s_ref > L_straight) & (s_ref <= L_straight + L_arc)

            s_arc = s_ref[idx_arc] - L_straight
            theta = s_arc / R

            X_ref[idx_arc] = L_straight + R * np.sin(theta)
            Y_ref[idx_arc] = direction * R * (1.0 - np.cos(theta))

            phi_ref[idx_arc] = direction * theta
            kappa_ref[idx_arc] = direction / R

            # =========================
            # 第三段：圆弧后的直线，可选
            # 沿圆弧末端切线方向继续走
            # =========================
            idx_end = s_ref > L_straight + L_arc

            if np.any(idx_end):
                s_end = s_ref[idx_end] - L_straight - L_arc

                theta_end = arc_angle
                phi_end = direction * theta_end

                X_arc_end = L_straight + R * np.sin(theta_end)
                Y_arc_end = direction * R * (1.0 - np.cos(theta_end))

                X_ref[idx_end] = X_arc_end + s_end * np.cos(phi_end)
                Y_ref[idx_end] = Y_arc_end + s_end * np.sin(phi_end)

                phi_ref[idx_end] = phi_end
                kappa_ref[idx_end] = 0.0

            ref_path = np.column_stack((X_ref, Y_ref, phi_ref, kappa_ref))

            return ref_path

    def FindNearestPoint(self, ref_path, x_now, y_now, psi_now, ID_last):
            m = ref_path.shape[0]
            ID_last = int(ID_last)

            if ID_last <= 0:
                Fstart = 0
                Fend = m - 1
            else:
                Fstart = max(ID_last - 10, 0)
                Fend = min(ID_last + 50, m - 1)

            idx_range = np.arange(Fstart, Fend + 1)

            dx = ref_path[idx_range, 0] - x_now
            dy = ref_path[idx_range, 1] - y_now

            distance = np.sqrt(dx ** 2 + dy ** 2)

            psi_ref = ref_path[idx_range, 2]
            psi_err = np.abs(self.wrap_to_pi(psi_ref - psi_now))

            valid = psi_err < 0.5 * np.pi

            if np.any(valid):
                local_id = np.argmin(np.where(valid, distance, np.inf))
            else:
                local_id = np.argmin(distance)

            ID = int(idx_range[local_id])

            return m, ID

    def CalErr(self, X, Y, Psi, ID, ref_path):
            X_r = ref_path[ID, 0]
            Y_r = ref_path[ID, 1]
            Psi_r = ref_path[ID, 2]
            kappa_ref = ref_path[ID, 3]

            dx = X - X_r
            dy = Y - Y_r

            lateral_err = -np.sin(Psi_r) * dx + np.cos(Psi_r) * dy
            psi_dev = self.wrap_to_pi(Psi - Psi_r)

            return lateral_err, psi_dev, kappa_ref
