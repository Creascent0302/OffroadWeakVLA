import numpy as np
from scipy.linalg import solve_discrete_are
from datetime import datetime


class PreviewLQRMixin:
    def CalPaperPreviewMatrix(self, r, C_R):
            """
            3-state preview LQR error model.

            State:
                x_p = [e_y;
                       e_psi;
                       e_psi_dot]

            Control:
                W = [W_l(k+1);
                     W_r(k+1)]

            Curvature:
                kappa(k) is scalar C_R.

            Model:
                x_p(k+1) = A_p x_p(k) + B_p W_c(k+1) + D_p kappa(k)

            Optional:
                if debug.use_df_R_dr_in_lqr_A == true,
                put learned df_R_dr into Riccati A_p.
            """

            Ur = max(float(self.u_r), 0.1)
            Ts = float(self.T)

            A_UR = np.asarray(self.A, dtype=float)
            B_UR = np.asarray(self.B, dtype=float)

            a22 = float(A_UR[1, 1])
            b21 = float(B_UR[1, 0])
            b22 = float(B_UR[1, 1])

            # 3-state error vector:
            #   x_p = [e_y, e_psi, e_psi_dot]^T
            A_p = np.array([
                [1.0, Ts * Ur, 0.0],
                [0.0, 1.0,     Ts],
                [0.0, 0.0,     a22],
            ], dtype=float)

            B_p = np.array([
                [0.0, 0.0],
                [0.0, 0.0],
                [b21, b22],
            ], dtype=float)

            # e_psi_dot = r - U_r * kappa
            #
            # r(k+1) = a22*r + B_r*W
            #
            # e_psi_dot(k+1)
            #   = r(k+1) - U_r*kappa
            #   = a22*e_psi_dot + B_r*W + U_r*(a22 - 1)*kappa
            D_p = np.array([
                [0.0],
                [0.0],
                [Ur * (a22 - 1.0)],
            ], dtype=float)

            # ==========================================================
            # Optional trigger:
            #
            #   debug:
            #     use_df_R_dr_in_lqr_A: true
            #
            # If true, insert learned df_R_dr into Riccati A_p.
            #
            # Keep useDisturbance as the master switch: if disturbance
            # learning is off, df_R_dr will not be used here either.
            # ==========================================================
            if self.useDisturbance and bool(getattr(self, "use_df_R_dr_in_lqr_A", False)):
                r0 = Ur * float(C_R)
                df_R_dr = float(self.yaw_residual_slope(r0))

                # Optional safety clamp:
                #   df_R_dr_clip: 0.0 means no extra clamp here.
                # yaw_residual_slope() itself is already clipped by L_n.
                df_clip = float(getattr(self, "df_R_dr_clip", 0.0))
                if df_clip > 0.0:
                    df_R_dr = float(np.clip(df_R_dr, -df_clip, df_clip))

                # 3-state version: e_psi_dot is index 2.
                # A_p[2, 2] += df_R_dr  
                # D_p[2, 0] += Ur * df_R_dr  #A D 矩阵仅依赖 参数更新 

                self.last_df_R_dr_lqr = df_R_dr

                if bool(getattr(self, "debug_lqr", False)):
                    print(
                        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                        f"df_R_dr inserted into Riccati A_p: "
                        f"df_R_dr={df_R_dr:.6f}, A_p[2,2]={A_p[2,2]:.6f}"
                    )

            return A_p, B_p, D_p

    def _debug_lqr_condition(self, A_p, B_p):
            """
            Print Riccati-related diagnostics before solve_discrete_are().
            """

            A_p = np.asarray(A_p, dtype=float)
            B_p = np.asarray(B_p, dtype=float)

            n = A_p.shape[0]

            ctrb = B_p.copy()
            Ak = np.eye(n)

            for _ in range(1, n):
                Ak = A_p @ Ak
                ctrb = np.hstack((ctrb, Ak @ B_p))

            eigA = np.linalg.eigvals(A_p)
            rank_ctrb = np.linalg.matrix_rank(ctrb)
            cond_ctrb = np.linalg.cond(ctrb)

            print(
                f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                f"LQR debug: eig(A)={eigA}, "
                f"rank_ctrb={rank_ctrb}/{n}, "
                f"cond_ctrb={cond_ctrb:.3e}, "
                f"A_last={A_p[-1, -1]:.6f}, "
                f"B_last_row={B_p[-1, :]}"
            )

    def CalPreviewLQRGain(self, A_p, B_p, D_p, N):
            """
            Analytic preview LQR gain.

            System:
                x(k+1) = A_p x(k) + B_p u(k) + D_p kappa(k)

            Preview vector:
                C_R(k) = [kappa(k), kappa(k+1), ..., kappa(k+N)]'

            Control:
                u(k) = -K_b x(k) - K_f C_R(k)
            """

            A_p = np.asarray(A_p, dtype=float)
            B_p = np.asarray(B_p, dtype=float)

            n = A_p.shape[0]
            D_p = np.asarray(D_p, dtype=float).reshape(n, 1)

            N = int(N)

            Q_lqr = np.asarray(self.Q, dtype=float)
            R_lqr = np.asarray(self.R, dtype=float)

            if Q_lqr.shape != (n, n):
                raise ValueError(
                    f"Q must be {n}x{n} for preview state "
                    f"[e_y, e_psi, e_psi_dot], got {Q_lqr.shape}"
                )

            if R_lqr.shape != (2, 2):
                raise ValueError(
                    f"R must be 2x2 for input [W_l, W_r], got {R_lqr.shape}"
                )

            R_lqr = R_lqr + 1e-8 * np.eye(2)

            if bool(getattr(self, "debug_lqr", False)):
                self._debug_lqr_condition(A_p, B_p)

            P = solve_discrete_are(A_p, B_p, Q_lqr, R_lqr)

            S = R_lqr + B_p.T @ P @ B_p
            S = S + 1e-8 * np.eye(S.shape[0])

            K_b = np.linalg.solve(
                S,
                B_p.T @ P @ A_p
            )

            A_cl_T = (A_p - B_p @ K_b).T

            K_f = np.zeros((2, N + 1), dtype=float)

            A_power = np.eye(n)

            for i in range(N + 1):
                K_f[:, i:i + 1] = np.linalg.solve(
                    S,
                    B_p.T @ A_power @ P @ D_p
                )

                A_power = A_cl_T @ A_power

            return K_b, K_f

    def VehiclePreviewLQRControl_WithGComp(
            self,
            paper_state,
            kappa_preview,
            g,
            last_w_l,
            last_w_r,
            r,
            u
        ):
            """
            3-state preview LQR + learned g compensation.

            State:
                x_p = [e_y;
                       e_psi;
                       e_psi_dot]

            Linear preview part:
                x_p(k+1) = A_p x_p(k) + B_p W_c(k+1) + D_p kappa(k)

                W_c = -K_b x_p - K_f C_R
            """

            Nu = 2
            N = int(self.Np)

            x_p = np.asarray(paper_state, dtype=float).reshape(3, 1)

            C_R = np.asarray(kappa_preview, dtype=float).reshape(-1, 1)

            if C_R.shape[0] < N + 1:
                C_R = np.vstack([
                    C_R,
                    np.tile(C_R[-1:, :], (N + 1 - C_R.shape[0], 1))
                ])

            C_R = C_R[:N + 1, :]

            # 1. 3-state linear preview model
            A_p, B_p, D_p = self.CalPaperPreviewMatrix(r, C_R[0, 0])

            # 2. Analytic preview LQR gains
            try:
                K_b, K_f = self.CalPreviewLQRGain(
                    A_p=A_p,
                    B_p=B_p,
                    D_p=D_p,
                    N=N
                )

                self.last_valid_K_b = K_b
                self.last_valid_K_f = K_f

            except Exception as e:
                print(
                    f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                    f"Preview LQR gain calculation failed: {e}"
                )

                if hasattr(self, "last_valid_K_b") and hasattr(self, "last_valid_K_f"):
                    K_b = self.last_valid_K_b
                    K_f = self.last_valid_K_f

                    print(
                        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                        f"Use last valid LQR gain"
                    )
                else:
                    w_last = np.array([
                        [float(last_w_l)],
                        [float(last_w_r)]
                    ], dtype=float)

                    U_value = np.tile(w_last.reshape(-1), self.Nc)

                    return float(last_w_l), float(last_w_r), U_value, "PREVIEW_LQR_FAILED"

            # 3. Straight-line base wheel speed.
            kappa_now = float(C_R[0, 0])
            W_base = self.solve_base_wheel_speed_UR(kappa_now, g,u,r)

            # 4. Preview feedback/feedforward correction
            if self.using_preview:
                # W_c = -K_b @ x_p -2.5*K_f @ C_R
                W_c = -K_b @ x_p -4.5*K_f @ C_R

                # print("K_f=",K_f)

            else:
                # Preview OFF:
                # Keep the outer W_base unchanged because it is the base
                # velocity-control term. Add only the differential-drive
                # kinematic wheel-speed feedforward into the correction term.
                W_f = self.solve_kinematic_wheel_speed_ff(kappa_now)
                W_c = -K_b @ x_p + W_f
            # print("K_b=",K_b)

            # print("preview_control=",-K_f @ C_R)
            # print("Feed_back=",-K_b @ x_p)
            # print("W_c=",W_c)
            # 5. g compensation
            if self.g_comp_mode == "yaw":
                W_g = self.CalGCompensation_YawOnly(g)
            else:
                W_g = self.CalGCompensation(g)

            # Current logic:
            # g has already been considered in W_base, so do not add W_g again.
            #
            # If you want explicit g compensation later, change to:
            #     W_cmd = W_base + W_c + W_g
   
            W_cmd = W_base + W_c

            # 6. Delta wheel-speed saturation
            w_last = np.array([
                [float(last_w_l)],
                [float(last_w_r)]
            ], dtype=float)

            dW_max = self._expand_bound_vec(self.Delta_InputMax, Nu)

            W_cmd = np.minimum(
                np.maximum(W_cmd, w_last - dW_max),
                w_last + dW_max
            )

            # 7. Absolute wheel-speed saturation
            lb = self._expand_bound_vec(self.lb_temp, Nu)
            ub = self._expand_bound_vec(self.ub_temp, Nu)

            W_cmd = np.clip(W_cmd, lb, ub)

            w_l = float(W_cmd[0, 0])
            w_r = float(W_cmd[1, 0])

            U_value = np.tile(W_cmd.reshape(-1), self.Nc)

            # Debug signals
            self.checkU = float(W_base[0, 0])
            self.checkR = float(W_base[1, 0])

            return w_l, w_r, U_value, "PREVIEW_LQR_G_COMP"

    def solve_kinematic_wheel_speed_ff(self, kappa_ref):
        v_ref = float(self.u_r)
        yaw_rate_ref = v_ref * float(kappa_ref)

        track_width = float(getattr(self, "L", 0.0))
        wheel_radius = float(getattr(self, "wheel_radius", 0.0))

        if abs(wheel_radius) < 1e-8:
            raise ValueError(
                "wheel_radius must be non-zero for kinematic wheel-speed feedforward"
            )

        w_diff = 0.5 * track_width * yaw_rate_ref / wheel_radius

        # 只输出差速项，不包含 v_ref / wheel_radius 的前进速度
        w_l_delta = -w_diff
        w_r_delta = +w_diff


        return np.array([[w_l_delta], [w_r_delta]], dtype=float)


    def yaw_residual_hat(self, r_query):
            """
            Learned yaw-rate residual:
                f_R = fhat_R(r)

            Current implementation learns f_R as a scalar function of r only.
            """
            return self.fhatpre_scalar(
                r_query,
                self.L_n,
                self.SampleSetState[1, :],
                self.SampleSetGy[1, :],
                self.e
            )

    def yaw_residual_slope(self, r0, h=None):
            """
            Numerical derivative df_R / dr at r0.

            This corresponds to the paper-style generalized derivative:
                if left/right derivatives differ, use their average.

            Since fhatpre_scalar is KI/LACKI-like and piecewise nonsmooth,
            central difference is the practical implementation.
            """
            r0 = float(r0)

            if h is None:
                h = 1e-3 * max(1.0, abs(r0))

            h = max(float(h), 1e-5)

            fp = float(self.yaw_residual_hat(r0 + h))
            fm = float(self.yaw_residual_hat(r0 - h))

            df_dr = (fp - fm) / (2.0 * h)

            # Lipschitz bound safety clamp
            df_dr = float(np.clip(df_dr, -abs(self.L_n), abs(self.L_n)))

            return df_dr
    def solve_base_wheel_speed_UR(self, kappa_ref, g, u, r):
        """
        Solve common-mode base wheel speed according to the paper.

        Paper method:
            U_v,k = [omega_base,k, omega_base,k]^T

        Longitudinal learned model:
            v_x,k+1 = a_v v_x,k + b_v,L omega_L,k
                    + b_v,R omega_R,k + f_vx,k

        For base speed:
            omega_L = omega_R = omega_base

        Steady-speed condition:
            v_x,k+1 = v_x,k = v_r

        Therefore:
            omega_base = ((1 - a_v) * v_r - f_vx_hat)
                        / (b_v,L + b_v,R)

        Road curvature is not handled here.
        It should be handled by preview feedforward / differential input.
        """
        Ur = float(self.u_r)

        A_UR = np.asarray(self.A, dtype=float)
        B_UR = np.asarray(self.B, dtype=float)

        # Learned longitudinal coefficients
        a_v = float(A_UR[0, 0])
        b_v_L = float(B_UR[0, 0])
        b_v_R = float(B_UR[0, 1])

        # ===== 前几个时间步不使用 learned g =====
        use_g_now = self.step_count >= self.g_warmup_steps

        if use_g_now and g is not None:
            g_arr = np.asarray(g, dtype=float)
            f_vx_hat = float(g_arr[0, 0])
        else:
            f_vx_hat = 0.0

        self.step_count += 1

        denom = b_v_L + b_v_R

        # Avoid division by zero or ill-conditioned base computation
        if abs(denom) < 1e-8:
            omega_base = 0.0
        else:
            # omega_base = ((1.0 - a_v) * Ur - f_vx_hat) / denom
            omega_base = ((1.0 - a_v) * Ur) / denom # 稳态前馈
            # omega_base = (1.0 * Ur - a_v * u) / denom # 偏差反馈

        W_base = np.array([[omega_base], [omega_base]], dtype=float)

        lb = self._expand_bound_vec(self.lb_temp, 2)
        ub = self._expand_bound_vec(self.ub_temp, 2)
        W_base = np.clip(W_base, lb, ub)

        return W_base

    # def solve_base_wheel_speed_UR(self,kappa_ref,g,u,r):
    #         """
    #         Solve straight-line base absolute wheel speed W_base.

    #         Desired steady U-R state:
    #             U_ref = U_r
    #             R_ref = 0

    #         Learned model:
    #             X_ref = A_UR X_ref + B_UR W_base

    #         This base input is only for maintaining forward speed and cancelling
    #         constant coupling terms. Road curvature is left to preview feedforward.
    #         """
    #         Ur = float(self.u_r)
    #         # w_temp=float(10.0*(Ur-u))

    #         Rr = 0.0

    #         x_ref = np.array([[Ur], [Rr]], dtype=float)
    #         x_now = np.array([[u], [r]], dtype=float)
    #         A_UR = np.asarray(self.A, dtype=float)
    #         B_UR = np.asarray(self.B, dtype=float)
    #         # ===== 前几个时间步不使用 learned g =====
    #         use_g_now = self.step_count >= self.g_warmup_steps

    #         if use_g_now:
    #             temp = np.array([
    #                 [float(g[0, 0])],
    #                 [float(0.0)]
    #             ], dtype=float)
    #         else:
    #             temp = np.zeros((2, 1), dtype=float)
    #         self.step_count += 1

    #         rhs = x_ref - A_UR @ x_now - temp


    #         if np.linalg.matrix_rank(B_UR) >= 2:
    #             W_base = np.linalg.solve(B_UR, rhs)
    #         else:
    #             W_base = np.linalg.lstsq(B_UR, rhs, rcond=None)[0]

    #         lb = self._expand_bound_vec(self.lb_temp, 2)
    #         ub = self._expand_bound_vec(self.ub_temp, 2)
    #         # W_base[0] = w_temp
    #         # W_base[1] = w_temp
    #         W_base = np.clip(W_base, lb, ub)

    #         return W_base

    def CalGCompensation(self, g):
            """
            Compensate learned nonlinear residual g in U-R model.

            Learned model:
                X_UR(k+1) = A_UR X_UR(k) + B_UR W(k+1) + g(k)

            Compensation:
                B_UR W_g + g ≈ 0

            Thus:
                W_g = -pinv(B_UR) g
            """

            g = np.asarray(g, dtype=float).reshape(2, 1)

            B_UR = np.asarray(self.B, dtype=float)

            if B_UR.shape != (2, 2):
                raise ValueError(f"B_UR must be 2x2, got {B_UR.shape}")

            lam = 1e-6

            W_g = -B_UR.T @ np.linalg.inv(
                B_UR @ B_UR.T + lam * np.eye(2)
            ) @ g

            return W_g

    def CalGCompensation_YawOnly(self, g):
            """
            Only compensate yaw-rate residual f_R.

            b_R W_g + f_R ≈ 0
            """

            g = np.asarray(g, dtype=float).reshape(2, 1)

            f_R = float(g[1, 0])

            B_UR = np.asarray(self.B, dtype=float)
            b_R = B_UR[1:2, :]

            lam = 1e-6

            denom = float(b_R @ b_R.T + lam)
            W_g = -b_R.T * f_R / denom

            return W_g

    def GetPreviewCurvature(self, ID, N):
            """
            Get preview curvature vector:
                C_R(k) = [kappa(k), kappa(k+1), ..., kappa(k+N)]'

            The index step is approximately:
                preview distance per sample = U_r * T
            """

            ID = int(ID)
            N = int(N)

            ref_path = self.ref_path
            m = ref_path.shape[0]

            if m < 2:
                return np.zeros((N + 1, 1), dtype=float)

            dx = np.diff(ref_path[:, 0])
            dy = np.diff(ref_path[:, 1])
            ds = np.sqrt(dx ** 2 + dy ** 2)

            ds_mean = float(np.mean(ds))
            ds_mean = max(ds_mean, 1e-6)

            preview_dist_step = abs(float(self.u_r)) * float(self.T)

            index_step = int(round(preview_dist_step / ds_mean))
            index_step = max(index_step, 1)

            ids = ID + index_step * np.arange(N + 1)
            ids = np.clip(ids, 0, m - 1).astype(int)

            kappa_preview = ref_path[ids, 3].reshape(N + 1, 1)

            return kappa_preview
