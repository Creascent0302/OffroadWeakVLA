import numpy as np


class LearningMixin:
    def update_Au(self, u, u_last_vec, g, last_u):
            """
            Update Au for U direction.

            Model:
                u = Au * last_u + B_u @ u_last_vec + g_u
            """

            u_last_vec = np.asarray(u_last_vec, dtype=float).reshape(2, 1)

            yu = float(
                u
                - float(self.B[0, :] @ u_last_vec.flatten())
                - g[0, 0]
            )

            phiu = np.array([[float(last_u)]], dtype=float)

            self.Xplus_u = self.push_window(
                self.Xplus_u,
                np.array([[yu]], dtype=float)
            )

            self.Xminus_u = self.push_window(
                self.Xminus_u,
                phiu
            )

            do_update, gram_new_u = self.should_update_param(
                Xminus=self.Xminus_u,
                gram_old=self.gram_u,
                epsilon=self.epsilon_u_l,
                name="Au"
            )

            if not do_update:
                return

            print("Au 更新触发" if self.usingTrigger else "Au 每时刻更新")

            self.Au_est = float(
                (self.Xplus_u @ np.linalg.pinv(self.Xminus_u))[0, 0]
            )

            self.A[0, 0] = self.Au_est
            self.gram_u = gram_new_u

            self.CalTrackWheelAu()

    def update_Ar(self, r, u_last_vec, g, last_r):
            """
            Update Ar for R direction.

            Model:
                r = Ar * last_r + B_r @ u_last_vec + g_r
            """

            u_last_vec = np.asarray(u_last_vec, dtype=float).reshape(2, 1)

            yr = float(
                r
                - float(self.B[1, :] @ u_last_vec.flatten())
                - g[1, 0]
            )

            phir = np.array([[float(last_r)]], dtype=float)

            self.Xplus_r = self.push_window(
                self.Xplus_r,
                np.array([[yr]], dtype=float)
            )

            self.Xminus_r = self.push_window(
                self.Xminus_r,
                phir
            )

            do_update, gram_new_r = self.should_update_param(
                Xminus=self.Xminus_r,
                gram_old=self.gram_r,
                epsilon=self.epsilon_r_l,
                name="Ar"
            )

            if not do_update:
                return

            print("Ar 更新触发" if self.usingTrigger else "Ar 每时刻更新")

            self.Ar_est = float(
                (self.Xplus_r @ np.linalg.pinv(self.Xminus_r))[0, 0]
            )

            self.A[1, 1] = self.Ar_est
            self.gram_r = gram_new_r

            self.CalTrackWheelAr()

    def update_ABr(self, r, u_last_vec, g, last_r):
            """
            Estimate Ar and Br for R direction.

            Model:
                r = Ar * last_r + Br @ u_last_vec + g_R
            """

            yr = float(r - g[1, 0])

            u_last_vec = np.asarray(u_last_vec, dtype=float).reshape(2, 1)

            phir = np.vstack((
                np.array([[float(last_r)]], dtype=float),
                u_last_vec
            ))

            self.Xplus_rAB = self.push_window(
                self.Xplus_rAB,
                np.array([[yr]], dtype=float)
            )

            self.Xminus_rAB = self.push_window(
                self.Xminus_rAB,
                phir
            )

            do_update, gram_new_rAB = self.should_update_param(
                Xminus=self.Xminus_rAB,
                gram_old=self.gram_rAB,
                epsilon=self.epsilon_rAB_l,
                name="rAB"
            )

            if not do_update:
                return

            print("Ar、Br 更新触发" if self.usingTrigger else "Ar、Br 每时刻更新")

            theta_r = self.Xplus_rAB @ np.linalg.pinv(self.Xminus_rAB)

            self.Ar_est = float(theta_r[0, 0])
            self.Br_est = theta_r[:, 1:3]

            self.A[1, 1] = self.Ar_est
            self.B[1, :] = self.Br_est.reshape(-1)

            self.gram_rAB = gram_new_rAB

    def update_ABu(self, u, u_last_vec, g, last_u):
            """
            Estimate Au and Bu for U direction.

            Model:
                u = Au * last_u + Bu @ u_last_vec + g_U
            """

            yu = float(u - g[0, 0])

            u_last_vec = np.asarray(u_last_vec, dtype=float).reshape(2, 1)

            phiu = np.vstack((
                np.array([[float(last_u)]], dtype=float),
                u_last_vec
            ))

            self.Xplus_uAB = self.push_window(
                self.Xplus_uAB,
                np.array([[yu]], dtype=float)
            )

            self.Xminus_uAB = self.push_window(
                self.Xminus_uAB,
                phiu
            )

            do_update, gram_new_uAB = self.should_update_param(
                Xminus=self.Xminus_uAB,
                gram_old=self.gram_uAB,
                epsilon=self.epsilon_uAB_l,
                name="uAB"
            )

            if not do_update:
                return

            print("Au、Bu 更新触发" if self.usingTrigger else "Au、Bu 每时刻更新")

            theta_u = self.Xplus_uAB @ np.linalg.pinv(self.Xminus_uAB)

            self.Au_est = float(theta_u[0, 0])
            self.Bu_est = theta_u[:, 1:3]

            self.A[0, 0] = self.Au_est
            self.B[0, :] = self.Bu_est.reshape(-1)

            self.gram_uAB = gram_new_uAB

    def update_AB_full(self, u, r, u_last_vec, g, last_u, last_r):
            """
            Estimate full A and B matrices.

            Model:
                x_now = A @ x_last + B @ u_last_vec + g
            """

            x_now = np.array([
                [float(u)],
                [float(r)]
            ], dtype=float)

            g = np.asarray(g, dtype=float).reshape(2, 1)

            y = x_now - g

            x_last = np.array([
                [float(last_u)],
                [float(last_r)]
            ], dtype=float)

            u_last_vec = np.asarray(u_last_vec, dtype=float).reshape(2, 1)

            phi = np.vstack((x_last, u_last_vec))

            self.Xplus_AB = self.push_window(
                self.Xplus_AB,
                y
            )

            self.Xminus_AB = self.push_window(
                self.Xminus_AB,
                phi
            )

            do_update, gram_new_AB = self.should_update_param(
                Xminus=self.Xminus_AB,
                gram_old=self.gram_AB,
                epsilon=self.epsilon_AB_l,
                name="AB_full"
            )

            if not do_update:
                return

            rank_AB = np.linalg.matrix_rank(self.Xminus_AB)

            if rank_AB < 4:
                print(f"AB 不更新：Xminus_AB rank = {rank_AB} < 4，激励不足")
                return

            print("完整 A、B 矩阵更新触发" if self.usingTrigger else "完整 A、B 每时刻更新")

            theta = self.Xplus_AB @ np.linalg.pinv(self.Xminus_AB)

            A_new = theta[:, 0:2]
            B_new = theta[:, 2:4]

            self.A = A_new.copy()
            self.B = B_new.copy()

            self.Au_est = float(self.A[0, 0])
            self.Ar_est = float(self.A[1, 1])

            self.Bu_est = self.B[0:1, :].copy()
            self.Br_est = self.B[1:2, :].copy()

            self.gram_AB = gram_new_AB

            print("self.A =")
            print(self.A)
            print("self.B =")
            print(self.B)

    def Matrix_update_learning(self, u, r, u_last_vec, g, last_u, last_r):
            """
            Matrix learning update.
            """

            if self.UsingFulllearning:
                self.update_AB_full(u, r, u_last_vec, g, last_u, last_r)
            else:
                if self.UsingYawRate is False and self.UsingABlearning is False:
                    if abs(u) > self.speed_threshold:
                        self.update_Au(u, u_last_vec, g, last_u)

                if self.UsingABlearning is True:
                    self.update_ABu(u, u_last_vec, g, last_u)
                    self.update_ABr(r, u_last_vec, g, last_r)

                if self.UsingYawRate is True and self.UsingABlearning is False:
                    if abs(u) > self.speed_threshold:
                        self.update_Au(u, u_last_vec, g, last_u)
                    if abs(r) > self.yaw_rate_threshold:
                        self.update_Ar(r, u_last_vec, g, last_r)

    def push_window(self, X_old, new_col):
            """
            Sliding data window.
            """

            new_col = np.asarray(new_col, dtype=float).reshape(-1, 1)

            return np.column_stack(
                (X_old[:, 1:], new_col)
            )

    def should_update_param(self, Xminus, gram_old, epsilon, name=""):
            """
            Decide whether parameter update is triggered.
            """

            if not np.any(Xminus):
                return False, gram_old

            if not self.usingTrigger:
                return True, gram_old

            IQII, gram_new = self.calculate_IQII_hat(
                Xminus,
                gram_old
            )

            if IQII > epsilon:
                return True, gram_new

            return False, gram_old

    def fhatpre(self, x, L, e, b):
            _, a = self.SampleSetState.shape

            mintempdata = np.zeros((self.SampleSetGy.shape[0], a))
            maxtempdata = np.zeros((self.SampleSetGy.shape[0], a))

            for j in range(a):
                diff_norm = np.linalg.norm(
                    x - self.SampleSetState[:, j:j + 1],
                    ord=np.inf
                )

                mintempdata[:, j] = self.SampleSetGy[:, j] + L * diff_norm + e
                maxtempdata[:, j] = self.SampleSetGy[:, j] - L * diff_norm - e

            un = np.zeros((b, 1))
            ln = np.zeros((b, 1))

            for i in range(b):
                un[i, 0] = np.min(mintempdata[i, :])
                ln[i, 0] = np.max(maxtempdata[i, :])

            f_pre = 0.5 * (un + ln)

            return f_pre

    def bhatpre(self, x, L, e, b):
            _, a = self.SampleSetState.shape

            mintempdata = np.zeros((self.SampleSetGy.shape[0], a))
            maxtempdata = np.zeros((self.SampleSetGy.shape[0], a))

            for j in range(a):
                diff_norm = np.linalg.norm(
                    x - self.SampleSetState[:, j:j + 1],
                    ord=np.inf
                )

                mintempdata[:, j] = self.SampleSetGy[:, j] + L * diff_norm + e
                maxtempdata[:, j] = self.SampleSetGy[:, j] - L * diff_norm - e

            un = np.zeros((b, 1))
            ln = np.zeros((b, 1))

            for i in range(b):
                un[i, 0] = np.min(mintempdata[i, :])
                ln[i, 0] = np.max(maxtempdata[i, :])

            b_pre = 0.5 * (un - ln)

            return float(np.max(b_pre))

    def fhatpre_scalar(self, x, L, sample_state, sample_gy, e):
            x = float(x)

            sample_state = np.asarray(sample_state, dtype=float).reshape(-1)
            sample_gy = np.asarray(sample_gy, dtype=float).reshape(-1)

            diff = np.abs(x - sample_state)

            upper = sample_gy + L * diff + e
            lower = sample_gy - L * diff - e

            un = np.min(upper)
            ln = np.max(lower)

            return float(0.5 * (un + ln))

    def calculate_IQII_hat(self, phi, gram_old):
            phi = np.asarray(phi, dtype=float)

            G_new = phi @ phi.T

            if G_new.shape == (1, 1):
                gram_new = float(G_new[0, 0])
            else:
                G_new = G_new + 1e-8 * np.eye(G_new.shape[0])
                gram_new = float(np.linalg.det(G_new))

            if gram_new < 1e-8:
                gram_new = 1e-8

            if gram_old < 1e-8:
                gram_old = 1e-8

            T_new = 1.0 / np.sqrt(gram_new)
            T_old = 1.0 / np.sqrt(gram_old)

            vol_old = np.sqrt(abs(T_old))
            vol_new = np.sqrt(abs(T_new))

            IQII_hat = 1.0 - vol_new / vol_old

            return IQII_hat, gram_new
