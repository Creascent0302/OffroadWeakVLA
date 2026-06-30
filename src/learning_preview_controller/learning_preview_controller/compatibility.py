import numpy as np

try:
    import cvxpy as cp
except Exception:
    cp = None


class CompatibilityMixin:
    """Compatibility methods for the QP-MPC branch.

    MPC state:
        [lateral_error, heading_error, speed, yaw_rate]

    MPC input:
        [left_wheel_speed, right_wheel_speed]

    This file intentionally keeps only QP-MPC for the MPC branch. There is no
    analytic/LQR MPC fallback here.
    """

    def CalMPCMatrix(self, kappa_ref, g):
        U0 = max(float(self.u_r), 0.1)

        g = np.asarray(g, dtype=float).reshape(-1)
        if g.size != 2:
            raise ValueError("g must be length-2: [f_U, f_R]")

        A_mpc = np.array([
            [1.0, self.T * U0,        0.0,               0.0],
            [0.0, 1.0,              -self.T * kappa_ref, self.T],
            [0.0, 0.0,               self.A[0, 0],       self.A[0, 1]],
            [0.0, 0.0,               self.A[1, 0],       self.A[1, 1]],
        ], dtype=float)

        B_mpc = np.array([
            [0.0,          0.0],
            [0.0,          0.0],
            [self.B[0, 0], self.B[0, 1]],
            [self.B[1, 0], self.B[1, 1]],
        ], dtype=float)

        d_mpc = np.array([
            [0.0],
            [0.0],
            [g[0]],
            [g[1]],
        ], dtype=float)

        return A_mpc, B_mpc, d_mpc

    def VehicleQPControl(
        self,
        A_mpc,
        B_mpc,
        d_mpc,
        R_r,
        u_r,
        state,
        last_w_l,
        last_w_r,
    ):
        """Solve QP-MPC with cvxpy/OSQP.

        This is used only when config has:
            controller:
              type: mpc
        """

        if cp is None:
            print("QP-MPC unavailable: cvxpy is not installed")
            U_value = np.tile(
                np.array([float(last_w_l), float(last_w_r)], dtype=float),
                self.Nc,
            )
            return float(last_w_l), float(last_w_r), U_value, "CVXPY_NOT_AVAILABLE"

        Nx = self.Nx
        Nu = self.Nu
        Np = self.Np
        Nc = self.Nc

        x0 = np.asarray(state, dtype=float).reshape(Nx, 1)
        u_last = np.array([last_w_l, last_w_r], dtype=float).reshape(Nu, 1)

        x_ref = np.array([
            [0.0],
            [0.0],
            [float(u_r)],
            [float(R_r)],
        ], dtype=float)

        X_ref = np.tile(x_ref, (Np, 1))

        Qbar = np.asarray(
            getattr(self, "mpc_Q_big", self.Q_big),
            dtype=float,
        )
        Rbar = np.asarray(
            getattr(self, "mpc_R_big", self.R_big),
            dtype=float,
        )

        A_mpc = np.asarray(A_mpc, dtype=float)
        B_mpc = np.asarray(B_mpc, dtype=float)
        d_mpc = np.asarray(d_mpc, dtype=float).reshape(Nx, 1)

        # Prediction matrices:
        #   X_pred = Abar*x0 + Bbar*U_seq + G
        Abar = np.zeros((Np * Nx, Nx))
        Bbar = np.zeros((Np * Nx, Nc * Nu))
        G = np.zeros((Np * Nx, 1))

        for j in range(1, Np + 1):
            row = slice((j - 1) * Nx, j * Nx)
            Abar[row, :] = np.linalg.matrix_power(A_mpc, j)

            for k in range(1, Nc + 1):
                if k <= j:
                    col = slice((k - 1) * Nu, k * Nu)
                    Bbar[row, col] = (
                        np.linalg.matrix_power(A_mpc, j - k) @ B_mpc
                    )

            g_j = np.zeros((Nx, 1))
            for s in range(j):
                g_j += np.linalg.matrix_power(A_mpc, s) @ d_mpc
            G[row, :] = g_j

        # Difference matrix for input increment penalty/constraint.
        D1 = np.eye(Nc)
        if Nc > 1:
            D1[1:, :-1] -= np.eye(Nc - 1)

        D = np.kron(D1, np.eye(Nu))

        LastVec = np.zeros((Nc * Nu, 1))
        LastVec[:Nu, :] = u_last

        dW_max = np.asarray(self.Delta_InputMax, dtype=float).reshape(-1)
        if dW_max.size == 1:
            dW_max = np.array([dW_max[0], dW_max[0]], dtype=float)
        if dW_max.size != Nu:
            raise ValueError("Delta_InputMax must be scalar or length Nu")

        DeltaWmax = np.tile(dW_max, Nc).reshape(-1, 1)

        A_cons = np.vstack([D, -D])
        B_cons = np.vstack([
            DeltaWmax + LastVec,
            DeltaWmax - LastVec,
        ])

        lb = np.asarray(self.lb_temp, dtype=float).reshape(-1)
        ub = np.asarray(self.ub_temp, dtype=float).reshape(-1)

        if lb.size == 1:
            lb = np.full(Nc * Nu, lb[0])
        elif lb.size == Nu:
            lb = np.tile(lb, Nc)
        else:
            raise ValueError("lb_temp must be scalar or length Nu")

        if ub.size == 1:
            ub = np.full(Nc * Nu, ub[0])
        elif ub.size == Nu:
            ub = np.tile(ub, Nc)
        else:
            raise ValueError("ub_temp must be scalar or length Nu")

        lb = lb.reshape(-1)
        ub = ub.reshape(-1)

        E = Abar @ x0 + G - X_ref

        H = 2.0 * (Bbar.T @ Qbar @ Bbar + D.T @ Rbar @ D)
        f = 2.0 * (Bbar.T @ Qbar @ E - D.T @ Rbar @ LastVec)

        H = (H + H.T) / 2.0
        H += float(getattr(self, "mpc_qp_regularization", 1e-5)) * np.eye(H.shape[0])
        f = f.reshape(-1)

        U_seq = cp.Variable(Nc * Nu)

        objective = cp.Minimize(
            0.5 * cp.quad_form(U_seq, cp.psd_wrap(H)) + f @ U_seq
        )

        constraints = [
            A_cons @ U_seq <= B_cons.reshape(-1),
            U_seq >= lb,
            U_seq <= ub,
        ]

        problem = cp.Problem(objective, constraints)

        w_l = float(last_w_l)
        w_r = float(last_w_r)
        U_value = np.tile(np.array([w_l, w_r], dtype=float), Nc)

        try:
            ustart = np.tile(u_last.reshape(-1), Nc)
            ustart = np.minimum(np.maximum(ustart, lb), ub)
            U_seq.value = ustart

            solver_name = str(getattr(self, "mpc_solver", "OSQP")).upper()
            solver = getattr(cp, solver_name, cp.OSQP)

            problem.solve(
                solver=solver,
                warm_start=bool(getattr(self, "mpc_warm_start", False)),
                verbose=bool(getattr(self, "mpc_verbose", False)),
                eps_abs=float(getattr(self, "mpc_eps_abs", 1e-6)),
                eps_rel=float(getattr(self, "mpc_eps_rel", 1e-6)),
                max_iter=int(getattr(self, "mpc_max_iter", 10000)),
                polish=bool(getattr(self, "mpc_polish", True)),
                adaptive_rho=bool(getattr(self, "mpc_adaptive_rho", False)),
            )

            if (
                problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]
                and U_seq.value is not None
            ):
                U_value = np.asarray(U_seq.value, dtype=float).reshape(-1)
                w_l = float(U_value[0])
                w_r = float(U_value[1])
            else:
                print("QP-MPC solve failed, status:", problem.status)
                w_l = float(last_w_l)
                w_r = float(last_w_r)

        except Exception as exc:
            print("QP-MPC solver error:", exc)
            w_l = float(last_w_l)
            w_r = float(last_w_r)

        self.checkU = float(w_l)
        self.checkR = float(w_r)

        return w_l, w_r, U_value, str(problem.status)
