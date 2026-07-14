import numpy as np

from .config import ConfigMixin
from .preview_lqr import PreviewLQRMixin
from .learning import LearningMixin
from .vehicle_model import VehicleModelMixin
from .path import PathMixin
from .utils import UtilsMixin
from .compatibility import CompatibilityMixin


class Learning_preview_controller(
    ConfigMixin,
    PreviewLQRMixin,
    LearningMixin,
    VehicleModelMixin,
    PathMixin,
    CompatibilityMixin,
    UtilsMixin,
):
    def __init__(self, config_path=None):
            cfg = self.load_config(config_path)

            basic = cfg["basic"]
            dim = cfg["dimension"]
            path = cfg["path"]
            cons = cfg["constraint"]
            veh = cfg["vehicle"]
            learn = cfg["learning"]
            disturb = cfg["disturbance"]
            weight = cfg["weight"]
            debug = cfg.get("debug", {})
            controller_cfg = cfg.get("controller", {}) or {}

            # ===== Controller selection =====
            controller_type = controller_cfg.get("type", None)
            if controller_type is None:
                # Backward compatibility with the old flag.
                controller_type = (
                    "preview"
                    if bool(learn.get("using_preview", True))
                    else "mpc"
                )

            self.controller_type = str(controller_type).strip().lower()

            if self.controller_type in ("preview", "preview_lqr", "lqr"):
                self.controller_type = "preview"
            elif self.controller_type in ("mpc", "qp_mpc", "qp-mpc"):
                self.controller_type = "mpc"
            else:
                raise ValueError(
                    "Unsupported controller.type: "
                    f"{self.controller_type}. Use 'preview' or 'mpc'."
                )

            mpc_cfg = cfg.get("mpc", {}) or {}

            # ===== Dimension parameters =====
            self.NumOutputs = int(dim["NumOutputs"])
            self.NumInputs = int(dim["NumInputs"])
            self.NumDiscStates = int(dim["NumDiscStates"])

            self.Nx = int(dim["Nx"])
            self.Nu = int(dim["Nu"])
            self.Np = int(dim["Np"])
            self.Nc = int(dim["Nc"])

            if self.controller_type == "mpc":
                self.Np = int(mpc_cfg.get("Np", self.Np))
                self.Nc = int(mpc_cfg.get("Nc", self.Nc))

            self.step_count = 0
            self.g_warmup_steps = 10   # 前10个控制周期不用 g，可以自己改

            # ===== Lateral slope-bias / integral compensation =====
            # Used by PreviewLQRMixin.CalLateralBiasIntegralCompensation().
            # Positive lat_bias_yaw_rate means a small north correction in the
            # current local path coordinate convention.
            lat_cfg = cfg.get("lateral_integral", {}) or {}
            self.use_lat_integral = bool(lat_cfg.get("enabled", True))
            self.lat_bias_yaw_rate = float(lat_cfg.get("bias_yaw_rate", 0.008))
            self.lat_bias_wheel_limit = float(lat_cfg.get("bias_wheel_limit", 0.08))
            self.lat_int_initial_value = float(lat_cfg.get("initial_integral", -0.20))
            self.lat_int_ki = float(lat_cfg.get("ki", 0.03))
            self.lat_int_limit = float(lat_cfg.get("integral_limit", 1.2))
            self.lat_int_wheel_limit = float(lat_cfg.get("wheel_correction_limit", 0.18))
            self.lat_total_wheel_limit = float(lat_cfg.get("total_wheel_correction_limit", 0.22))
            self.lat_int_deadband = float(lat_cfg.get("error_deadband", 0.005))
            self.lat_int_kappa_enable = float(lat_cfg.get("curvature_limit", 0.08))
            self.lat_int_leak = float(lat_cfg.get("leak", 0.9997))
            self.lat_int_min_speed = float(lat_cfg.get("min_speed", 0.3))
            self.lat_int_stop_reset_speed = float(lat_cfg.get("stop_reset_speed", 0.12))

            if hasattr(self, "ResetLateralBiasIntegral"):
                self.ResetLateralBiasIntegral()

            if self.Nu != 2:
                raise ValueError("This controller assumes Nu = 2: [W_l, W_r].")

            # ===== Basic parameters =====
            self.T = float(basic["T"])
            self.ts = float(basic["ts"])

            # ===== Reference speed =====
            reference = cfg.get("reference", {}) or {}
            self.u_r = float(
                path.get(
                    "v_ref",
                    reference.get("u_r", 0.5),
                )
            )   # m/s

            # ===== Constraints =====
            self.Delta_InputMax = float(cons["Delta_InputMax"])
            self.ub_temp = float(cons["ub_temp"])
            self.lb_temp = float(cons["lb_temp"])

            # ===== Vehicle parameters =====
            self.L = float(veh["L"])
            self.Rmax = float(veh["Rmax"])
            self.m = float(veh["m"])
            self.b = float(veh["b"])
            self.Iz = float(veh["Iz"])
            self.wheel_radius = float(veh["wheel_radius"])

            # ===== Learning parameters =====
            self.useDisturbance = bool(learn["useDisturbance"])
            self.useMatrixLearning = bool(learn["useMatrixLearning"])
            self.UsingYawRate = bool(learn["UsingYawRate"])
            self.UsingABlearning = bool(learn["UsingABlearning"])
            self.UsingFulllearning = bool(learn["UsingFulllearning"])
            self.usingTrigger = bool(learn["usingTrigger"])
            self.using_preview = bool(learn["using_preview"])
            print("usingTrigger=,", self.usingTrigger)

            self.Au_nom = float(learn["Au_nom"])
            self.Ar_nom = float(learn["Ar_nom"])
            self.Bu_nom = np.asarray(learn["Bu_nom"], dtype=float).reshape(1, 2)
            self.Br_nom = np.asarray(learn["Br_nom"], dtype=float).reshape(1, 2)

            self.Au_est = self.Au_nom
            self.Ar_est = self.Ar_nom
            self.Bu_est = self.Bu_nom.copy()
            self.Br_est = self.Br_nom.copy()

            self.Nd = int(learn["Nd"])
            self.Nm = int(learn["Nm"])
            self.epsilon_u_l = float(learn["epsilon_u_l"])
            self.epsilon_r_l = float(learn["epsilon_r_l"])
            self.epsilon_uAB_l = float(learn["epsilon_uAB_l"])
            self.epsilon_rAB_l = float(learn["epsilon_rAB_l"])
            self.epsilon_AB_l = float(learn["epsilon_AB_l"])

            self.speed_threshold = float(learn["speed_threshold"])
            self.yaw_rate_threshold = float(learn["yaw_rate_threshold"])

            # ===== Disturbance learning parameters =====
            self.L_n = float(disturb["L_n"])
            self.e = float(disturb["e"])
            self.sample_size = int(disturb["sample_size"])

            # ===== Learned U-R model =====
            self.Ad_ur = np.eye(self.Nm)
            self.Ad_nom = np.eye(self.Nm)
            self.Ad_nom[0, 0] = self.Au_nom

            self.A = np.zeros((2, 2), dtype=float)
            self.B = np.zeros((2, 2), dtype=float)
            self.CalTrackWheelAu()

            self.Au_nom_hist = []
            self.Ar_nom_hist = []
            self.Au_hist = []
            self.Ar_hist = []

            self.Xminus_u = np.zeros((1, self.Nm))
            self.Xplus_u = np.zeros((1, self.Nm))
            self.Xminus_r = np.zeros((1, self.Nm))
            self.Xplus_r = np.zeros((1, self.Nm))
            self.Xplus_uAB = np.zeros((1, self.Nm))
            self.Xminus_uAB = np.zeros((3, self.Nm))
            self.Xplus_rAB = np.zeros((1, self.Nm))
            self.Xminus_rAB = np.zeros((3, self.Nm))
            self.Xplus_AB = np.zeros((2, self.Nm))
            self.Xminus_AB = np.zeros((4, self.Nm))

            self.out = np.zeros(20, dtype=float)

            self.gram_u = 1e-6
            self.gram_r = 1e-6
            self.gram_uAB = 1e-6
            self.gram_rAB = 1e-6
            self.gram_AB = 1e-6

            # ===== Disturbance learning samples =====
            self.SampleSetState = np.zeros((2, self.sample_size))
            self.SampleSetGy = np.zeros((2, self.sample_size))

            # ===== Path =====
            self.path_cfg = cfg["path"]

            path_type = str(
                self.path_cfg.get("type", "straight_circle")
            ).strip().lower()

            if path_type in ("straight_circle", "circle"):
                self.ref_path = self.GenerateStraightCircleRef()
            elif path_type in ("sine", "sin"):
                self.ref_path = self.GenerateSineRef()
            elif path_type in (
                "double_lane_change",
                "double_lane",
                "dlc",
            ):
                self.ref_path = self.GenerateDoubleLaneChangeRef()
            else:
                raise ValueError(
                    f"Unsupported path.type: {path_type}. "
                    "Use straight_circle, sine, or double_lane_change."
                )

            # Make every generated path start from local (0, 0, 0).
            self.ref_path = self.NormalizeRefPathStart(
                self.ref_path
            )

            self.ID_last = 0
            self.lateral_err = 0.0

            # ===== Preview-LQR weights =====
            # Preview LQR uses 3 states:
            #   [e_y, e_psi, e_psi_dot]
            #
            # MPC still uses self.Nx = 4.
            Q_diag = np.asarray(weight["Q_diag"], dtype=float)
            R_diag = np.asarray(weight["R_diag"], dtype=float)

            self.preview_Nx = 3

            self.Q = np.diag(Q_diag)
            self.R = np.diag(R_diag)

            if self.Q.shape != (self.preview_Nx, self.preview_Nx):
                raise ValueError(
                    "For preview LQR, Q_diag must have length 3 for state "
                    "[e_y, e_psi, e_psi_dot]."
                )

            if self.R.shape != (2, 2):
                raise ValueError("R_diag must have length 2 for input [W_l, W_r].")

            self.preview_Q_big = np.zeros(
                (self.preview_Nx * self.Np, self.preview_Nx * self.Np)
            )

            for i in range(self.Np):
                row = slice(i * self.preview_Nx, (i + 1) * self.preview_Nx)
                self.preview_Q_big[row, row] = self.Q

            self.R_big = np.zeros((self.Nu * self.Nc, self.Nu * self.Nc))
            for i in range(self.Nc):
                row = slice(i * self.Nu, (i + 1) * self.Nu)
                self.R_big[row, row] = self.R

            # Compatibility placeholder.
            # Real MPC Q_big is self.mpc_Q_big below.
            self.Q_big = np.zeros((self.Nx * self.Np, self.Nx * self.Np))

            # ===== QP-MPC weights and solver options =====
            # MPC state is [lateral_error, heading_error, speed, yaw_rate].
            mpc_weight = mpc_cfg.get("weight", {}) or {}
            mpc_Q_diag = np.asarray(
                mpc_cfg.get(
                    "Q_diag",
                    mpc_weight.get(
                        "Q_diag",
                        [45400.0, 97800.0, 100.0, 1000.0],
                    ),
                ),
                dtype=float,
            )
            mpc_R_diag = np.asarray(
                mpc_cfg.get(
                    "R_diag",
                    mpc_weight.get("R_diag", R_diag),
                ),
                dtype=float,
            )

            if mpc_Q_diag.size != self.Nx:
                raise ValueError(
                    "mpc.Q_diag must have length 4 for MPC state "
                    "[lateral_error, heading_error, speed, yaw_rate]."
                )

            if mpc_R_diag.size != self.Nu:
                raise ValueError(
                    "mpc.R_diag must have length 2 for input [W_l, W_r]."
                )

            self.mpc_Q = np.diag(mpc_Q_diag)
            self.mpc_R = np.diag(mpc_R_diag)

            self.mpc_Q_big = np.zeros((self.Nx * self.Np, self.Nx * self.Np))
            for i in range(self.Np):
                row = slice(i * self.Nx, (i + 1) * self.Nx)
                self.mpc_Q_big[row, row] = self.mpc_Q

            self.mpc_R_big = np.zeros((self.Nu * self.Nc, self.Nu * self.Nc))
            for i in range(self.Nc):
                row = slice(i * self.Nu, (i + 1) * self.Nu)
                self.mpc_R_big[row, row] = self.mpc_R

            self.mpc_solver = str(mpc_cfg.get("solver", "OSQP")).upper()
            self.mpc_warm_start = bool(mpc_cfg.get("warm_start", False))
            self.mpc_verbose = bool(mpc_cfg.get("verbose", False))
            self.mpc_eps_abs = float(mpc_cfg.get("eps_abs", 1e-6))
            self.mpc_eps_rel = float(mpc_cfg.get("eps_rel", 1e-6))
            self.mpc_max_iter = int(mpc_cfg.get("max_iter", 10000))
            self.mpc_polish = bool(mpc_cfg.get("polish", True))
            self.mpc_adaptive_rho = bool(mpc_cfg.get("adaptive_rho", False))
            self.mpc_qp_regularization = float(
                mpc_cfg.get("qp_regularization", 1e-5)
            )

            print("controller_type=", self.controller_type)

            self.checkU = 0.0
            self.checkR = 0.0

            # Optional debug behavior.
            # mode = "full": compensate [f_U, f_R] through B_UR pinv.
            # mode = "yaw": only compensate f_R.
            self.g_comp_mode = str(debug.get("g_comp_mode", "full")).lower()
            self.debug_lqr = bool(debug.get("debug_lqr", False))
            self.use_df_R_dr_in_lqr_A = bool(debug.get("use_df_R_dr_in_lqr_A", False))
            self.df_R_dr_clip = float(debug.get("df_R_dr_clip", 0.0))

    def output(self, StateIn):
            x = StateIn[0]
            y = StateIn[1]
            psi = StateIn[2]
            r = StateIn[3]
            u = StateIn[4]
            last_r = StateIn[5]
            last_u = StateIn[6]
            last_w_l = StateIn[7]
            last_w_r = StateIn[8]

            x_last = np.array([[last_u], [last_r]], dtype=float)
            u_last_vec = np.array([[last_w_l], [last_w_r]], dtype=float)
            x_now = np.array([[u], [r]], dtype=float)

            # ===== Disturbance observation in learned U-R model =====
            Obserdata = x_now - (self.A @ x_last + self.B @ u_last_vec)

            self.SampleSetState = np.delete(self.SampleSetState, 0, axis=1)
            self.SampleSetState = np.column_stack((self.SampleSetState, x_now))

            self.SampleSetGy = np.delete(self.SampleSetGy, 0, axis=1)
            self.SampleSetGy = np.column_stack((self.SampleSetGy, Obserdata))

            if self.useDisturbance:
                f_U = self.fhatpre_scalar(
                    u,
                    self.L_n,
                    self.SampleSetState[0, :],
                    self.SampleSetGy[0, :],
                    self.e
                )

                f_R = self.fhatpre_scalar(
                    r,
                    self.L_n,
                    self.SampleSetState[1, :],
                    self.SampleSetGy[1, :],
                    self.e
                )

                g = np.array([[f_U], [f_R]], dtype=float)
            else:
                g = np.zeros((2, 1), dtype=float)

            # ===== Matrix learning =====
            if self.useMatrixLearning:
                isHighSpeed = float(u) > self.speed_threshold

                if isHighSpeed:
                    self.Matrix_update_learning(
                        u=u,
                        r=r,
                        u_last_vec=u_last_vec,
                        g=g,
                        last_u=last_u,
                        last_r=last_r
                    )

            # ===== Nearest path point =====
            _, ID = self.FindNearestPoint(
                self.ref_path,
                x,
                y,
                psi,
                self.ID_last
            )

            lateral_err, psi_err, kappa_ref = self.CalErr(
                x,
                y,
                psi,
                ID,
                self.ref_path
            )

            self.ID_last = ID
            self.lateral_err = lateral_err

            X_r = self.ref_path[ID, 0]
            Y_r = self.ref_path[ID, 1]
            Psi_r = self.ref_path[ID, 2]

            # ==================================================================
            # 3-state Preview-LQR error state:
            #
            #   x_p = [e_y;
            #          e_psi;
            #          e_psi_dot]
            #
            # Using:
            #   e_psi_dot = r - U_r * kappa
            #
            # The old 4-state version included e_y_dot ≈ U_r * e_psi.
            # That created a redundant state and made Riccati fragile.
            # ==================================================================
            epsi_dot = r - self.u_r * kappa_ref

            Paper_state = np.array([
                [lateral_err],
                [psi_err],
                [epsi_dot]
            ], dtype=float)
            # print("Paper_state=",Paper_state)
            kappa_preview = self.GetPreviewCurvature(ID, self.Np)

            # t0 = time.perf_counter()

            w_l, w_r, U_seq, qp_status = self.VehiclePreviewLQRControl_WithGComp(
                paper_state=Paper_state,
                kappa_preview=kappa_preview,
                g=g,
                last_w_l=last_w_l,
                last_w_r=last_w_r,
                r=r,
                u=u
            )

            # solve_time_ms = (time.perf_counter() - t0) * 1000.0
            # print(f"Preview LQR with g compensation solve time = {solve_time_ms:.3f} ms")

            w_l = float(w_l)
            w_r = float(w_r) 
            self.out[0] = g[0, 0]
            self.out[1] = g[1, 0]
            self.out[2] = Obserdata[0, 0]
            self.out[3] = Obserdata[1, 0]
            # Save/plot the real-time matrix entries that are actually used
            # by the controller.  Do not use the legacy *_est helper fields
            # here because, depending on the selected learning mode, some of
            # them can remain nominal while self.A/self.B have already changed.
            self.out[4] = self.A[0, 0]  # A00
            self.out[5] = self.A[1, 1]  # A11
            self.out[6] = self.B[0, 0]  # B00
            self.out[7] = self.B[0, 1]  # B01
            self.out[8] = self.B[1, 0]  # B10
            self.out[9] = self.B[1, 1]  # B11
            self.out[10] = self.A[0, 1]  # A01
            self.out[11] = self.A[1, 0]  # A10
            self.out[12] = w_l
            self.out[13] = w_r
            self.out[14] = X_r
            self.out[15] = Y_r
            self.out[16] = Psi_r
            self.out[17] = self.lateral_err
            self.out[18] = self.checkU
            self.out[19] = self.checkR

            return self.out


