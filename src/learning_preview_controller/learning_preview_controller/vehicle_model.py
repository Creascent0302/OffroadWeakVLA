import numpy as np


class VehicleModelMixin:
    def CalTrackWheelAu(self):
            """
            Compute nominal A and B from Au_est.
            """

            ku = self.m * (1.0 - self.Au_est) / (2.0 * self.T)
            wkomega = ku * self.wheel_radius

            self.A = np.array([
                [self.Au_est, 0.0],
                [0.0, 1.0 + self.T * (-(self.b ** 2) * ku) / (2.0 * self.Iz)]
            ], dtype=float)

            self.B = np.array([
                [self.T * wkomega / self.m, self.T * wkomega / self.m],
                [
                    -self.T * (self.b * wkomega / (2.0 * self.Iz)),
                     self.T * (self.b * wkomega / (2.0 * self.Iz))
                ]
            ], dtype=float)

    def CalTrackWheelAr(self):
            """
            Update yaw-rate diagonal term.
            """

            self.A[1, 1] = self.Ar_est
