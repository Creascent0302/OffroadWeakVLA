import numpy as np


class UtilsMixin:
    def wrap_to_pi(self, angle):
            return (angle + np.pi) % (2.0 * np.pi) - np.pi

    def _expand_bound_vec(self, value, n):
            arr = np.asarray(value, dtype=float).reshape(-1)

            if arr.size == 1:
                return np.full((n, 1), arr[0], dtype=float)

            if arr.size == n:
                return arr.reshape(n, 1)

            raise ValueError(f"Bound size must be 1 or {n}, got {arr.size}")
