import numpy as np
import pandas as pd
from typing import Optional, Tuple


class KalmanFilter:
    """
    General multivariate Kalman filter.

    State:
        x_t = F x_{t-1} + w_t
        w_t ~ N(0, Q)

    Observation:
        y_t = H x_t + v_t
        v_t ~ N(0, R)
    """

    def __init__(
        self,
        F: np.ndarray,
        H: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
        x0: Optional[np.ndarray] = None,
        P0: Optional[np.ndarray] = None,
    ) -> None:

        self.F = F
        self.H = H
        self.Q = Q
        self.R = R

        dim_x = F.shape[0]

        self.x = x0 if x0 is not None else np.zeros((dim_x, 1))
        self.P = P0 if P0 is not None else np.eye(dim_x)

    def step(self, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Execute one Kalman filter predict + update step.
        """

        # ---- Predict ----
        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + self.Q

        # ---- Update ----
        y = y.reshape(-1, 1)
        S = self.H @ P_pred @ self.H.T + self.R
        K = P_pred @ self.H.T @ np.linalg.inv(S)

        innovation = y - (self.H @ x_pred)

        self.x = x_pred + K @ innovation
        self.P = (np.eye(self.P.shape[0]) - K @ self.H) @ P_pred

        return self.x.copy(), self.P.copy()

    def filter_series(
        self,
        observations: np.ndarray,
        return_covariances: bool = False,
    ) -> Tuple[pd.DataFrame, Optional[np.ndarray]]:

        states, covs = [], []

        for obs in observations:
            x, P = self.step(obs)
            states.append(x.flatten())

            if return_covariances:
                covs.append(P.copy())

        df = pd.DataFrame(states)

        return df, (np.array(covs) if return_covariances else None)
