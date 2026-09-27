"""
Probability calibration module using Isotonic Regression for Business Entity Resolution.
Evaluates calibrated vs uncalibrated probabilities to improve decisioning and threshold stability.
"""

from typing import Optional, Tuple
import numpy as np
from sklearn.isotonic import IsotonicRegression


class ProbabilityCalibrator:
    """
    Fits and applies Isotonic Regression for probability calibration.
    """

    def __init__(self):
        self.ir: Optional[IsotonicRegression] = None

    def fit(self, y_prob: np.ndarray, y_true: np.ndarray) -> None:
        """
        Fit isotonic regression mapping uncalibrated prob -> calibrated prob.
        y_prob: 1D array of uncalibrated probabilities in [0, 1]
        y_true: 1D array of binary ground truth labels {0, 1}
        """
        self.ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self.ir.fit(y_prob, y_true)

    def calibrate(self, y_prob: np.ndarray) -> np.ndarray:
        """
        Transform uncalibrated probabilities into calibrated probabilities.
        """
        if self.ir is None:
            return y_prob
        return self.ir.predict(y_prob)

    def evaluate_calibration(
        self, y_prob: np.ndarray, y_true: np.ndarray
    ) -> Tuple[float, float]:
        """
        Compute Brier score before and after calibration.
        Returns: (brier_before, brier_after)
        """
        brier_before = float(np.mean((y_prob - y_true) ** 2))
        calibrated = self.calibrate(y_prob)
        brier_after = float(np.mean((calibrated - y_true) ** 2))
        return brier_before, brier_after
