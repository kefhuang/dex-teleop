from __future__ import annotations

import numpy as np


class CalibrationError(RuntimeError):
    """Raised when VR-to-robot frame calibration is missing or unusable."""


class TrackingFrameCalibration:
    """One arm's rotation from the Oculus tracking frame into its robot base.

    The Oculus LOCAL_FLOOR frame uses +Y as up. The robot convention used by
    this application is +X forward, +Y left, and +Z up. A controller pointer
    heading supplies the remaining horizontal alignment.
    """

    VR_UP = np.array([0.0, 1.0, 0.0])
    ROBOT_FORWARD = np.array([1.0, 0.0, 0.0])
    ROBOT_LEFT = np.array([0.0, 1.0, 0.0])
    ROBOT_UP = np.array([0.0, 0.0, 1.0])

    def __init__(self, *, min_horizontal_norm: float = 0.2) -> None:
        threshold = float(min_horizontal_norm)
        if not np.isfinite(threshold) or not 0.0 < threshold <= 1.0:
            raise ValueError("min_horizontal_norm must be finite and in (0, 1]")
        self.min_horizontal_norm = threshold
        self._robot_from_vr: np.ndarray | None = None

    @property
    def is_calibrated(self) -> bool:
        return self._robot_from_vr is not None

    @property
    def robot_from_vr(self) -> np.ndarray:
        if self._robot_from_vr is None:
            raise CalibrationError("VR-to-robot rotation has not been calibrated")
        return self._robot_from_vr

    def calibrate(self, pointer_direction_vr: np.ndarray) -> np.ndarray:
        direction = np.asarray(pointer_direction_vr, dtype=float)
        if direction.shape != (3,) or not np.all(np.isfinite(direction)):
            raise CalibrationError("pointer direction must be a finite shape-(3,) vector")
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-12:
            raise CalibrationError("pointer direction must be non-zero")
        direction = direction / norm

        forward_vr = direction - float(direction @ self.VR_UP) * self.VR_UP
        horizontal_norm = float(np.linalg.norm(forward_vr))
        if horizontal_norm < self.min_horizontal_norm:
            raise CalibrationError(
                "controller pointer is too close to vertical for heading calibration"
            )
        forward_vr /= horizontal_norm
        left_vr = np.cross(self.VR_UP, forward_vr)
        left_vr /= np.linalg.norm(left_vr)

        vr_basis = np.column_stack((forward_vr, left_vr, self.VR_UP))
        robot_basis = np.column_stack(
            (self.ROBOT_FORWARD, self.ROBOT_LEFT, self.ROBOT_UP)
        )
        rotation = robot_basis @ vr_basis.T
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
            raise CalibrationError("calculated rotation is not orthonormal")
        if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6):
            raise CalibrationError("calculated rotation is not right-handed")
        rotation.flags.writeable = False
        self._robot_from_vr = rotation
        return rotation

    def map_vector(self, vector_vr: np.ndarray) -> np.ndarray:
        vector = np.asarray(vector_vr, dtype=float)
        if vector.shape != (3,) or not np.all(np.isfinite(vector)):
            raise ValueError("vector_vr must be a finite shape-(3,) vector")
        return self.robot_from_vr @ vector
