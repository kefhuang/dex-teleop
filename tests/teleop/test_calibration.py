import numpy as np
import pytest

from dex_teleop.teleop import CalibrationError, TrackingFrameCalibration


def test_pointer_heading_and_gravity_determine_proper_frame_rotation():
    calibration = TrackingFrameCalibration()

    rotation = calibration.calibrate(np.array([0.0, 0.0, 1.0]))

    np.testing.assert_allclose(rotation @ [0, 0, 1], [1, 0, 0])
    np.testing.assert_allclose(rotation @ [0, 1, 0], [0, 0, 1])
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
    assert np.linalg.det(rotation) == pytest.approx(1.0)


def test_calibration_uses_only_horizontal_pointer_heading():
    flat = TrackingFrameCalibration().calibrate(np.array([0.0, 0.0, 1.0]))
    tilted = TrackingFrameCalibration().calibrate(np.array([0.0, 0.5, 1.0]))

    np.testing.assert_allclose(tilted, flat, atol=1e-12)


def test_calibration_rejects_nearly_vertical_pointer():
    calibration = TrackingFrameCalibration()

    with pytest.raises(CalibrationError, match="too close to vertical"):
        calibration.calibrate(np.array([0.01, 1.0, 0.01]))

    assert calibration.is_calibrated is False


def test_mapping_requires_calibration():
    calibration = TrackingFrameCalibration()

    with pytest.raises(CalibrationError, match="has not been calibrated"):
        calibration.map_vector(np.ones(3))
