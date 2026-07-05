from __future__ import annotations

import math

import numpy as np
import pytest

from dex_teleop.utils.transforms import (
    apply_deadzone,
    axis_mapping,
    clamp_vector_norm,
    limit_components,
    quaternion_delta_rpy,
)


def test_axis_mapping_applies_signs_and_source_axes():
    result = axis_mapping(
        vector=np.array([1.0, 2.0, 3.0]),
        mapping={"x": "+z", "y": "-x", "z": "+y"},
        output_axes=("x", "y", "z"),
        input_axes=("x", "y", "z"),
    )

    assert np.allclose(result, np.array([3.0, -1.0, 2.0]))


def test_axis_mapping_rejects_unknown_axis():
    with pytest.raises(ValueError, match="Unknown source axis"):
        axis_mapping(
            vector=np.zeros(3),
            mapping={"x": "+bad", "y": "+x", "z": "+z"},
            output_axes=("x", "y", "z"),
            input_axes=("x", "y", "z"),
        )


def test_apply_deadzone_zeroes_small_components():
    result = apply_deadzone(np.array([0.001, -0.01, 0.2]), threshold=0.01)

    assert np.allclose(result, np.array([0.0, 0.0, 0.2]))


def test_limit_components_clamps_each_component():
    result = limit_components(np.array([0.2, -0.2, 0.03]), max_abs=0.05)

    assert np.allclose(result, np.array([0.05, -0.05, 0.03]))


def test_clamp_vector_norm_preserves_direction():
    result = clamp_vector_norm(np.array([3.0, 4.0, 0.0]), max_norm=1.0)

    assert np.allclose(result, np.array([0.6, 0.8, 0.0]))


def test_quaternion_delta_rpy_identity_is_zero():
    delta = quaternion_delta_rpy(
        anchor_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
        current_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
    )

    assert np.allclose(delta, np.zeros(3))


def test_axis_mapping_accepts_unsigned_axis_specs_and_custom_axes():
    result = axis_mapping(
        vector=np.array([10.0, 20.0]),
        mapping={"vertical": "height", "forward": "depth"},
        output_axes=("forward", "vertical"),
        input_axes=("height", "depth"),
    )

    assert np.allclose(result, np.array([20.0, 10.0]))


def test_clamp_vector_norm_keeps_zero_vector_zero():
    result = clamp_vector_norm(np.zeros(3), max_norm=1.0)

    assert np.allclose(result, np.zeros(3))


def test_quaternion_delta_rpy_reports_positive_yaw():
    half_angle = math.pi / 4.0
    delta = quaternion_delta_rpy(
        anchor_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
        current_xyzw=np.array([0.0, 0.0, math.sin(half_angle), math.cos(half_angle)]),
    )

    assert np.allclose(delta, np.array([0.0, 0.0, math.pi / 2.0]))
