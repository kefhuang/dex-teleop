import numpy as np

from dex_teleop.visualization.controller_3d import _project_point


def test_project_point_puts_origin_at_canvas_center():
    projected = _project_point(
        np.zeros(3),
        azimuth_deg=-45.0,
        elevation_deg=25.0,
        extent_m=0.5,
        zoom=1.0,
        width=800,
        height=600,
    )

    assert projected == (400.0, 300.0)


def test_project_point_separates_each_positive_axis_from_origin():
    kwargs = dict(
        azimuth_deg=-45.0,
        elevation_deg=25.0,
        extent_m=1.0,
        zoom=1.0,
        width=800,
        height=600,
    )
    origin = np.asarray(_project_point(np.zeros(3), **kwargs))
    axes = [
        np.asarray(_project_point(np.eye(3)[index], **kwargs))
        for index in range(3)
    ]

    assert all(np.linalg.norm(axis - origin) > 100.0 for axis in axes)
