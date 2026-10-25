from __future__ import annotations

import numpy as np


FRANKA_HOME_JOINTS = np.array(
    [
        0.09162008114028396,
        -0.19826458111314524,
        -0.01990020486871322,
        -2.4732269941140346,
        -0.01307073642274261,
        2.30396583422025,
        0.8480939705504309,
    ],
    dtype=float,
)
FRANKA_HOME_JOINTS.flags.writeable = False
