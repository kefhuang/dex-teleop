# Deoxys Cartesian Velocity Transition Duration Design

## Problem

After the Cartesian velocity interpolator was changed to emit a continuous
sample on every 1 kHz Franka callback, the original
`cartesian_motion_generator_joint_acceleration_discontinuity` no longer
occurred. A forward/back teleoperation motion instead stopped with:

```text
cartesian_motion_generator_joint_velocity_discontinuity
```

The frozen 100 ms diagnostic window shows that the producer command, the
1 kHz interpolator output, and libfranka's rate-limited Cartesian command are
all continuous. The first limit crossing occurs at 5.211 seconds:

```text
raw Cartesian velocity:  [-0.0145, -0.0259, -0.0212] m/s
Cartesian acceleration:  [-1.204, -1.520, -1.429] m/s^2
joint 4 desired velocity: -0.09640 rad/s
joint 4 acceleration:     -10.00869 rad/s^2
```

The local FR3 libfranka defines the joint acceleration limit as
`10.0 - 0.001 = 9.999 rad/s^2`. The desired joint velocity therefore changes
by more than the permitted amount in one millisecond. The error is named a
joint *velocity* discontinuity because the firmware checks the per-cycle
change in desired joint velocity; the value that exceeds the derivative
limit is joint acceleration.

The measured Jacobian remains full rank, and `elbow_c` is constant throughout
the diagnostic window. This rules out a singularity and an elbow-branch switch
for this failure. The direction asymmetry is produced by the Cartesian-to-joint
mapping: the forward/back command excites joint 4 more strongly than the
left/right command at this posture.

## Cause

The Python policy sends a new velocity target at 20 Hz, once every 50 ms. The
Deoxys Cartesian velocity controller currently sets:

```text
time_fraction = 0.3
transition duration = (1 / 20 Hz) * 0.3 = 15 ms
```

`time_fraction` does not scale the final velocity. It controls how much of one
policy period the cosine interpolator uses to move from the last emitted
velocity to the newest target.

The 15 ms cosine is continuous, but it is still too steep in the failing
direction. For a fixed velocity change, cosine peak acceleration is inversely
proportional to transition duration. Extending the transition to 25 ms scales
the measured transition acceleration by approximately:

```text
15 ms / 25 ms = 0.6
```

The observed joint 4 acceleration at the first crossing is therefore expected
to fall from about `10.01 rad/s^2` to about `6.01 rad/s^2`, leaving substantial
margin below the `9.999 rad/s^2` limit.

## Scope

Change the default Cartesian velocity controller configuration from:

```yaml
time_fraction: 0.3
```

to:

```yaml
time_fraction: 0.5
```

This produces a 25 ms transition with the current 20 Hz teleoperation policy.
It preserves:

- the requested steady-state Cartesian velocity;
- the 20 Hz Python policy rate;
- the continuous 1 kHz cosine evaluation;
- libfranka Cartesian rate limiting;
- all OSC, Cartesian pose, and joint-space controller configurations.

Do not add a new joint-space IK controller in this change. That is a larger
architecture option if a joint-aware limiter is later required across a wider
workspace, but it is not needed to test the confirmed 15 ms limit crossing.

## Testing

Add a regression test that loads the shipped Cartesian velocity controller
configuration and requires `time_fraction == 0.5`. The test must fail against
the current `0.3` configuration before the production change.

Retain and run the existing C++ interpolator tests to confirm that the 1 kHz
continuity, finite outputs, mid-transition command adoption, and concurrency
properties are unchanged. Rebuild and install `franka-interface`, and confirm
the build and installed binaries have matching Build IDs.

## Hardware Acceptance

Run one installed arm backend and repeat the same forward/back motion that
reproduced the failure. Acceptance requires:

1. the arm backend remains alive after repeated forward/back reversals;
2. no new `cartesian_motion_generator_joint_velocity_discontinuity` appears;
3. no regression to `cartesian_motion_generator_joint_acceleration_discontinuity`;
4. left/right motion remains responsive;
5. a diagnostic sample, if captured, keeps the maximum absolute desired joint
   acceleration below `9.999 rad/s^2`, with a target margin of at least
   `1.0 rad/s^2`.

If the same error persists, do not further increase `time_fraction` blindly.
Capture another frozen 1 kHz window and evaluate a joint-space motion generator
with explicit joint velocity, acceleration, and jerk limiting.
