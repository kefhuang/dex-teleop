# Deoxys Cartesian Velocity Interpolation Continuity Design

## Problem

The rate-limited Deoxys Cartesian velocity backend still aborts with:

```text
cartesian_motion_generator_joint_acceleration_discontinuity
```

libfranka defines this error as discontinuous joint acceleration after the
Cartesian motion generator performs inverse kinematics. It is not a direct
check of the requested Cartesian acceleration.

The current cosine interpolator uses `TRAJ_RATE=500` while the Franka control
callback runs at 1 kHz. It therefore holds each interpolated velocity for two
FCI cycles and applies the next two-millisecond change in one millisecond. For
the largest command transition immediately before the latest failure
(`0.02503 m/s` over a 15 ms cosine transition), the current implementation
produces approximately:

```text
Cartesian acceleration: 0, 1.082, 0, 3.059, 0, 4.508, 0, 5.176, ... m/s^2
Cartesian jerk peak: 5176 m/s^3
```

libfranka's Cartesian rate limiter caps this at about `4500 m/s^3`, but that
does not constrain the joints after IK. Using the post-failure joint state and
the robot's own libfranka model, that command direction maps to approximately
`7901 rad/s^3` on joint 7, above libfranka's listed `5000 rad/s^3` joint jerk
limit. The measured Jacobian is full rank with condition number `9.83`, so the
failure is not caused by a singular configuration.

Evaluating the same cosine transition every one-millisecond callback produces
a smooth acceleration sequence with a Cartesian jerk peak near `535 m/s^3`;
the same model maps it to approximately `940 rad/s^3` on the most affected
joint.

## Scope

This change affects only
`CosineCartesianVelocityTrajInterpolator`. It preserves:

- the 20 Hz Python policy and its speed/acceleration limits;
- the configured 15 ms transition duration (`time_fraction=0.3`);
- the newly enabled libfranka rate limiter;
- all OSC, joint-space, pose, and linear Cartesian interpolators.

## Design

### Thread ownership

The policy thread will no longer mutate active interpolation state. Its
`Reset` call will publish a pending start sample, goal, transition duration,
and monotonically increasing sequence number under a small mutex.

The 1 kHz Franka callback is the sole owner of active trajectory state. Each
`GetNextStep` call uses `try_lock` to inspect the pending command:

- if no lock is immediately available, it continues the current trajectory;
- if a new sequence is available, it starts the next trajectory from the last
  value actually emitted, not from the previous target;
- the first sequence starts from the measured Cartesian velocity supplied by
  the callback's initial `Reset`.

The real-time callback never blocks on the policy thread.

### Sampling

`GetNextStep` evaluates the cosine profile on every invocation using the
actual accumulated Franka callback time. It removes the `last_time + 1/rate`
gate that currently creates the 500 Hz sample-and-hold pattern.

For normalized time `u` clamped to `[0, 1]`, output remains:

```text
v(u) = v_start + 0.5 * (1 - cos(pi * u)) * (v_goal - v_start)
```

The duration remains `1 / policy_rate * time_fraction`, so this phase changes
continuity without intentionally adding teleoperation latency.

### Safety layers

The existing Cartesian component safety guard remains unchanged. The
libfranka rate limiter remains enabled as a second layer after interpolation.
The existing exception diagnostics continue to preserve and log any later
Franka reflex reason.

## Testing

Add a standalone C++ test for the cosine interpolator that verifies:

1. a 15 ms transition produces a new finite output on every 1 ms sample;
2. a `0.02503 m/s` transition has discrete jerk below `600 m/s^3`;
3. a goal update during an active transition starts from the last emitted
   sample without a velocity jump;
4. repeated producer updates concurrent with consumer sampling never produce
   non-finite output.

Build and run the new test together with the existing rate-limit and exception
tests. Install `franka-interface` and verify build/install Build IDs match.

## Hardware Acceptance

Run exactly one real-time backend from the installed binary and repeat the
previous rapid direction-change sequence. The backend must remain alive and
the console/file logs must contain no new
`cartesian_motion_generator_joint_acceleration_discontinuity` during the test
interval.

If the same error persists, stop and capture the final 100 ms of 1 kHz
`O_dP_EE_c`, `O_ddP_EE_c`, `q`, `dq`, and `ddq_d` rather than further changing
interpolation parameters.

