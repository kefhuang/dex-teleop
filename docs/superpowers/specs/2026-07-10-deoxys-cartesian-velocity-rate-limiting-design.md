# Deoxys Cartesian Velocity Rate Limiting Design

## Problem

During Oculus teleoperation, the Franka backend repeatedly aborts active
Cartesian velocity control with:

```text
libfranka: Move command aborted: motion aborted by reflex!
["cartesian_motion_generator_joint_acceleration_discontinuity"]
```

The Python policy limits Cartesian speed and command-to-command acceleration
at 20 Hz. The Deoxys C++ backend then sends Cartesian velocity commands in the
1 kHz Franka control loop. Its `robot.control` call uses libfranka's default
`limit_rate=false`, while the local safety guard only clips velocity values and
does not enforce acceleration or jerk continuity.

## Scope

This change affects only Deoxys `CARTESIAN_VELOCITY` control. It does not alter
OSC, joint position, joint impedance, the Python teleoperation policy, or the
Cartesian velocity interpolator.

## Design

Add a small, testable C++ helper that invokes Cartesian velocity control with:

- `franka::ControllerMode::kJointImpedance`, matching the existing default;
- `limit_rate=true`;
- libfranka's existing default cutoff frequency.

The Cartesian velocity branch in `franka_control_node.cpp` will call this
helper. A fake robot in the unit test will record the arguments and verify that
the callback is forwarded, the controller mode is unchanged, and rate limiting
is enabled.

The existing exception diagnostic path remains in place so any later Franka
abort still records the original exception and exits after joining worker
threads.

## Alternatives Considered

1. Call `franka::limitRate` directly inside the callback. This requires
   maintaining prior commanded velocity and acceleration correctly at 1 kHz,
   duplicating behavior already provided by libfranka.
2. Reduce Python velocity and acceleration limits only. This cannot guarantee
   continuity at the Franka control-loop boundary.
3. Rewrite the cosine interpolator at the same time. This would mix two
   behavioral variables and could add teleoperation latency before the minimal
   fix is evaluated.

## Verification

1. Write the helper unit test first and verify it fails because the helper is
   absent.
2. Implement the helper and integrate it into only the Cartesian velocity
   branch.
3. Build and run the new C++ test plus the existing exception utility test.
4. Build and install `franka-interface`; verify the installed binary matches
   the build by Build ID.
5. Launch one diagnostic backend with real-time priority and repeat the same
   teleoperation motion.

The hardware acceptance criterion is that the backend remains running and the
console log contains no new
`cartesian_motion_generator_joint_acceleration_discontinuity` error during the
reproduction interval. If the same reflex still occurs, stop and design the
interpolator-continuity change separately.

