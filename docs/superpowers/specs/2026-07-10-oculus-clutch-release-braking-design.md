# Oculus Clutch Release Braking Design

## Problem

The latest hardware acceptance run still ended with:

```text
cartesian_motion_generator_joint_velocity_discontinuity
```

The teleoperation JSONL log aligns the failure with the falling edge of the
right-hand grip button, not with the initial grip press and not with an
ordinary held-grip command transition. Immediately before release, iteration
1544 sent:

```text
linear velocity = [-0.0382146, 0.0103731, -0.0305294] m/s
speed norm      = 0.0500 m/s
```

At iteration 1545, `grip` changed from `true` to `false`. The policy returned a
disabled velocity command, and the app called `FrankaArmController.stop_motion`,
which sent exactly zero velocity. This bypassed the policy's configured
`max_linear_deceleration` and changed the Cartesian velocity target from the
maximum speed to zero in one 20 Hz cycle.

The Deoxys 25 ms cosine interpolation reduced the severity of ordinary target
transitions but did not make this full-speed clutch stop safe after Franka's
internal Cartesian-to-joint inverse kinematics. The backend entered its stop
trajectory and exited with a joint velocity discontinuity.

## Scope Decomposition

This change addresses only normal grip-release braking. It does not combine
the separate direction-following problem into the same implementation.

The latest JSONL also shows why a nominally single-axis hand movement can
produce a diagonal robot command: accumulated target-position error and the
`position_kp` feedback term dominate the instantaneous controller-velocity
feedforward. That behavior needs a separate single-axis diagnostic and design
decision between strict direction following and exact clutch-relative pose
tracking after safe clutch release is established.

## Alternatives

### Chosen: policy-level clutch braking

Keep ordinary held-grip behavior unchanged. On the falling grip edge, preserve
the last commanded velocity and reduce it to zero over approximately 100 ms.
This directly fixes the path that bypassed the policy limiter and does not add
latency to ordinary hand motion.

### Rejected for this change: increase Deoxys `time_fraction` globally

A longer backend transition would slow every target update, including normal
tracking, and would not express the intended clutch-stop behavior explicitly.
The current 25 ms transition remains useful for continuous 1 kHz sampling.

### Deferred: joint-space Cartesian velocity execution

Computing joint velocities explicitly and applying joint velocity,
acceleration, and jerk limits would provide the strongest workspace-wide
guarantee. It is a larger controller architecture change and is not required
to test the newly confirmed application-level stop discontinuity.

## Configuration

Add the following required Cartesian velocity setting:

```yaml
cartesian_velocity:
  clutch_release_stop_time: 0.1
```

The value is expressed in seconds and must be finite and greater than zero.
It defines the maximum-speed braking horizon. Linear and angular clutch
braking rates are derived as:

```text
linear braking rate  = max_linear_velocity  / clutch_release_stop_time
angular braking rate = max_angular_velocity / clutch_release_stop_time
```

With the current `max_linear_velocity = 0.05 m/s`, the release braking rate is
`0.5 m/s^2`. At the 20 Hz policy rate, a full-speed command is reduced in two
steps rather than changing directly to zero.

## State Machine

Extend each arm's velocity state with a braking phase and a flag recording
whether the final zero command has been sent.

### Held grip

Held-grip target tracking, filtering, speed limiting, and acceleration limiting
remain unchanged.

### Grip falling edge

If the previous linear and angular commands are already zero, reset the clutch
anchor and return a disabled command as today.

Otherwise:

1. clear the active clutch anchors so a later press cannot reuse them;
2. enter `clutch_braking` while retaining the previous commands and timestamp;
3. slew both velocity vectors toward zero using the derived braking rates and
   the elapsed policy time;
4. return an enabled velocity command so the app calls `send_velocity` instead
   of `stop_motion`.

The braking command never changes direction; it only reduces the magnitude of
the last linear and angular velocity vectors.

### Final zero

Send one enabled zero-velocity command and mark it as delivered. On the next
policy update, reset the arm state and return disabled. The app may then call
`stop_motion`, but that call repeats an already-zero target and cannot create a
new velocity discontinuity.

### Grip pressed during braking

Continue braking to zero before accepting a new clutch anchor. The next cycle
with grip held anchors the controller and current robot pose and emits zero,
matching the existing first-press behavior.

### Missing input or robot state

Missing controller data or missing robot state remains an immediate disabled
stop. The 100 ms braking behavior applies only to a normal, observed grip
release and must not delay fault handling.

## Diagnostics

During braking, report:

```text
status = clutch_braking
enabled = true
command_linear_velocity
command_angular_velocity
elapsed
clutch_release_stop_time
```

The existing JSONL logger will therefore show each braking command and allow
the stop duration to be verified without relying on feel.

## Testing

Add policy regression tests that first fail against the current immediate
disable behavior and then verify:

1. releasing from full linear speed emits an enabled, same-direction command
   with reduced magnitude;
2. subsequent released-grip updates emit exactly one enabled zero command and
   then a disabled command;
3. re-pressing grip during braking completes the stop before re-anchoring;
4. releasing while already at zero remains immediately disabled;
5. missing controller or missing arm state still disables immediately;
6. invalid `clutch_release_stop_time` values are rejected by configuration
   validation.

Run the focused policy, config, app, and Franka adapter tests, followed by the
complete repository test suite.

## Hardware Acceptance

Start exactly one installed arm backend from a reset joint posture. Accelerate
to a representative command near `0.05 m/s`, then release grip.

Acceptance requires:

1. JSONL records `clutch_braking` commands before zero;
2. the Cartesian velocity reaches zero in approximately 100 ms;
3. the backend remains alive after repeated grip releases;
4. no new Cartesian motion-generator joint velocity or acceleration
   discontinuity appears in the acceptance interval;
5. ordinary held-grip motion responsiveness is unchanged.

Only after this safety acceptance passes should the separate direction
tracking diagnostic and design be implemented.
