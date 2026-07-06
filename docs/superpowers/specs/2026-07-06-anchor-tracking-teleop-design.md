# Anchor Tracking Teleop Design

## Goal

Replace the current controller-frame incremental motion policy with anchor-based target tracking for Oculus-to-Franka arm motion.

## Current Behavior

While grip is held, `BimanualClutchPolicy` computes arm command delta from adjacent controller samples:

```text
command_delta = current_controller_pose - previous_controller_pose
```

This is open loop with respect to the robot: after each controller sample, the policy advances `previous_pose` even if the robot did not physically follow the command. Small deltas can be filtered by deadzones or limits, and robot tracking error is not recovered.

## New Behavior

On the rising edge of grip for each arm:

```text
controller_anchor_pose = current_controller_pose
robot_anchor_pose = current_robot_pose
```

While grip remains held:

```text
controller_total_delta = current_controller_pose - controller_anchor_pose
desired_robot_pose = robot_anchor_pose + mapped(controller_total_delta)
tracking_error = desired_robot_pose - current_robot_pose
command_delta = clamp(tracking_error, max_translation_per_cycle / max_rotation_per_cycle)
```

The policy therefore maintains an absolute desired robot target relative to the grip press, but still sends bounded delta commands to Deoxys OSC.

## Deoxys Control Mode

Do not switch Deoxys to `is_delta=false` yet. Continue using `OSC_POSE` with `is_delta=true` because that path is tested and already moves the robot. The absolute target exists in the teleop policy; the robot adapter executes each step as a small delta based on target error.

## Idle Behavior

When grip is not held or a controller is missing:

- Clear that arm's anchors.
- Emit disabled motion.
- The app sends `NO_CONTROL` keepalive through `FrankaArmController.stop_motion()` when configured to keep the backend alive.
- Do not send `OSC_POSE` zero while idle, because Deoxys OSC can apply nullspace posture torque even with zero Cartesian delta.

## Required Code Changes

1. Add real Franka EEF pose reads in `FrankaArmController.get_state()` using `interface.last_eef_pose`.
2. Extend the teleop app to collect arm states each cycle and pass them to the policy.
3. Replace `BimanualClutchPolicy`'s adjacent-frame controller delta with anchor-based target tracking.
4. Keep gripper behavior unchanged.
5. Keep Deoxys execution as bounded delta OSC commands.

## Testing

Add tests for:

- Grip press captures controller and robot anchors and sends no initial motion.
- Holding grip computes command from desired robot target minus current robot pose.
- If the robot lags behind, repeated updates continue commanding correction even if the controller stops moving.
- When the robot reaches the target, command is disabled or zero-equivalent and app sends `NO_CONTROL` rather than OSC zero.
- Grip release clears anchors.
