---
name: deoxys-franka-control
description: Use when debugging or choosing Deoxys Franka control modes in this dex-teleop repo, including JOINT_POSITION, JOINT_IMPEDANCE, CARTESIAN_VELOCITY, OSC_POSE, OSC_POSITION, reset/home behavior, per-joint tests, Cartesian motion tests, controller gains, or symptoms like small OSC deltas not moving, impedance steady-state error, drift, or weird teleop control.
---

# Deoxys Franka Control

Use this skill to debug Franka control in `dex-teleop` method-by-method. Do not use old broad benchmark scripts as evidence for root cause; test one controller at a time, move small, and read back robot state after every action.

## Environment and safety

Always run from the repo root with the existing conda env:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate dex-teleop
export PYTHONNOUSERSITE=1
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
```

Use `.3` with `franka-three.yml` unless the user explicitly chooses another arm. Keep moves small first, verify the area is safe, and finish by returning to home. Prefer `has_gripper=False` and `automatic_gripper_reset=False` for diagnostics.

Start/check the backend in tmux:

```bash
cd dependency/deoxys_control/deoxys
tmux new-session -d -s franka-three -c "$PWD" "bash auto_scripts/auto_arm.sh config/franka-three.yml"
tmux split-window -h -t franka-three -c "$PWD" "bash auto_scripts/auto_gripper.sh config/franka-three.yml"
tmux capture-pane -pt franka-three:0.0 -S -80
```

Read state before controlling:

```bash
python scripts/test_deoxys_control.py --configs franka-three.yml --duration 3 --interval 1 --no-gripper
```

Default home joints:

```python
HOME = [0.09162008114028396, -0.19826458111314524, -0.01990020486871322,
        -2.4732269941140346, -0.01307073642274261, 2.30396583422025,
        0.8480939705504309]
```

## Debugging discipline

1. Test exactly one controller family at a time.
2. Move one axis or joint at a time.
3. After every command, read `last_q` or `last_eef_pose` and compare against the requested target.
4. Reset/home between independent trials so errors do not accumulate.
5. Separate controller-switch warmup/preprocess from timed motion.
6. Record controller config (`is_delta`, gains, `action_scale`, interpolator) with results.
7. If a controller looks bad, test one hypothesis at a time: command semantics, scale, gain, duration, or target size.

## Controller quick reference

| Controller | What input means | Observed behavior on `.3` | Use for |
|---|---|---|---|
| `JOINT_POSITION` | Absolute 7-joint target + gripper slot | Accurate for per-joint `±0.05 rad`, errors about `0.0005-0.001 rad` | Home/reset, precise joint trajectories, validating hardware/control link |
| `JOINT_IMPEDANCE` | Absolute joint target with spring-damper gains | Works but default gains are soft; steady-state error on J4/J5/J7; higher `Kp/Kd` improves it | Softer joint-space motion after tuning |
| `CARTESIAN_VELOCITY` | Base-frame Cartesian velocity, `is_delta=false` | Clean directions; `0.005 m/s` for `2s` gave about `1.1-1.18x` expected distance with low drift | Teleop-style continuous small Cartesian motion |
| `OSC_POSE is_delta=true` | Delta from current pose in C++ | Small `5mm/1cm` deltas mostly fail; larger z deltas work; x/y under-track and drift | Avoid for precision small-delta teleop unless redesigned/tuned |
| `OSC_POSE is_delta=false` | Absolute base-frame pose target | Much better; 5-10cm targets track well, small 2cm targets under-track | Absolute Cartesian target tracking with maintained target pose |
| `OSC_POSITION` | Intended position-only OSC | C++ `is_delta=false` branch appears to still add xyz to current; observed weird behavior | Avoid until code/config is inspected and fixed |

## JOINT_POSITION test pattern

Use this first to prove the robot and ZMQ/control path are healthy.

- Start from home.
- For each joint, command `HOME[j] + 0.05`, read `last_q`, then command home.
- Repeat with `HOME[j] - 0.05`.
- Expected: target error around `<= 0.001 rad`; other joints should barely move.

Use `scripts/reset_franka_joints.py --config franka-three.yml --timeout 12` for final cleanup, but when testing a controller's own home behavior, home with that same controller.

## JOINT_IMPEDANCE test pattern

Default config:

```yaml
joint_kp: [100, 100, 100, 100, 75, 150, 50]
joint_kd: [20, 20, 20, 20, 7.5, 15, 5]
```

Important: if evaluating impedance behavior, use `JOINT_IMPEDANCE` for both target moves and home moves. Using `JOINT_POSITION` to home isolates trials but hides impedance home steady-state error.

Use 3 seconds per target; 10 seconds is usually unnecessary because errors plateau by about 3 seconds. Test J4/J5/J7 first. Default findings:

- J5 is softest; errors can be about `0.016-0.024 rad`.
- J4 has about `0.008-0.014 rad` error.
- J7 has smaller joint error but can inherit max error from J5 after impedance-only home.
- Larger motion amplitudes reduce relative error but not necessarily absolute error.

Tuned gains that improved results in-memory:

```yaml
joint_kp: [120, 120, 120, 150, 150, 150, 100]
joint_kd: [24, 24, 24, 30, 15, 15, 10]
```

With these, 3-second `±0.05 rad` tests improved J4/J5/J7 errors substantially. If editing YAML, document the environment change and retest per joint.

## CARTESIAN_VELOCITY test pattern

Warm up with zero velocity after switching into this controller, then time the actual command.

Use small safe commands first:

```text
velocity = 0.005 m/s
duration = 2.0 s
expected displacement = 0.010 m
```

For each axis: send `+axis`, read `last_eef_pose`, stop with zero velocity, then send `-axis`. Expected on `.3`: directions are clean, drift is sub-millimeter, measured distance is about `1.1-1.18x` command. For teleop, start with a command scale around `0.85` and tune by feel/data.

## OSC_POSE test pattern

Do not assume delta mode is suitable for fine motion.

### Delta mode (`is_delta=true`)

C++ computes:

```text
goal_pos = current_pos + [x, y, z]
```

Python multiplies xyz by `action_scale.translation` before sending. Default `action_scale.translation = 0.05`, so pass `desired_delta / 0.05` from Python if you want SI-meter deltas.

Observed behavior: `5mm/1cm` deltas are often too small and get swallowed by impedance/friction/nullspace/interpolation. At `5cm`, z tracks well but x/y still under-track with z drift.

### Absolute mode (`is_delta=false`)

C++ computes:

```text
goal_pos = [x, y, z]
goal_quat = axis_angle(ax, ay, az)
```

When testing absolute mode, preserve the current end-effector orientation: convert `last_eef_pose[:3,:3]` to axis-angle and send that with the absolute xyz target. Do not send `[0,0,0]` orientation unless identity orientation is intended.

Absolute mode findings on `.3`:

- 2cm targets under-track, roughly `0.46-0.84x` depending on axis/direction.
- 5cm targets track much better, roughly `0.76-0.95x`.
- 10cm targets track best, roughly `0.88-0.97x` for safe tested directions.

For fine teleop with OSC, maintain an accumulated absolute target pose and continuously send `OSC_POSE is_delta=false`; do not send tiny stateless deltas each frame.

## Common mistakes

- Using broad benchmark scripts to compare controllers before each controller is understood.
- Counting Deoxys controller-switch `preprocess()` time as motion time.
- Sending one OSC delta and expecting a complete step move.
- Sending OSC absolute mode with orientation `[0,0,0]`, unintentionally commanding identity orientation.
- Using `JOINT_POSITION` home while claiming to have tested impedance home behavior.
- Treating impedance steady-state error as communication failure; it is often low stiffness/no integral action.
- Stopping OSC with repeated zero OSC actions and assuming it is equivalent to idle. Prefer `NO_CONTROL` for teleop idle/keepalive when available.
