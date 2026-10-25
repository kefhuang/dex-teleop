# Oculus Clutch Release Braking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the full-speed-to-zero command on normal grip release with a configuration-driven braking sequence that reaches zero in approximately 100 ms before disabling motion.

**Architecture:** Add a per-arm `clutch_braking` phase to `BimanualCartesianVelocityPolicy`. The policy retains the last velocity command after the falling grip edge, slews its magnitude toward zero at a rate derived from configured maximum velocity and `clutch_release_stop_time`, sends one enabled zero command, then clears the clutch anchor and disables motion.

**Tech Stack:** Python 3.10, NumPy, dataclasses, YAML configuration, pytest, existing JSONL teleoperation diagnostics.

## Global Constraints

- `clutch_release_stop_time` is `0.1` seconds in shipped Oculus Franka configs.
- Normal held-grip target tracking and its latency remain unchanged.
- Braking commands preserve the last velocity direction and only reduce magnitude.
- Re-pressing grip during braking must not bypass the stop or reuse an old anchor.
- Missing controller or robot state remains an immediate disabled stop.
- Do not change the direction-following or axis-mapping behavior in this plan.
- Do not stage or overwrite unrelated dirty workspace changes.

---

### Task 1: Add failing braking and configuration tests

**Files:**
- Modify: `tests/policies/test_bimanual_cartesian_velocity.py`
- Modify: `tests/core/test_config.py`

**Interfaces:**
- Consumes: `BimanualCartesianVelocityPolicy.update(DeviceState, arm_states=...)`.
- Produces: expected policy statuses `clutch_braking`, `anchored`, and `clutch_released`, plus a required `cartesian_velocity.clutch_release_stop_time` config key.

- [ ] **Step 1: Add the policy fixture setting**

Add this key to `_cfg()["cartesian_velocity"]`:

```python
"clutch_release_stop_time": 0.1,
```

- [ ] **Step 2: Add a release braking regression test**

Add a test that anchors at `t=0.0`, reaches `0.05 m/s` at `t=0.05`, releases at `t=0.10`, and asserts:

```python
assert braking.enabled is True
assert np.allclose(braking.linear_velocity, np.array([0.025, 0.0, 0.0]))
assert policy.last_diagnostics["left"]["status"] == "clutch_braking"
```

Continue released updates at `t=0.15` and `t=0.20` and assert that exactly one enabled zero command is emitted before the command becomes disabled.

- [ ] **Step 3: Add re-press and zero-speed tests**

Verify that a grip press while braking first emits the final zero and that the next held-grip cycle re-anchors at the current controller and robot poses. Separately verify that releasing immediately after the initial zero anchor stays disabled without entering braking.

- [ ] **Step 4: Add invalid setting tests**

Parametrize `0.0`, `float("nan")`, and `float("inf")` and require policy construction to raise:

```python
ValueError("clutch_release_stop_time must be finite and > 0.0")
```

Add a config-loader regression test that removes the key from a shipped config and requires:

```text
Missing required config field: cartesian_velocity.clutch_release_stop_time
```

- [ ] **Step 5: Run tests to verify RED**

Run:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate dex-teleop
export PYTHONNOUSERSITE=1
pytest -q tests/policies/test_bimanual_cartesian_velocity.py tests/core/test_config.py
```

Expected: the new tests fail because grip release still immediately returns a disabled zero command and the config loader does not require the new key.

### Task 2: Implement the clutch braking state machine

**Files:**
- Modify: `dex_teleop/policies/bimanual_cartesian_velocity.py`
- Modify: `dex_teleop/core/config.py`
- Modify: `configs/oculus_franka_bimanual.yml`
- Modify: `configs/oculus_franka_right.yml`
- Modify: embedded valid YAML strings in `tests/core/test_config.py`

**Interfaces:**
- Consumes: `clutch_release_stop_time: float` from `cartesian_velocity` config.
- Produces: enabled `CartesianVelocityCommand` braking samples and JSONL-compatible policy diagnostics.

- [ ] **Step 1: Extend configuration loading**

Add `clutch_release_stop_time` to `REQUIRED_CARTESIAN_VELOCITY`, add
`clutch_release_stop_time: 0.1` to both shipped configs and all valid embedded
test YAML, pass it through `from_config`, and reject values that are not finite
or are less than or equal to zero.

- [ ] **Step 2: Extend per-arm state**

Add these fields to `_ArmVelocityState`:

```python
braking: bool = False
brake_zero_sent: bool = False
```

Retain `previous_linear_command`, `previous_angular_command`, and
`previous_timestamp` while braking. Clear active anchors when braking begins.

- [ ] **Step 3: Add a focused braking helper**

Add a helper that computes elapsed time, clamps it by `max_dt`, and calls
`_limit_velocity_change` toward zero using:

```python
linear_rate = max_linear_velocity / clutch_release_stop_time
angular_rate = max_angular_velocity / clutch_release_stop_time
```

It must update retained commands and timestamp, return an enabled command even
when the newly computed velocity is exactly zero, and set `brake_zero_sent` on
that zero sample. Its diagnostic record must include:

```python
{
    "status": "clutch_braking",
    "enabled": True,
    "elapsed": elapsed,
    "clutch_release_stop_time": self.clutch_release_stop_time,
    "command_linear_velocity": linear_command.copy(),
    "command_angular_velocity": angular_command.copy(),
}
```

- [ ] **Step 4: Route normal grip release through braking**

After controller and robot-state presence checks:

- if braking is active and the zero sample has not been sent, continue braking regardless of the current grip value;
- if the zero sample was sent, reset state, then either return disabled for a released grip or fall through to the existing held-grip anchoring logic;
- if grip falls while a prior velocity is nonzero, clear anchors, enter braking, and emit the first reduced command;
- if grip falls while prior commands are already zero, reset immediately as before.

- [ ] **Step 5: Run focused tests to verify GREEN**

Run:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate dex-teleop
export PYTHONNOUSERSITE=1
pytest -q \
  tests/policies/test_bimanual_cartesian_velocity.py \
  tests/core/test_config.py \
  tests/apps/test_oculus_franka_app.py \
  tests/robots/test_franka_adapter.py
```

Expected: all focused tests pass.

### Task 3: Verify and commit the software change

**Files:**
- Verify all files modified in Tasks 1 and 2.

**Interfaces:**
- Consumes: completed braking state machine.
- Produces: a clean, focused commit containing only clutch braking and its tests/config.

- [ ] **Step 1: Run formatting and diff checks**

Run:

```bash
git diff --check
git diff -- \
  dex_teleop/policies/bimanual_cartesian_velocity.py \
  dex_teleop/core/config.py \
  configs/oculus_franka_bimanual.yml \
  configs/oculus_franka_right.yml \
  tests/policies/test_bimanual_cartesian_velocity.py \
  tests/core/test_config.py
```

Expected: no whitespace errors and no unrelated changes introduced by this task.

- [ ] **Step 2: Run the complete repository suite**

Run:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate dex-teleop
export PYTHONNOUSERSITE=1
pytest -q
```

Expected: all tests pass.

- [ ] **Step 3: Commit only a separable implementation scope**

Inspect each in-scope file for pre-existing uncommitted work. If the braking
hunks can be staged without including other work, stage only those hunks and
commit:

```bash
git commit -m "fix: brake Cartesian velocity on clutch release"
```

If an in-scope untracked file or overlapping hunk would pull broader existing
teleoperation work into the commit, do not commit it in this task. Leave the
verified implementation in place and report the overlap. Never stage unrelated
or inseparable user work merely to create a commit.

### Task 4: Perform hardware acceptance

**Files:**
- Inspect: newest `logs/oculus_franka_*.jsonl`
- Inspect: `dependency/deoxys_control/deoxys/logs/deoxys_control_arm_program.log`

**Interfaces:**
- Consumes: installed Deoxys backend at submodule commit `5bf1175` and the updated Python policy.
- Produces: evidence that repeated grip release no longer aborts the backend.

- [ ] **Step 1: Confirm the robot workspace is safe**

Ask the user before starting an arm backend or commanding motion.

- [ ] **Step 2: Start exactly one backend without an auto-restart loop**

Run the installed `bin/franka-interface config/franka-three.yml`, then launch
the Oculus teleoperation app with `configs/oculus_franka_right.yml`.

- [ ] **Step 3: Exercise the release path**

Reach a representative velocity near `0.05 m/s`, release grip, and repeat at
least three times. Do not use broad multi-controller benchmarking.

- [ ] **Step 4: Verify JSONL braking evidence**

The falling grip edge must be followed by `clutch_braking` records whose speed
norm monotonically decreases, including one enabled zero record before
`clutch_released` becomes disabled.

- [ ] **Step 5: Verify backend survival and error-free logs**

The same backend PID/session must remain alive and the acceptance interval must
contain no new `cartesian_motion_generator_joint_velocity_discontinuity` or
`cartesian_motion_generator_joint_acceleration_discontinuity`.

- [ ] **Step 6: Continue with the separate direction diagnostic**

After release acceptance passes, use a short single-axis capture to compare
instantaneous controller target velocity, position-error feedback, commanded
Cartesian velocity, and measured robot displacement. Do not tune mapping or
`position_kp` until that evidence is reviewed.
