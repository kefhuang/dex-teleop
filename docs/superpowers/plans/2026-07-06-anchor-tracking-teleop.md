# Anchor Tracking Teleop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace adjacent-frame controller delta teleop with anchor-based robot target tracking while continuing to execute bounded Deoxys OSC delta commands.

**Architecture:** `FrankaArmController.get_state()` will expose real EEF pose. The app will gather arm states each loop and pass them into `BimanualClutchPolicy.update()`. The policy will store grip anchors for controller and robot pose, compute an absolute desired robot target, then output a bounded delta equal to desired target minus current robot pose.

**Tech Stack:** Python 3.10, pytest, numpy, vendored Deoxys FrankaInterface, existing `dex_teleop` policy/app/robot abstractions.

---

## File Structure

- Modify `dex_teleop/robots/franka.py`: implement real EEF pose reads from Deoxys `last_eef_pose` and keep `NO_CONTROL` idle behavior.
- Modify `dex_teleop/policies/bimanual_clutch.py`: add arm-state input, anchor state, and target-error delta calculation.
- Modify `dex_teleop/apps/oculus_franka.py`: collect arm states before policy update and pass them in.
- Modify `tests/robots/test_franka_adapter.py`: cover real EEF pose conversion.
- Modify `tests/policies/test_bimanual_clutch.py`: replace incremental-delta expectations with anchor-tracking expectations.
- Modify `tests/apps/test_oculus_franka_app.py`: cover app passing arm states into policy.

---

### Task 1: Real Franka EEF State

**Files:**
- Modify: `dex_teleop/robots/franka.py`
- Test: `tests/robots/test_franka_adapter.py`

- [ ] **Step 1: Write failing tests for real EEF pose reads**

Add to `tests/robots/test_franka_adapter.py`:

```python
def test_franka_arm_get_state_reads_last_eef_pose_from_interface():
    fake = FakeFrankaInterface()
    fake.last_eef_pose = np.array(
        [
            [1.0, 0.0, 0.0, 0.4],
            [0.0, 1.0, 0.0, -0.2],
            [0.0, 0.0, 1.0, 0.3],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    arm = FrankaArmController("left", interface_cfg="unused.yml", interface=fake, controller_cfg={})
    arm.connect()

    state = arm.get_state()

    assert state.arm_id == "left"
    assert np.allclose(state.pose.position, [0.4, -0.2, 0.3])
    assert np.allclose(state.pose.orientation, [0.0, 0.0, 0.0, 1.0])


def test_franka_arm_get_state_returns_identity_when_no_eef_pose_available():
    fake = FakeFrankaInterface()
    fake.last_eef_pose = None
    arm = FrankaArmController("left", interface_cfg="unused.yml", interface=fake, controller_cfg={})
    arm.connect()

    state = arm.get_state()

    assert state.arm_id == "left"
    assert np.allclose(state.pose.position, [0.0, 0.0, 0.0])
    assert np.allclose(state.pose.orientation, [0.0, 0.0, 0.0, 1.0])
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate dex-teleop
export PYTHONNOUSERSITE=1
pytest tests/robots/test_franka_adapter.py::test_franka_arm_get_state_reads_last_eef_pose_from_interface tests/robots/test_franka_adapter.py::test_franka_arm_get_state_returns_identity_when_no_eef_pose_available -q
```

Expected: first test fails because `get_state()` returns identity instead of interface pose.

- [ ] **Step 3: Implement real EEF pose reads**

In `dex_teleop/robots/franka.py`, add a helper using the existing quaternion conversion logic or local rotation-to-quaternion function, then update `get_state()`:

```python
    def get_state(self) -> ArmState:
        if self.interface is None:
            return ArmState(arm_id=self.arm_id, pose=Pose.identity())
        eef_pose = getattr(self.interface, "last_eef_pose", None)
        if eef_pose is None:
            return ArmState(arm_id=self.arm_id, pose=Pose.identity())
        arr = np.asarray(eef_pose, dtype=float)
        if arr.shape != (4, 4):
            return ArmState(arm_id=self.arm_id, pose=Pose.identity())
        return ArmState(
            arm_id=self.arm_id,
            pose=Pose(
                position=arr[:3, 3],
                orientation=_quaternion_xyzw_from_rotation(arr[:3, :3]),
            ),
        )
```

- [ ] **Step 4: Run tests and verify pass**

Run the same pytest command. Expected: both pass.

- [ ] **Step 5: Commit**

```bash
git add dex_teleop/robots/franka.py tests/robots/test_franka_adapter.py
git commit -m "feat: read Franka end-effector state"
```

---

### Task 2: App Passes Arm State Into Policy

**Files:**
- Modify: `dex_teleop/apps/oculus_franka.py`
- Test: `tests/apps/test_oculus_franka_app.py`

- [ ] **Step 1: Write failing test**

Add to `tests/apps/test_oculus_franka_app.py`:

```python
def test_run_passes_arm_states_to_policy(monkeypatch):
    seen_arm_states = []

    class FakeDevice:
        def __init__(self) -> None:
            self.reads = 0
        def start(self) -> None:
            pass
        def read(self):
            self.reads += 1
            if self.reads > 1:
                raise KeyboardInterrupt
            return object()
        def stop(self) -> None:
            pass

    class FakeArm:
        def connect(self) -> None:
            pass
        def get_state(self):
            return "left-state"
        def send_motion(self, command: MotionCommand) -> None:
            pass
        def stop_motion(self) -> None:
            pass
        def close(self) -> None:
            pass

    class FakePolicy:
        @classmethod
        def from_config(cls, cfg):
            return cls()
        def update(self, state, arm_states=None):
            seen_arm_states.append(arm_states)
            return TeleopCommandBatch(motions={"left": MotionCommand.disabled("left")})

    cfg = {
        "app": {"dry_run": True, "rate_hz": 1000},
        "input": {"mock_controller": True},
        "arms": {"left": {}},
        "grippers": {},
        "safety": {"send_zero_when_idle": True},
    }
    monkeypatch.setattr(oculus_franka, "_make_device", lambda cfg: FakeDevice())
    monkeypatch.setattr(oculus_franka, "_make_arms", lambda cfg: {"left": FakeArm()})
    monkeypatch.setattr(oculus_franka, "_make_grippers", lambda cfg, arms: {})
    monkeypatch.setattr(oculus_franka, "BimanualClutchPolicy", FakePolicy)

    assert oculus_franka.run(cfg) == 0
    assert seen_arm_states == [{"left": "left-state"}]
```

- [ ] **Step 2: Run test and verify failure**

Run:

```bash
pytest tests/apps/test_oculus_franka_app.py::test_run_passes_arm_states_to_policy -q
```

Expected: fails because `run()` calls `policy.update(state)` without arm states.

- [ ] **Step 3: Implement arm state collection**

In `dex_teleop/apps/oculus_franka.py`, replace:

```python
batch = policy.update(state)
```

with:

```python
arm_states = {arm_id: arm.get_state() for arm_id, arm in arms.items()}
batch = policy.update(state, arm_states=arm_states)
```

- [ ] **Step 4: Run test and verify pass**

Run the same pytest command. Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add dex_teleop/apps/oculus_franka.py tests/apps/test_oculus_franka_app.py
git commit -m "feat: pass arm states to teleop policy"
```

---

### Task 3: Anchor-Based Position Tracking

**Files:**
- Modify: `dex_teleop/policies/bimanual_clutch.py`
- Test: `tests/policies/test_bimanual_clutch.py`

- [ ] **Step 1: Write failing tests**

Add helpers to `tests/policies/test_bimanual_clutch.py`:

```python
from dex_teleop.core.types import ArmState


def _arm_states(left=(0.0, 0.0, 0.0), right=(0.0, 0.0, 0.0)):
    return {
        "left": ArmState("left", Pose(position=left, orientation=(0.0, 0.0, 0.0, 1.0))),
        "right": ArmState("right", Pose(position=right, orientation=(0.0, 0.0, 0.0, 1.0))),
    }
```

Add tests:

```python
def test_first_grip_press_anchors_controller_and_robot_and_emits_enabled_zero_delta():
    policy = _policy()
    batch = policy.update(
        _state(left_pos=(1.0, 0.0, 0.0), left_grip=True),
        arm_states=_arm_states(left=(0.5, 0.0, 0.2)),
    )

    assert batch.motions["left"].enabled is True
    assert np.allclose(batch.motions["left"].delta_position, np.zeros(3))


def test_held_grip_tracks_absolute_controller_offset_against_robot_current_pose():
    policy = _policy_with(
        mapping={"position_axes": {"x": "+x", "y": "+y", "z": "+z"}},
        scales={"position": 1.0, "rotation": 1.0},
        safety={
            "max_translation_per_cycle": 0.01,
            "max_rotation_per_cycle": 0.05,
            "position_deadzone": 0.0,
            "rotation_deadzone": 0.0,
            "send_zero_when_idle": True,
        },
    )
    policy.update(
        _state(left_pos=(0.0, 0.0, 0.0), left_grip=True),
        arm_states=_arm_states(left=(0.5, 0.0, 0.2)),
    )

    batch = policy.update(
        _state(left_pos=(0.03, 0.0, 0.0), left_grip=True),
        arm_states=_arm_states(left=(0.51, 0.0, 0.2)),
    )

    assert batch.motions["left"].enabled is True
    assert np.allclose(batch.motions["left"].delta_position, [0.01, 0.0, 0.0])


def test_robot_lag_continues_to_emit_correction_when_controller_stops():
    policy = _policy_with(
        mapping={"position_axes": {"x": "+x", "y": "+y", "z": "+z"}},
        scales={"position": 1.0, "rotation": 1.0},
        safety={
            "max_translation_per_cycle": 0.01,
            "max_rotation_per_cycle": 0.05,
            "position_deadzone": 0.0,
            "rotation_deadzone": 0.0,
            "send_zero_when_idle": True,
        },
    )
    policy.update(
        _state(left_pos=(0.0, 0.0, 0.0), left_grip=True),
        arm_states=_arm_states(left=(0.5, 0.0, 0.2)),
    )
    policy.update(
        _state(left_pos=(0.03, 0.0, 0.0), left_grip=True),
        arm_states=_arm_states(left=(0.51, 0.0, 0.2)),
    )

    batch = policy.update(
        _state(left_pos=(0.03, 0.0, 0.0), left_grip=True),
        arm_states=_arm_states(left=(0.515, 0.0, 0.2)),
    )

    assert np.allclose(batch.motions["left"].delta_position, [0.01, 0.0, 0.0])
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
pytest tests/policies/test_bimanual_clutch.py::test_first_grip_press_anchors_controller_and_robot_and_emits_enabled_zero_delta tests/policies/test_bimanual_clutch.py::test_held_grip_tracks_absolute_controller_offset_against_robot_current_pose tests/policies/test_bimanual_clutch.py::test_robot_lag_continues_to_emit_correction_when_controller_stops -q
```

Expected: tests fail because `update()` does not accept `arm_states` and uses previous controller pose.

- [ ] **Step 3: Implement anchor state**

Change `_ArmClutchState` in `dex_teleop/policies/bimanual_clutch.py` to:

```python
@dataclass
class _ArmClutchState:
    active: bool = False
    controller_anchor_pose: Pose | None = None
    robot_anchor_pose: Pose | None = None
```

Change `update` signature:

```python
def update(
    self,
    state: DeviceState,
    *,
    arm_states: dict[str, ArmState] | None = None,
) -> TeleopCommandBatch:
```

Import `ArmState` from `dex_teleop.core.types`.

On first grip press, require current robot pose:

```python
robot_state = (arm_states or {}).get(arm_id)
if robot_state is None:
    motions[arm_id] = MotionCommand.disabled(arm_id)
    continue
self._states[arm_id] = _ArmClutchState(
    active=True,
    controller_anchor_pose=_snapshot_pose(controller.pose),
    robot_anchor_pose=_snapshot_pose(robot_state.pose),
)
motions[arm_id] = MotionCommand(arm_id, np.zeros(3), np.zeros(3), True)
continue
```

While active, compute:

```python
robot_state = (arm_states or {}).get(arm_id)
if robot_state is None:
    self._states[arm_id] = _ArmClutchState()
    motions[arm_id] = MotionCommand.disabled(arm_id)
    continue
controller_anchor = clutch_state.controller_anchor_pose
robot_anchor = clutch_state.robot_anchor_pose
raw_position_delta = controller.pose.position - controller_anchor.position
mapped_position = axis_mapping(...)
desired_robot_position = robot_anchor.position + mapped_position * self.position_scale
position_error = desired_robot_position - robot_state.pose.position
delta_position = limit_components(apply_deadzone(position_error, self.position_deadzone), self.max_translation_per_cycle)
```

For rotation in the first implementation, continue using controller anchor to current controller orientation to compute desired rotation error, but keep existing delta execution and limit/deadzone. If rotation proves unstable in hardware, disable rotation separately in config by setting `rotation: 0.0`.

- [ ] **Step 4: Run targeted policy tests**

Run the same pytest command. Expected: pass.

- [ ] **Step 5: Run full policy tests and adjust existing expectations**

Run:

```bash
pytest tests/policies/test_bimanual_clutch.py -q
```

Update existing tests that asserted adjacent-frame behavior so they now assert anchor-tracking behavior. Keep tests for missing controller, trigger gripper, deadzones, and grip release.

- [ ] **Step 6: Commit**

```bash
git add dex_teleop/policies/bimanual_clutch.py tests/policies/test_bimanual_clutch.py
git commit -m "feat: track arm targets from clutch anchors"
```

---

### Task 4: Integration Verification

**Files:**
- Modify: `README.md` if command docs need a note about anchor tracking.

- [ ] **Step 1: Run all tests**

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate dex-teleop
export PYTHONNOUSERSITE=1
pytest -q
```

Expected: all tests pass.

- [ ] **Step 2: Dry-run print commands with mock controller**

```bash
python -m dex_teleop.apps.oculus_franka \
  --config configs/oculus_franka_bimanual.yml \
  --mock-controller \
  --dry-run \
  --duration 0.5 \
  --rate-hz 10 \
  --print-commands
```

Expected: grip false cycles emit disabled motions; grip true cycles emit enabled commands based on target error.

- [ ] **Step 3: Dry-run print commands with real controller**

```bash
python -m dex_teleop.apps.oculus_franka \
  --config configs/oculus_franka_bimanual.yml \
  --real-controller \
  --dry-run \
  --duration 10 \
  --print-commands
```

Expected: with grip released, arms remain disabled. With grip pressed and controller moved, corresponding arm command is enabled and bounded.

- [ ] **Step 4: Commit docs if changed**

```bash
git add README.md
git commit -m "docs: describe anchor tracking teleop" || true
```

---

## Self-Review

- Spec coverage: plan covers real robot state, app state flow, anchor-based policy, idle NO_CONTROL behavior, and verification.
- Placeholder scan: no TBD/TODO placeholders remain.
- Type consistency: uses existing `ArmState`, `Pose`, `MotionCommand`, and `TeleopCommandBatch` names.
