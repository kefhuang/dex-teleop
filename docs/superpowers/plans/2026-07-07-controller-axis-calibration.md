# Controller Axis Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a guided script that uses visible robot reference motions to calibrate controller position and rotation mappings, then prints a YAML snippet for teleop configs.

**Architecture:** The script moves one configured robot arm in small positive translation/rotation directions, asks the user to mimic each movement with the controller, infers dominant signed controller axes from the captured deltas, and returns the robot after each reference motion. Pure helper functions keep the inference testable without hardware; wizard tests use fake arms and fake controller states.

**Tech Stack:** Python 3.10, NumPy, existing `dex_teleop` config/device/robot/type/transform utilities, pytest.

---

### Task 1: Axis inference helpers and YAML formatting

**Files:**
- Create/modify: `scripts/calibrate_controller_axes.py`
- Create/modify: `tests/scripts/test_calibrate_controller_axes_script.py`

- [x] **Step 1: Write failing tests** for dominant signed axis selection, too-small movement errors, position/rotation mapping inference, and YAML output.
- [x] **Step 2: Verify RED** with `pytest tests/scripts/test_calibrate_controller_axes_script.py -q`.
- [x] **Step 3: Implement helpers**: `dominant_axis_spec`, `infer_mapping`, `run_calibration_from_deltas`, and `format_mapping_yaml`.
- [x] **Step 4: Verify GREEN** with targeted tests.

### Task 2: Robot-reference calibration wizard

**Files:**
- Modify: `scripts/calibrate_controller_axes.py`
- Modify: `tests/scripts/test_calibrate_controller_axes_script.py`

- [x] **Step 1: Write failing tests** proving `run_calibration` sends robot `+axis` reference motions before controller capture and sends the opposite `-axis` return motion afterwards.
- [x] **Step 2: Implement robot reference commands** using `MotionCommand` for position and rotation axes.
- [x] **Step 3: Write and pass safety test** proving robot return motion still runs if controller capture fails.
- [x] **Step 4: Add CLI args** for `--arm`, `--dry-run`, `--real-robot`, `--position-step`, `--rotation-step`, and `--hold-time`.
- [x] **Step 5: Verify help and full suite** with `python scripts/calibrate_controller_axes.py --help` and `pytest -q`.
