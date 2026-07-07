# Controller Axis Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a guided script that calibrates controller position and rotation axis mappings and prints a YAML snippet for teleop configs.

**Architecture:** Keep the script safe and inspectable: it only reads controller poses and never moves robots or writes config files. Put axis inference in pure helper functions so tests cover the important behavior without an Oculus device.

**Tech Stack:** Python 3.10, NumPy, existing `dex_teleop` config/device/types/transform utilities, pytest.

---

### Task 1: Pure axis inference helpers and tests

**Files:**
- Create: `scripts/calibrate_controller_axes.py`
- Create: `tests/scripts/test_calibrate_controller_axes_script.py`

- [ ] **Step 1: Write failing helper tests**

Add tests for dominant signed axis selection, position mapping formatting, rotation mapping formatting, and too-small movement errors.

- [ ] **Step 2: Run helper tests to verify RED**

Run:

```bash
pytest tests/scripts/test_calibrate_controller_axes_script.py -q
```

Expected: import failure because `scripts/calibrate_controller_axes.py` does not exist yet.

- [ ] **Step 3: Implement minimal helpers**

Implement:

- `dominant_axis_spec(delta, axes, min_delta)`
- `infer_mapping(measurements, output_axes, input_axes, min_delta)`
- `format_mapping_yaml(position_axes, rotation_axes)`

- [ ] **Step 4: Run helper tests to verify GREEN**

Run:

```bash
pytest tests/scripts/test_calibrate_controller_axes_script.py -q
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add scripts/calibrate_controller_axes.py tests/scripts/test_calibrate_controller_axes_script.py
git commit -m "feat: add controller axis calibration helpers"
```

### Task 2: Interactive calibration wizard

**Files:**
- Modify: `scripts/calibrate_controller_axes.py`
- Modify: `tests/scripts/test_calibrate_controller_axes_script.py`

- [ ] **Step 1: Write failing tests for sampling and prompt orchestration**

Test that fake controller states produce expected mapping results when the wizard receives deterministic before/after samples.

- [ ] **Step 2: Run tests to verify RED**

Run:

```bash
pytest tests/scripts/test_calibrate_controller_axes_script.py -q
```

Expected: fail because wizard functions are not implemented.

- [ ] **Step 3: Implement wizard**

Implement CLI args and functions:

- `parse_args`
- `make_device`
- `read_controller_pose`
- `capture_delta`
- `run_calibration`
- `main`

- [ ] **Step 4: Run script smoke test with mock controller**

Run a non-interactive unit test and a short import/`--help` smoke command:

```bash
pytest tests/scripts/test_calibrate_controller_axes_script.py -q
python scripts/calibrate_controller_axes.py --help
```

Expected: tests pass and help text prints.

- [ ] **Step 5: Run full suite and commit**

```bash
pytest -q
git add scripts/calibrate_controller_axes.py tests/scripts/test_calibrate_controller_axes_script.py
git commit -m "feat: add guided controller axis calibration script"
```
