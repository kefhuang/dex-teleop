# Test Oculus Reader Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a small diagnostic script that verifies whether the Oculus Reader dependency can import, connect to ADB/Oculus, and receive controller transforms/buttons.

**Architecture:** Create one standalone script under `scripts/` that adds the vendored `~/Robots/quest/oculus_reader` package to `sys.path`, checks `adb devices`, starts `OculusReader`, samples data for a bounded duration, prints summaries, and exits non-zero if no data arrives. The script should support USB by default and network mode via `--ip`/`--port`.

**Tech Stack:** Python 3.10, existing `~/Robots/quest/oculus_reader`, `adb`, conda env `dex-teleop`.

---

### Task 1: Diagnostic Script

**Files:**
- Create: `scripts/test_oculus_reader.py`

- [ ] **Step 1: Create executable script**

Create `scripts/test_oculus_reader.py` with argument parsing, ADB checks, import path setup, bounded sampling, and clean shutdown.

- [ ] **Step 2: Run syntax check**

Run:
```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate dex-teleop
export PYTHONNOUSERSITE=1
python -m py_compile scripts/test_oculus_reader.py
```
Expected: no output and exit code 0.

- [ ] **Step 3: Run help check**

Run:
```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate dex-teleop
export PYTHONNOUSERSITE=1
python scripts/test_oculus_reader.py --help
```
Expected: usage text listing `--ip`, `--port`, `--duration`, `--interval`, and `--skip-adb-check`.

- [ ] **Step 4: Manual hardware check**

With Oculus connected and USB debugging allowed, run:
```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate dex-teleop
export PYTHONNOUSERSITE=1
python scripts/test_oculus_reader.py --duration 5
```
Expected: script prints ADB devices, starts OculusReader, reports transforms/buttons if received, and exits 0 when data arrives.
