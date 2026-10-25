# Remove Cartesian Velocity Failure Telemetry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove temporary Cartesian velocity failure telemetry without changing Deoxys motion generation, safety limiting, or exact libfranka exception reporting.

**Architecture:** Delete the standalone ring-buffer unit and every ownership/record/dump integration point. Retain the existing control exception helper, rate-limited Cartesian callback, continuous cosine interpolator, transition configuration, and their focused tests.

**Tech Stack:** C++11, CMake, libfranka 0.10, spdlog, pytest.

## Global Constraints

- Do not change returned Cartesian velocity commands.
- Preserve `robot.control failed: <exact libfranka reason>` logging.
- Preserve rate limiting, 1 kHz interpolation, and `time_fraction: 0.5`.
- Do not delete existing on-disk logs.
- Only modify the Deoxys telemetry files and integration lines named below.

---

### Task 1: Establish the telemetry-removal acceptance check

**Files:**
- Inspect: `dependency/deoxys_control/deoxys/franka-interface/include/utils/cartesian_velocity_diagnostics.h`
- Inspect: all Deoxys source references returned by `rg`.

- [ ] **Step 1: Run focused baseline tests**

Run the existing diagnostics test plus exception, rate-limit, interpolation,
and configuration tests. Expected: all exit `0` before cleanup.

- [ ] **Step 2: Verify the removal check is RED**

Run:

```bash
if rg -n "cartesian_velocity_diagnostics|CartesianVelocityDiagnostics|Writing .*Cartesian velocity diagnostic|cartesian_velocity_diagnostic" \
  dependency/deoxys_control/deoxys \
  --glob '!logs/**' --glob '!build/**'; then
  exit 1
fi
```

Expected: exit `1` because the temporary telemetry source and integrations
still exist.

### Task 2: Remove telemetry and verify focused behavior

**Files:**
- Delete: `dependency/deoxys_control/deoxys/franka-interface/include/utils/cartesian_velocity_diagnostics.h`
- Delete: `dependency/deoxys_control/deoxys/franka-interface/src/test/test_cartesian_velocity_diagnostics.cpp`
- Modify: `dependency/deoxys_control/deoxys/franka-interface/include/utils/shared_memory.h`
- Modify: `dependency/deoxys_control/deoxys/franka-interface/include/utils/control_callbacks/cartesian_velocity_calllback.h`
- Modify: `dependency/deoxys_control/deoxys/franka-interface/src/franka_control_node.cpp`
- Modify: `dependency/deoxys_control/deoxys/CMakeLists.txt`

- [ ] **Step 1: Delete ownership and callback recording**

Remove the telemetry include and buffer member from `SharedMemory`, and remove
the `Record` call from the 1 kHz Cartesian velocity callback without changing
the surrounding safety guard or returned velocity.

- [ ] **Step 2: Delete failure dumping**

Remove the buffer `Clear`, `Snapshot`, and 100-frame debug loop. Keep the
unconditional `logger->error` and `logger->flush`. Remove
`<spdlog/fmt/ranges.h>` because its only use is the deleted dump.

- [ ] **Step 3: Delete the standalone unit and build target**

Delete the telemetry header and test source with `apply_patch`, and remove only
the corresponding CMake target and link lines.

- [ ] **Step 4: Verify the source check is GREEN**

Repeat Task 1 Step 2. Expected: exit `0` and no output.

- [ ] **Step 5: Build and run remaining focused tests**

Configure the existing Release build and build/run:

```text
control-exception-utils-test
cartesian-velocity-control-utils-test
cosine-cartesian-velocity-interpolator-test
cartesian-velocity-controller-config-test
```

Expected: all exit `0`.

- [ ] **Step 6: Commit the Deoxys cleanup**

Stage only the two deletions and four modified Deoxys files, inspect the staged
diff, and commit:

```text
refactor: remove Cartesian velocity telemetry
```

### Task 3: Rebuild, install, and run complete verification

- [ ] **Step 1: Build and install Deoxys**

Run `make build_franka=1 build_deoxys=1 -j2` in the Deoxys source directory.
Expected: build and install succeed.

- [ ] **Step 2: Verify installed artifact identity**

Compare `readelf -n` Build IDs for `build/franka-interface` and
`bin/franka-interface`. Expected: identical non-empty Build IDs.

- [ ] **Step 3: Run root tests**

Activate the repository conda environment with `PYTHONNOUSERSITE=1` and run
`pytest -q`. Expected: all collected tests pass.

- [ ] **Step 4: Report cleanup scope**

Report the Deoxys commit, exact focused/full test results, Build ID, and that
existing logs were left untouched. Do not commit the dirty root submodule
pointer together with unrelated root work.
