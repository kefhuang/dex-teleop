# Deoxys Cartesian Velocity Transition Duration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the shipped Deoxys Cartesian velocity transition from 15 ms to 25 ms so forward/back commands remain below the FR3 joint acceleration limit after inverse kinematics.

**Architecture:** Keep the existing 20 Hz policy, continuous 1 kHz cosine interpolator, and libfranka Cartesian rate limiter. Change only the Cartesian velocity controller's dimensionless `time_fraction` from `0.3` to `0.5`, with a compiled regression test that loads the shipped YAML and enforces the safety-critical value.

**Tech Stack:** C++11, CMake, yaml-cpp, Deoxys controller YAML, libfranka 0.10, pytest for repository regression tests.

## Global Constraints

- Preserve the requested steady-state Cartesian velocity.
- Preserve the 20 Hz Python policy rate and continuous 1 kHz interpolation.
- Do not change OSC, Cartesian pose, or joint-space controller configurations.
- Do not add a new joint-space IK controller in this change.
- Hardware acceptance requires no Cartesian motion-generator acceleration or joint-velocity discontinuity and an arm backend that remains alive.

---

### Task 1: Add the configuration regression test

**Files:**
- Create: `dependency/deoxys_control/deoxys/franka-interface/src/test/test_cartesian_velocity_controller_config.cpp`
- Modify: `dependency/deoxys_control/deoxys/CMakeLists.txt`

**Interfaces:**
- Consumes: `DEOXYS_CARTESIAN_VELOCITY_CONFIG_PATH`, supplied by CMake as an absolute source-tree path.
- Produces: executable target `cartesian-velocity-controller-config-test`, returning zero only when `traj_interpolator_cfg.time_fraction == 0.5`.

- [ ] **Step 1: Write the failing test**

Create the test with:

```cpp
#include <cmath>
#include <iostream>

#include <yaml-cpp/yaml.h>

#ifndef DEOXYS_CARTESIAN_VELOCITY_CONFIG_PATH
#error "DEOXYS_CARTESIAN_VELOCITY_CONFIG_PATH is required"
#endif

int main() {
  const YAML::Node config =
      YAML::LoadFile(DEOXYS_CARTESIAN_VELOCITY_CONFIG_PATH);
  const double time_fraction =
      config["traj_interpolator_cfg"]["time_fraction"].as<double>();
  if (std::abs(time_fraction - 0.5) > 1e-12) {
    std::cerr << "expected Cartesian velocity time_fraction 0.5, got "
              << time_fraction << std::endl;
    return 1;
  }
  return 0;
}
```

Add this target after the cosine interpolator test in `CMakeLists.txt`:

```cmake
  # Test the safety-critical Cartesian velocity transition duration.
  add_executable(cartesian-velocity-controller-config-test
    franka-interface/src/test/test_cartesian_velocity_controller_config.cpp)
  target_compile_definitions(cartesian-velocity-controller-config-test PRIVATE
    DEOXYS_CARTESIAN_VELOCITY_CONFIG_PATH="${CMAKE_CURRENT_SOURCE_DIR}/config/cartesian-velocity-controller.yml")
  target_link_libraries(cartesian-velocity-controller-config-test yaml-cpp)
```

- [ ] **Step 2: Build and run the test to verify RED**

Run:

```bash
cd dependency/deoxys_control/deoxys
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$PWD" -DCMAKE_LIBRARY_OUTPUT_DIRECTORY="$PWD" -DBUILD_DEOXYS=1 -DBUILD_FRANKA=1
cmake --build build --target cartesian-velocity-controller-config-test -j2
./build/cartesian-velocity-controller-config-test
```

Expected: the executable exits `1` and prints
`expected Cartesian velocity time_fraction 0.5, got 0.3`.

### Task 2: Extend the Cartesian velocity transition

**Files:**
- Modify: `dependency/deoxys_control/deoxys/config/cartesian-velocity-controller.yml:7`
- Test: `dependency/deoxys_control/deoxys/franka-interface/src/test/test_cartesian_velocity_controller_config.cpp`

**Interfaces:**
- Consumes: 20 Hz policy rate and `CosineCartesianVelocityTrajInterpolator::Reset` duration formula `1 / policy_rate * time_fraction`.
- Produces: `time_fraction: 0.5`, giving a 25 ms transition without changing the target velocity.

- [ ] **Step 1: Make the minimal production change**

Change:

```yaml
  time_fraction: 0.3
```

to:

```yaml
  time_fraction: 0.5
```

- [ ] **Step 2: Run focused tests to verify GREEN**

Run:

```bash
cd dependency/deoxys_control/deoxys
cmake --build build --target \
  cartesian-velocity-controller-config-test \
  cosine-cartesian-velocity-interpolator-test \
  cartesian-velocity-control-utils-test \
  cartesian-velocity-diagnostics-test \
  control-exception-utils-test -j2
./build/cartesian-velocity-controller-config-test
./build/cosine-cartesian-velocity-interpolator-test
./build/cartesian-velocity-control-utils-test
./build/cartesian-velocity-diagnostics-test
./build/control-exception-utils-test
```

Expected: all commands exit `0` with no test error output.

- [ ] **Step 3: Commit the Deoxys change**

```bash
git -C dependency/deoxys_control add \
  deoxys/CMakeLists.txt \
  deoxys/config/cartesian-velocity-controller.yml \
  deoxys/franka-interface/src/test/test_cartesian_velocity_controller_config.cpp
git -C dependency/deoxys_control commit -m "fix: lengthen Cartesian velocity transitions"
```

### Task 3: Rebuild and install the backend

**Files:**
- Verify: `dependency/deoxys_control/deoxys/build/franka-interface`
- Verify: `dependency/deoxys_control/deoxys/bin/franka-interface`

**Interfaces:**
- Consumes: committed Deoxys source and controller configuration.
- Produces: installed `bin/franka-interface` built from the same source as the verified build artifact.

- [ ] **Step 1: Build and install Franka and Python artifacts**

Run:

```bash
cd dependency/deoxys_control/deoxys
make build_franka=1 build_deoxys=1 -j2
```

Expected: build and install complete successfully.

- [ ] **Step 2: Compare Build IDs**

Run:

```bash
cd dependency/deoxys_control/deoxys
readelf -n build/franka-interface | rg "Build ID"
readelf -n bin/franka-interface | rg "Build ID"
```

Expected: both commands print the same non-empty Build ID.

- [ ] **Step 3: Run repository regression tests**

Run:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate dex-teleop
export PYTHONNOUSERSITE=1
pytest -q
```

Expected: all collected repository tests pass.

### Task 4: Perform hardware acceptance and record the result

**Files:**
- Inspect: `dependency/deoxys_control/deoxys/logs/deoxys_control_arm_program.log`
- Modify: root git submodule pointer for `dependency/deoxys_control` after acceptance.

**Interfaces:**
- Consumes: installed backend with matching Build ID and the existing Oculus teleoperation launcher.
- Produces: hardware evidence that repeated forward/back reversals remain below the FR3 limit and the backend remains alive.

- [ ] **Step 1: Ask the user to confirm the workspace is safe for motion**

Do not start an arm backend or command motion until the user confirms the robot is enabled and its workspace is clear.

- [ ] **Step 2: Start exactly one installed arm backend**

Use the existing repository launcher/backend command for `franka-three.yml`, and verify only one arm process owns the connection before teleoperation begins.

- [ ] **Step 3: Repeat the reproducing motion**

Have the user perform repeated forward/back reversals, then left/right motion, while the backend remains running.

- [ ] **Step 4: Inspect the latest error and diagnostic records**

Run:

```bash
rg -n "cartesian_motion_generator_joint_(velocity|acceleration)_discontinuity|robot.control failed" \
  dependency/deoxys_control/deoxys/logs/deoxys_control_arm_program.log
```

Expected: no matching record from the acceptance interval and the backend process remains alive.

- [ ] **Step 5: Commit only the accepted submodule pointer**

```bash
git add dependency/deoxys_control
git commit -m "fix: smooth Deoxys Cartesian velocity transitions"
```

Do not stage unrelated dirty workspace files.
