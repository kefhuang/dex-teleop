# Deoxys Cartesian Velocity Rate Limiting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable libfranka's rate limiter for Deoxys Cartesian velocity control so teleoperation no longer triggers `cartesian_motion_generator_joint_acceleration_discontinuity`.

**Architecture:** A focused template helper owns the exact `franka::Robot::control` invocation and makes its arguments testable with a fake robot. Only the Cartesian velocity branch delegates to it; all other controllers, the Python policy, and the interpolator remain unchanged.

**Tech Stack:** C++17, libfranka 0.10, CMake, Deoxys standalone C++ tests, Franka hardware verification.

## Global Constraints

- Change only Deoxys `CARTESIAN_VELOCITY` behavior.
- Preserve `franka::ControllerMode::kJointImpedance` and the default cutoff frequency.
- Set libfranka `limit_rate=true`.
- Preserve unrelated dirty-worktree changes.

---

### Task 1: Preserve Completed Exception Diagnostics

**Files:**
- Modify: `dependency/deoxys_control/deoxys/franka-interface/include/utils/control_exception_utils.h`
- Modify: `dependency/deoxys_control/deoxys/franka-interface/src/franka_control_node.cpp`
- Modify: `dependency/deoxys_control/deoxys/franka-interface/src/test/test_control_exception_utils.cpp`

**Interfaces:**
- Consumes: the already-tested `RunWithShutdownOnFrankaException(operation_name, callable, running, termination)`.
- Produces: committed operation-specific `robot.readOnce` and `robot.control` diagnostics.

- [ ] **Step 1: Re-run the focused test**

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate dex-teleop
export PYTHONNOUSERSITE=1
cmake --build dependency/deoxys_control/deoxys/build --target control-exception-utils-test -j2
dependency/deoxys_control/deoxys/build/control-exception-utils-test
```

Expected: build succeeds and the executable exits 0.

- [ ] **Step 2: Review and commit only the diagnostic diff**

```bash
git -C dependency/deoxys_control diff --check
git -C dependency/deoxys_control add deoxys/franka-interface/include/utils/control_exception_utils.h deoxys/franka-interface/src/franka_control_node.cpp deoxys/franka-interface/src/test/test_control_exception_utils.cpp
git -C dependency/deoxys_control commit -m "fix: report Franka read failures"
```

Expected: a submodule commit containing only the three listed files.

### Task 2: Add a Tested Rate-Limited Control Wrapper

**Files:**
- Create: `dependency/deoxys_control/deoxys/franka-interface/include/utils/cartesian_velocity_control_utils.h`
- Create: `dependency/deoxys_control/deoxys/franka-interface/src/test/test_cartesian_velocity_control_utils.cpp`
- Modify: `dependency/deoxys_control/deoxys/CMakeLists.txt`

**Interfaces:**
- Consumes: a robot-like object exposing `control(callback, franka::ControllerMode, bool)`.
- Produces: `cartesian_velocity_control_utils::RunWithRateLimiting(Robot&, Callback&&) -> void`.

- [ ] **Step 1: Write the failing test**

Create `test_cartesian_velocity_control_utils.cpp`:

```cpp
#include <iostream>
#include <utility>
#include <franka/robot.h>
#include "utils/cartesian_velocity_control_utils.h"

struct FakeRobot {
  franka::ControllerMode mode = franka::ControllerMode::kCartesianImpedance;
  bool limit_rate = false;
  bool callback_forwarded = false;

  template <typename Callback>
  void control(Callback&& callback,
               franka::ControllerMode requested_mode,
               bool requested_limit_rate) {
    mode = requested_mode;
    limit_rate = requested_limit_rate;
    std::forward<Callback>(callback)();
  }
};

int main() {
  FakeRobot robot;
  cartesian_velocity_control_utils::RunWithRateLimiting(
      robot, [&robot]() { robot.callback_forwarded = true; });
  if (!robot.callback_forwarded ||
      robot.mode != franka::ControllerMode::kJointImpedance ||
      !robot.limit_rate) {
    std::cerr << "Cartesian velocity control options are unsafe" << std::endl;
    return 1;
  }
  return 0;
}
```

Add under `BUILD_TESTS` in `deoxys/CMakeLists.txt`:

```cmake
add_executable(cartesian-velocity-control-utils-test
  franka-interface/src/test/test_cartesian_velocity_control_utils.cpp)
target_link_libraries(cartesian-velocity-control-utils-test Franka::Franka)
```

- [ ] **Step 2: Verify RED**

```bash
cmake --build dependency/deoxys_control/deoxys/build --target cartesian-velocity-control-utils-test -j2
```

Expected: compilation fails because `utils/cartesian_velocity_control_utils.h` is absent.

- [ ] **Step 3: Implement the minimal helper**

Create `cartesian_velocity_control_utils.h`:

```cpp
#pragma once
#include <utility>
#include <franka/robot.h>

namespace cartesian_velocity_control_utils {
template <typename Robot, typename Callback>
void RunWithRateLimiting(Robot& robot, Callback&& callback) {
  robot.control(std::forward<Callback>(callback),
                franka::ControllerMode::kJointImpedance,
                true);
}
}  // namespace cartesian_velocity_control_utils
```

- [ ] **Step 4: Verify GREEN and commit**

```bash
cmake --build dependency/deoxys_control/deoxys/build --target cartesian-velocity-control-utils-test -j2
dependency/deoxys_control/deoxys/build/cartesian-velocity-control-utils-test
git -C dependency/deoxys_control add deoxys/CMakeLists.txt deoxys/franka-interface/include/utils/cartesian_velocity_control_utils.h deoxys/franka-interface/src/test/test_cartesian_velocity_control_utils.cpp
git -C dependency/deoxys_control commit -m "test: require Cartesian velocity rate limiting"
```

Expected: test exits 0 and the helper/test commit contains only the three listed files.

### Task 3: Integrate and Verify

**Files:**
- Modify: `dependency/deoxys_control/deoxys/franka-interface/src/franka_control_node.cpp`
- Modify: `dependency/deoxys_control` (root submodule pointer)
- Modify: `docs/superpowers/plans/2026-07-10-deoxys-cartesian-velocity-rate-limiting.md`

**Interfaces:**
- Consumes: `cartesian_velocity_control_utils::RunWithRateLimiting(Robot&, Callback&&)`.
- Produces: the rate-limited Deoxys Cartesian velocity branch.

- [ ] **Step 1: Integrate only the Cartesian velocity branch**

Add the helper include, then replace that branch's one-argument `robot.control` call with:

```cpp
cartesian_velocity_control_utils::RunWithRateLimiting(
    robot,
    control_callbacks::CreateCartesianVelocitiesCallback(
        global_handler, state_publisher, model, current_state_info,
        goal_state_info, policy_rate, traj_rate));
```

- [ ] **Step 2: Build and run focused verification**

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate dex-teleop
export PYTHONNOUSERSITE=1
cmake --build dependency/deoxys_control/deoxys/build --target control-exception-utils-test cartesian-velocity-control-utils-test franka-interface -j2
dependency/deoxys_control/deoxys/build/control-exception-utils-test
dependency/deoxys_control/deoxys/build/cartesian-velocity-control-utils-test
```

Expected: all targets build and both tests exit 0; only the pre-existing unused `state_sub` warning is allowed.

- [ ] **Step 3: Commit the integration**

```bash
git -C dependency/deoxys_control add deoxys/franka-interface/src/franka_control_node.cpp
git -C dependency/deoxys_control commit -m "fix: rate-limit Cartesian velocity control"
```

- [ ] **Step 4: Install and compare Build IDs**

```bash
cmake --install dependency/deoxys_control/deoxys/build
readelf -n dependency/deoxys_control/deoxys/build/franka-interface | rg -A1 "Build ID"
readelf -n dependency/deoxys_control/deoxys/bin/franka-interface | rg -A1 "Build ID"
```

Expected: build and installed binary Build IDs match.

- [ ] **Step 5: Run the hardware acceptance test**

Launch from `dependency/deoxys_control/deoxys`:

```bash
export LD_LIBRARY_PATH="$PWD/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
chrt -f 80 bin/franka-interface config/franka-three.yml 2>&1 | tee logs/franka-three-rate-limit-20260710.log
```

Repeat the prior grip/motion reproduction. Expected: backend remains alive and the log gets no new `cartesian_motion_generator_joint_acceleration_discontinuity`. If it recurs, stop and design the interpolator phase separately.

- [ ] **Step 6: Run final checks and commit the root pointer**

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate dex-teleop
export PYTHONNOUSERSITE=1
pytest -q
git -C dependency/deoxys_control status --short
git diff --check -- dependency/deoxys_control docs/superpowers/plans/2026-07-10-deoxys-cartesian-velocity-rate-limiting.md
git add dependency/deoxys_control docs/superpowers/plans/2026-07-10-deoxys-cartesian-velocity-rate-limiting.md
git commit -m "fix: rate-limit Franka Cartesian velocity"
```

Expected: Python tests pass, submodule worktree is clean, and the root commit contains only the submodule pointer and this plan.

