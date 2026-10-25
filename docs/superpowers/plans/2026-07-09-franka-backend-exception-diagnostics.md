# Franka Backend Exception Diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve and print the original `libfranka` exception, shut the backend down cleanly, and use one user-driven reproduction to classify the real backend failure.

**Architecture:** Add a header-only, hardware-independent wrapper that catches `franka::Exception`, records its exact message, and requests shutdown through the existing atomic flags. Route the existing `robot.control()` dispatch through that wrapper so the subscription thread remains in scope long enough for orderly stop and join. Build and run a standalone C++ test before installing and launching the diagnostic backend.

**Tech Stack:** C++11, libfranka 0.10, `std::thread`, `std::atomic_bool`, CMake, spdlog, Deoxys `franka-interface`.

## Global Constraints

- Do not change controller gains, action scaling, velocity limits, trajectory interpolation, or robot collision behavior.
- Do not operate the robot automatically as part of the test suite.
- Do not auto-restart the diagnostic backend during the reproduction; preserve the first original failure.
- Keep existing user changes outside the vendored Deoxys submodule untouched.
- Use the existing Deoxys build tree at `dependency/deoxys_control/deoxys/build`.
- Launch the hardware reproduction with `SCHED_FIFO` priority 80 and the local Deoxys `lib` directory in `LD_LIBRARY_PATH`.

---

### Task 1: Capture `franka::Exception` without unwinding past a live thread

**Files:**
- Create: `dependency/deoxys_control/deoxys/franka-interface/include/utils/control_exception_utils.h`
- Create: `dependency/deoxys_control/deoxys/franka-interface/src/test/test_control_exception_utils.cpp`
- Modify: `dependency/deoxys_control/deoxys/CMakeLists.txt:123-138`

**Interfaces:**
- Consumes: `std::atomic_bool& running`, `std::atomic_bool& termination`, and a zero-argument callable.
- Produces: `control_exception_utils::ControlCallResult` and `control_exception_utils::RunWithShutdownOnFrankaException(Callable&&, std::atomic_bool&, std::atomic_bool&)`.

- [ ] **Step 1: Write the failing standalone test**

Create `franka-interface/src/test/test_control_exception_utils.cpp` with:

```cpp
#include <atomic>
#include <iostream>
#include <string>
#include <thread>

#include <franka/exception.h>

#include "utils/control_exception_utils.h"

int main() {
  std::atomic_bool running{true};
  std::atomic_bool termination{false};

  std::thread worker([&termination]() {
    while (!termination) {
      std::this_thread::yield();
    }
  });

  const auto failure =
      control_exception_utils::RunWithShutdownOnFrankaException(
          []() { throw franka::Exception("sentinel control failure"); },
          running,
          termination);

  if (worker.joinable()) {
    worker.join();
  }

  if (failure.succeeded) {
    std::cerr << "throwing call unexpectedly succeeded" << std::endl;
    return 1;
  }
  if (failure.error_message != "sentinel control failure") {
    std::cerr << "exception message was not preserved" << std::endl;
    return 1;
  }
  if (running || !termination) {
    std::cerr << "shutdown flags were not updated" << std::endl;
    return 1;
  }

  running = true;
  termination = false;
  const auto success =
      control_exception_utils::RunWithShutdownOnFrankaException(
          []() {}, running, termination);
  if (!success.succeeded || !success.error_message.empty()) {
    std::cerr << "non-throwing call did not report success" << std::endl;
    return 1;
  }
  if (!running || termination) {
    std::cerr << "successful call changed shutdown flags" << std::endl;
    return 1;
  }

  return 0;
}
```

Add this target after the existing `cartesian-velocity` target in `CMakeLists.txt`:

```cmake
  # Test libfranka exception capture without connecting to a robot.
  add_executable(control-exception-utils-test
    franka-interface/src/test/test_control_exception_utils.cpp)
  target_link_libraries(control-exception-utils-test Franka::Franka Threads::Threads)
```

- [ ] **Step 2: Configure and verify the test fails for the missing helper**

Run:

```bash
cd /home/kefhuang/agent-workspace/dex-teleop/dependency/deoxys_control/deoxys
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$PWD" -DCMAKE_LIBRARY_OUTPUT_DIRECTORY="$PWD" -DBUILD_DEOXYS=0 -DBUILD_FRANKA=1
cmake --build build --target control-exception-utils-test -- -j2
```

Expected: compilation fails because `utils/control_exception_utils.h` does not exist. This confirms the test exercises the new interface.

- [ ] **Step 3: Implement the minimal hardware-independent helper**

Create `franka-interface/include/utils/control_exception_utils.h` with:

```cpp
#pragma once

#include <atomic>
#include <string>
#include <utility>

#include <franka/exception.h>

namespace control_exception_utils {

struct ControlCallResult {
  bool succeeded;
  std::string error_message;
};

template <typename Callable>
ControlCallResult RunWithShutdownOnFrankaException(
    Callable&& control_call,
    std::atomic_bool& running,
    std::atomic_bool& termination) {
  try {
    std::forward<Callable>(control_call)();
  } catch (const franka::Exception& exception) {
    running = false;
    termination = true;
    return {false, exception.what()};
  }
  return {true, ""};
}

}  // namespace control_exception_utils
```

- [ ] **Step 4: Build and run the standalone test**

Run:

```bash
cd /home/kefhuang/agent-workspace/dex-teleop/dependency/deoxys_control/deoxys
cmake --build build --target control-exception-utils-test -- -j2
LD_LIBRARY_PATH="$PWD/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  ./build/control-exception-utils-test
```

Expected: build exits 0, the executable exits 0, and it prints no error output.

- [ ] **Step 5: Commit the tested helper inside the Deoxys submodule**

```bash
cd /home/kefhuang/agent-workspace/dex-teleop/dependency/deoxys_control
git add deoxys/CMakeLists.txt \
  deoxys/franka-interface/include/utils/control_exception_utils.h \
  deoxys/franka-interface/src/test/test_control_exception_utils.cpp
git commit -m "test: cover Franka control exception shutdown"
```

### Task 2: Route robot control failures through orderly cleanup

**Files:**
- Modify: `dependency/deoxys_control/deoxys/franka-interface/src/franka_control_node.cpp:18-23`
- Modify: `dependency/deoxys_control/deoxys/franka-interface/src/franka_control_node.cpp:537-595`

**Interfaces:**
- Consumes: `control_exception_utils::RunWithShutdownOnFrankaException` from Task 1.
- Produces: persistent `robot.control failed: <original message>` logging, normal state-publisher stop, subscription-thread join, and exit status 1 after a captured failure.

- [ ] **Step 1: Add the exception helper include**

Add alongside the existing utility includes:

```cpp
#include "utils/control_exception_utils.h"
```

- [ ] **Step 2: Wrap the existing controller dispatch without changing callbacks**

Immediately before `while (!global_handler->termination)`, add:

```cpp
    bool control_failed = false;
```

Replace the controller-type dispatch inside `if (global_handler->running)` with this structure, retaining the existing callback construction exactly:

```cpp
        const auto control_result =
            control_exception_utils::RunWithShutdownOnFrankaException(
                [&]() {
                  if (controller_type == ControllerType::OSC_POSE ||
                      controller_type == ControllerType::OSC_POSITION ||
                      controller_type == ControllerType::OSC_YAW) {
                    robot.control(
                        control_callbacks::CreateTorqueFromCartesianSpaceCallback(
                            global_handler, state_publisher, model,
                            current_state_info, goal_state_info, policy_rate,
                            traj_rate));
                  } else if (controller_type == ControllerType::JOINT_IMPEDANCE) {
                    global_handler->logger->info("Joint impedance callback");
                    robot.control(
                        control_callbacks::CreateTorqueFromJointSpaceCallback(
                            global_handler, state_publisher, model,
                            current_state_info, goal_state_info, policy_rate,
                            traj_rate));
                  } else if (controller_type == ControllerType::JOINT_POSITION) {
                    global_handler->logger->info("Joint position callback");
                    robot.control(control_callbacks::CreateJointPositionCallback(
                        global_handler, state_publisher, model,
                        current_state_info, goal_state_info, policy_rate,
                        traj_rate));
                  } else if (controller_type ==
                             ControllerType::CARTESIAN_VELOCITY) {
                    global_handler->logger->info(
                        "Cartesian velocity callback");
                    robot.control(
                        control_callbacks::CreateCartesianVelocitiesCallback(
                            global_handler, state_publisher, model,
                            current_state_info, goal_state_info, policy_rate,
                            traj_rate));
                  }
                },
                global_handler->running,
                global_handler->termination);

        if (!control_result.succeeded) {
          global_handler->logger->error(
              "robot.control failed: {0}", control_result.error_message);
          global_handler->logger->flush();
          control_failed = true;
        }
```

Do not modify `init_state = robot.readOnce()`, callback bodies, controller configuration, or command values.

- [ ] **Step 3: Make cleanup idempotent and return failure after joining**

Replace the unconditional join and fall-through return section with:

```cpp
    state_publisher->StopPublishing();

    if (control_msg_sub.joinable()) {
      control_msg_sub.join();
    }
    if (control_failed) {
      return -1;
    }
```

Keep the existing outer `catch (franka::Exception const& e)` for startup failures that occur before `control_msg_sub` is created.

- [ ] **Step 4: Build the production target and rerun the unit test**

Run:

```bash
cd /home/kefhuang/agent-workspace/dex-teleop/dependency/deoxys_control/deoxys
cmake --build build --target control-exception-utils-test franka-interface -- -j2
LD_LIBRARY_PATH="$PWD/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  ./build/control-exception-utils-test
```

Expected: both targets build successfully, the standalone test exits 0, and no robot connection is attempted by the test.

- [ ] **Step 5: Confirm only the intended Deoxys files changed**

```bash
cd /home/kefhuang/agent-workspace/dex-teleop/dependency/deoxys_control
git status --short
git diff --check
git diff -- deoxys/franka-interface/src/franka_control_node.cpp
```

Expected: only `deoxys/franka-interface/src/franka_control_node.cpp` is uncommitted after Task 1, and `git diff --check` prints nothing.

- [ ] **Step 6: Commit the control-node integration inside the submodule**

```bash
cd /home/kefhuang/agent-workspace/dex-teleop/dependency/deoxys_control
git add deoxys/franka-interface/src/franka_control_node.cpp
git commit -m "fix: preserve Franka control exceptions"
```

### Task 3: Install and verify the diagnostic backend without moving the robot

**Files:**
- Update generated artifact: `dependency/deoxys_control/deoxys/bin/franka-interface`
- Inspect: `dependency/deoxys_control/deoxys/logs/franka-three-diagnostic.log`

**Interfaces:**
- Consumes: the rebuilt `franka-interface` executable from Task 2.
- Produces: an installed diagnostic backend that runs with FIFO priority 80 and writes the first original failure to a persistent console log.

- [ ] **Step 1: Install the rebuilt executable**

```bash
cd /home/kefhuang/agent-workspace/dex-teleop/dependency/deoxys_control/deoxys
cmake --install build
```

Expected: `bin/franka-interface` is updated from `build/franka-interface` without changing source files.

- [ ] **Step 2: Verify dynamic libraries and the diagnostic test**

```bash
cd /home/kefhuang/agent-workspace/dex-teleop/dependency/deoxys_control/deoxys
LD_LIBRARY_PATH="$PWD/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  ldd bin/franka-interface | rg 'not found' && exit 1 || true
LD_LIBRARY_PATH="$PWD/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  ./build/control-exception-utils-test
```

Expected: `ldd` reports no missing libraries and the test exits 0.

- [ ] **Step 3: Ensure no stale backend or auto-restart loop is running**

```bash
pgrep -af 'auto_arm.sh|bin/franka-interface'
```

Expected: no matching process. If a process is listed, stop it from the terminal that launched it before continuing; do not run two robot backends simultaneously.

- [ ] **Step 4: Launch the diagnostic backend once with persistent logging**

```bash
cd /home/kefhuang/agent-workspace/dex-teleop/dependency/deoxys_control/deoxys
: > logs/franka-three-diagnostic.log
export LD_LIBRARY_PATH="$PWD/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
chrt -f 80 bin/franka-interface config/franka-three.yml \
  2>&1 | tee logs/franka-three-diagnostic.log
```

Expected before teleoperation: output includes `Deoxys starting`; the process remains running and does not command motion by itself.

- [ ] **Step 5: Verify the running process uses FIFO priority 80**

From another terminal, run:

```bash
pid=$(pgrep -o -f '^bin/franka-interface config/franka-three.yml$')
chrt -p "$pid"
```

Expected: scheduling policy `SCHED_FIFO` and priority `80`.

- [ ] **Step 6: Perform one user-driven reproduction and preserve the first failure**

Run the existing teleoperation command normally. Keep the robot workspace clear and the emergency stop accessible. Stop after the backend exits once; do not start `auto_arm.sh` during this reproduction.

Expected: `logs/franka-three-diagnostic.log` contains a line beginning with `robot.control failed:` and does not replace it solely with `terminate called without an active exception`.

### Task 4: Classify the original backend failure

**Files:**
- Inspect: `dependency/deoxys_control/deoxys/logs/franka-three-diagnostic.log`
- Inspect: newest `logs/oculus_franka_*.jsonl`

**Interfaces:**
- Consumes: the preserved original exception and the matching command/state timeline.
- Produces: an evidence-backed classification of the backend failure and the next narrowly scoped repair.

- [ ] **Step 1: Extract the original exception**

```bash
cd /home/kefhuang/agent-workspace/dex-teleop
rg -n 'robot\.control failed:|communication_constraints|motion generator|reflex|limit|violation' \
  dependency/deoxys_control/deoxys/logs/franka-three-diagnostic.log
```

Expected: at least one `robot.control failed:` line containing the original libfranka message.

- [ ] **Step 2: Identify the matching teleoperation log and failure cycle**

```bash
latest=$(find logs -maxdepth 1 -type f -name 'oculus_franka_*.jsonl' \
  -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)
printf 'latest=%s\n' "$latest"
jq -c 'select(.event == "cycle" and .arm_states.right == null) |
  {iteration, timestamp_monotonic, policy: .policy.right,
   command: .commands.velocity_motions.right}' "$latest" | head -1
```

Expected: the first missing-state cycle is printed with the last policy/command context available in the log.

- [ ] **Step 3: Classify without applying a second fix**

Use the exact original message and matching timeline:

- `communication_constraints_violation`, `UDP receive`, or timeout text means the next repair must focus on real-time scheduling, CPU affinity, and the robot network path.
- `cartesian_motion_generator_*`, velocity discontinuity, or acceleration discontinuity means the next repair must focus on Cartesian command slew and trajectory interpolation.
- joint-position, elbow, or Cartesian-position limit text means the next repair must use the logged pose plus full Franka joint/error telemetry.
- collision, reflex, force, or power-limit text means the next repair must preserve the safety stop and investigate contact/load conditions before any controller change.

Record the exact error text, last valid pose, last command, and failure-cycle index in the task handoff. Do not change controller behavior in this diagnostic plan.

- [ ] **Step 4: Commit the root repository's submodule pointer and plan**

```bash
cd /home/kefhuang/agent-workspace/dex-teleop
git add dependency/deoxys_control \
  docs/superpowers/plans/2026-07-09-franka-backend-exception-diagnostics.md
git commit -m "fix: expose Franka backend control failures"
```

Expected: the root commit records only the Deoxys submodule pointer and this implementation plan; unrelated user changes remain unstaged.

## Final Verification

Before claiming the diagnostic is complete, run:

```bash
cd /home/kefhuang/agent-workspace/dex-teleop/dependency/deoxys_control/deoxys
cmake --build build --target control-exception-utils-test franka-interface -- -j2
LD_LIBRARY_PATH="$PWD/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  ./build/control-exception-utils-test
cd /home/kefhuang/agent-workspace/dex-teleop
git diff --check HEAD^..HEAD
```

Required evidence:

- both C++ targets build successfully;
- the standalone exception test exits 0;
- the runtime log contains the original `robot.control failed:` message;
- the root-cause classification cites the matching teleoperation cycle; and
- no unrelated user files were staged or committed.
