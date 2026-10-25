# Deoxys Cartesian Velocity Interpolation Continuity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the 500 Hz sample-and-hold jerk and cross-thread active-state mutation from the Deoxys cosine Cartesian velocity interpolator.

**Architecture:** `Reset` becomes a producer-side pending-command publication. `GetNextStep` remains the sole steady-state owner of active trajectory state, adopts pending commands with non-blocking `try_lock`, starts from the last emitted sample, and evaluates the cosine profile every Franka callback.

**Tech Stack:** C++17, Eigen, CMake, libfranka 0.10, standalone C++ tests, Franka hardware verification.

## Global Constraints

- Modify only the cosine Cartesian velocity interpolation path and its test target.
- Preserve the 15 ms transition duration and Python policy.
- Keep libfranka rate limiting enabled.
- Preserve unrelated dirty-worktree changes.

---

### Task 1: Add Failing Continuity Tests

**Files:**
- Create: `dependency/deoxys_control/deoxys/franka-interface/src/test/test_cosine_cartesian_velocity_traj_interpolator.cpp`
- Modify: `dependency/deoxys_control/deoxys/CMakeLists.txt`

**Interfaces:**
- Consumes: existing `CosineCartesianVelocityTrajInterpolator::Reset` and `GetNextStep`.
- Produces: executable `cosine-cartesian-velocity-interpolator-test`.

- [ ] **Step 1: Add a test that samples a 0.02503 m/s transition every 1 ms**

Create the test with these assertions:

```cpp
#include <algorithm>
#include <atomic>
#include <cmath>
#include <iostream>
#include <thread>
#include <vector>

#include <Eigen/Dense>

#include "utils/traj_interpolators/cosine_cartesian_velocity_traj_interpolator.h"

namespace {
constexpr double kSamplePeriod = 0.001;
constexpr double kGoal = 0.025033843499841165;

void Reset(traj_utils::CosineCartesianVelocityTrajInterpolator& interpolator,
           const Eigen::Vector3d& start,
           const Eigen::Vector3d& goal,
           double time = 0.0) {
  interpolator.Reset(time, start, Eigen::Vector3d::Zero(), goal,
                     Eigen::Vector3d::Zero(), 20, 500, 0.3);
}
}  // namespace

int main() {
  traj_utils::CosineCartesianVelocityTrajInterpolator interpolator;
  Reset(interpolator, Eigen::Vector3d::Zero(),
        Eigen::Vector3d(kGoal, 0.0, 0.0));

  std::vector<double> samples;
  for (int i = 0; i <= 16; ++i) {
    Eigen::Vector3d translation;
    Eigen::Vector3d rotation;
    interpolator.GetNextStep(i * kSamplePeriod, translation, rotation);
    if (!translation.allFinite() || !rotation.allFinite()) return 1;
    samples.push_back(translation.x());
  }
  for (int i = 1; i <= 15; ++i) {
    if (!(samples[i] > samples[i - 1])) {
      std::cerr << "interpolator repeated an active 1 ms sample" << std::endl;
      return 1;
    }
  }
  double max_jerk = 0.0;
  for (size_t i = 2; i < samples.size(); ++i) {
    const double previous_acceleration =
        (samples[i - 1] - samples[i - 2]) / kSamplePeriod;
    const double acceleration =
        (samples[i] - samples[i - 1]) / kSamplePeriod;
    max_jerk = std::max(
        max_jerk,
        std::abs(acceleration - previous_acceleration) / kSamplePeriod);
  }
  if (max_jerk >= 600.0) {
    std::cerr << "jerk bound exceeded: " << max_jerk << std::endl;
    return 1;
  }

  traj_utils::CosineCartesianVelocityTrajInterpolator retargeted;
  Reset(retargeted, Eigen::Vector3d::Zero(),
        Eigen::Vector3d(kGoal, 0.0, 0.0));
  Eigen::Vector3d before;
  Eigen::Vector3d rotation;
  for (int i = 0; i <= 5; ++i) {
    retargeted.GetNextStep(i * kSamplePeriod, before, rotation);
  }
  Reset(retargeted, Eigen::Vector3d::Constant(0.5),
        Eigen::Vector3d(-0.01, 0.0, 0.0), 0.005);
  Eigen::Vector3d after;
  retargeted.GetNextStep(0.005, after, rotation);
  if (!after.isApprox(before, 1e-12)) {
    std::cerr << "retarget introduced a velocity jump" << std::endl;
    return 1;
  }

  traj_utils::CosineCartesianVelocityTrajInterpolator concurrent;
  Reset(concurrent, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());
  std::atomic_bool start{false};
  std::thread writer([&]() {
    while (!start) std::this_thread::yield();
    for (int i = 0; i < 2000; ++i) {
      const double goal = 0.02 * std::sin(i * 0.01);
      Reset(concurrent, Eigen::Vector3d::Zero(),
            Eigen::Vector3d(goal, 0.0, 0.0), i * 0.001);
    }
  });
  start = true;
  for (int i = 0; i < 4000; ++i) {
    Eigen::Vector3d translation;
    concurrent.GetNextStep(i * kSamplePeriod, translation, rotation);
    if (!translation.allFinite()) return 1;
  }
  writer.join();
  return 0;
}
```

- [ ] **Step 2: Add the CMake target**

```cmake
add_executable(cosine-cartesian-velocity-interpolator-test
  franka-interface/src/test/test_cosine_cartesian_velocity_traj_interpolator.cpp)
target_link_libraries(cosine-cartesian-velocity-interpolator-test Threads::Threads)
```

- [ ] **Step 3: Configure/build/run and verify RED**

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate dex-teleop
export PYTHONNOUSERSITE=1
cmake -S dependency/deoxys_control/deoxys -B dependency/deoxys_control/deoxys/build
cmake --build dependency/deoxys_control/deoxys/build --target cosine-cartesian-velocity-interpolator-test -j2
dependency/deoxys_control/deoxys/build/cosine-cartesian-velocity-interpolator-test
```

Expected: the executable fails because the current 500 Hz gate repeats samples and exceeds the jerk bound.

### Task 2: Make Cosine Interpolation Callback-Continuous

**Files:**
- Modify: `dependency/deoxys_control/deoxys/franka-interface/include/utils/traj_interpolators/cosine_cartesian_velocity_traj_interpolator.h`

**Interfaces:**
- Consumes: the existing virtual `Reset`/`GetNextStep` signatures.
- Produces: pending command state protected by a mutex and callback-owned active interpolation state.

- [ ] **Step 1: Replace active-state mutation in `Reset` with pending publication**

Store pending start/goal/duration/sequence under `std::mutex`. Do not modify active start time or active output from `Reset`.

- [ ] **Step 2: Adopt pending commands without blocking in `GetNextStep`**

Use `try_lock`; initialize the first trajectory from the pending measured start, and later trajectories from the last emitted output. If no pending command is available, return the last finite output.

- [ ] **Step 3: Evaluate cosine on every callback**

Remove the `last_time_ + dt_` gate and calculate the clamped normalized time and cosine profile on every `GetNextStep` call.

- [ ] **Step 4: Verify GREEN**

```bash
cmake --build dependency/deoxys_control/deoxys/build --target cosine-cartesian-velocity-interpolator-test -j2
dependency/deoxys_control/deoxys/build/cosine-cartesian-velocity-interpolator-test
```

Expected: executable exits 0.

- [ ] **Step 5: Commit the interpolator and test checkpoint in the submodule**

```bash
git -C dependency/deoxys_control add deoxys/CMakeLists.txt deoxys/franka-interface/include/utils/traj_interpolators/cosine_cartesian_velocity_traj_interpolator.h deoxys/franka-interface/src/test/test_cosine_cartesian_velocity_traj_interpolator.cpp
git -C dependency/deoxys_control commit -m "fix: make Cartesian velocity interpolation continuous"
```

### Task 3: Build, Install, and Hardware-Test

**Files:**
- Modify: `dependency/deoxys_control` (root submodule pointer)
- Modify: `docs/superpowers/plans/2026-07-10-deoxys-cartesian-velocity-rate-limiting.md`
- Modify: `docs/superpowers/plans/2026-07-10-deoxys-cartesian-velocity-interpolation-continuity.md`

**Interfaces:**
- Consumes: the continuous interpolator and existing rate-limited control wrapper.
- Produces: installed backend and hardware acceptance evidence.

- [ ] **Step 1: Build all focused targets**

```bash
cmake --build dependency/deoxys_control/deoxys/build --target cosine-cartesian-velocity-interpolator-test cartesian-velocity-control-utils-test control-exception-utils-test franka-interface -j2
dependency/deoxys_control/deoxys/build/cosine-cartesian-velocity-interpolator-test
dependency/deoxys_control/deoxys/build/cartesian-velocity-control-utils-test
dependency/deoxys_control/deoxys/build/control-exception-utils-test
```

Expected: all targets build and all three tests exit 0.

- [ ] **Step 2: Install and compare Build IDs**

```bash
cmake --install dependency/deoxys_control/deoxys/build
readelf -n dependency/deoxys_control/deoxys/build/franka-interface | rg -A1 "Build ID"
readelf -n dependency/deoxys_control/deoxys/bin/franka-interface | rg -A1 "Build ID"
```

Expected: Build IDs match.

- [ ] **Step 3: Launch exactly one backend with `SCHED_FIFO/80`**

Use `LD_LIBRARY_PATH=$PWD/lib` and capture a new dedicated log. Confirm the process is still alive before asking for user motion.

- [ ] **Step 4: Repeat the rapid direction-change reproduction**

Expected: backend remains alive and no new `cartesian_motion_generator_joint_acceleration_discontinuity` appears. If it fails, stop and add the specified 100 ms 1 kHz ring-buffer telemetry instead of tuning again.

- [ ] **Step 5: Run final verification and commit the root pointer/plans**

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate dex-teleop
export PYTHONNOUSERSITE=1
pytest -q
git -C dependency/deoxys_control status --short
git diff --check
git add dependency/deoxys_control docs/superpowers/plans/2026-07-10-deoxys-cartesian-velocity-rate-limiting.md docs/superpowers/plans/2026-07-10-deoxys-cartesian-velocity-interpolation-continuity.md
git commit -m "fix: stabilize Franka Cartesian velocity motion"
```

Expected: all tests pass, submodule is clean, and the root commit contains only the submodule pointer and the two implementation plans.
