# Franka Backend Exception Diagnostics Design

## Goal

Expose the original `libfranka` exception that currently gets replaced by
`terminate called without an active exception`, while preserving the existing
robot motion commands and controller behavior.

## Current Failure

`franka_control_node.cpp` creates the `control_msg_sub` worker thread before
entering the main robot-control loop. If `robot.control()` throws a
`franka::Exception`, stack unwinding reaches the still-joinable
`control_msg_sub` object before the outer catch handler. Destroying a joinable
`std::thread` invokes `std::terminate`, so the backend exits before logging the
original `libfranka` error.

The latest runtime evidence separates this cleanup defect from the earlier
Oculus pose discontinuity. In the latest failure, the Cartesian velocity
command was smooth and the controller pose did not jump at grip activation,
but the Franka state stream stopped immediately after the backend exception.

## Selected Approach

Wrap each `robot.control()` invocation in a small, testable exception-capture
helper. The helper will return the original exception message instead of
allowing it to unwind past the subscription thread. The main loop will then:

1. log the original exception message;
2. mark the controller as no longer running;
3. request backend termination;
4. leave the control loop;
5. stop the state publisher;
6. join the control-message subscription thread; and
7. return a non-zero process exit code.

This keeps cleanup on the existing normal path and avoids a broad lifecycle
refactor.

## Alternatives Considered

### RAII thread guard

A general thread-lifetime guard could join workers during every stack-unwind
path. This is more comprehensive, but it changes lifecycle behavior beyond the
specific diagnostic need and could interact with the state publisher's own
thread management.

### Debugger-only exception capture

GDB could break on C++ exceptions without changing source. It would complicate
real-time scheduling, make repeated hardware reproduction harder, and would
not permanently improve backend diagnostics.

## Components and Files

- `~/Robots/franka/deoxys_control/deoxys/franka-interface/include/utils/control_exception_utils.h`
  will contain the hardware-independent exception-capture helper and its result
  type.
- `~/Robots/franka/deoxys_control/deoxys/franka-interface/src/franka_control_node.cpp`
  will use the helper around `robot.control()` and route failures through the
  existing orderly shutdown path.
- `~/Robots/franka/deoxys_control/deoxys/franka-interface/src/test/test_control_exception_utils.cpp`
  will exercise the helper without connecting to a robot.
- `~/Robots/franka/deoxys_control/deoxys/CMakeLists.txt` will add the standalone
  diagnostic test executable.

## Error Handling

The helper will capture `franka::Exception` and retain `what()`. The backend
will emit a clear message prefixed with `robot.control failed:` before cleanup.
No automatic retry, motion recovery, controller switch, or command mutation
will occur in this change.

Unexpected non-`franka` exceptions remain outside this diagnostic scope. The
existing outer handler remains in place for exceptions raised before the
subscription thread starts.

## Testing

The standalone C++ test will supply a callable that throws
`franka::Exception("sentinel control failure")`. It must verify that:

- the process does not terminate;
- the helper reports failure;
- the exact sentinel message is retained; and
- a non-throwing callable reports success with no error message.

The implementation cycle will first add the failing test and CMake target,
then add the minimal helper and control-node integration. Verification will
include:

- running the standalone exception test;
- rebuilding only the `franka-interface` target;
- confirming the installed backend binary links successfully with the local
  Deoxys libraries; and
- launching the backend with `SCHED_FIFO` priority 80 and persistent console
  logging.

The final root-cause classification requires one user-driven hardware
reproduction. The resulting original exception will distinguish communication
constraints, motion-generator safety, and other `libfranka` failures.

## Safety and Scope

- Do not change controller gains, action scaling, velocity limits, trajectory
  interpolation, or robot collision behavior.
- Do not operate the robot automatically as part of the test suite.
- Do not auto-restart the diagnostic backend during the reproduction; preserve
  the first original failure.
- Keep existing user changes outside the vendored Deoxys submodule untouched.

## Success Criteria

The diagnostic build is successful when a simulated exception completes the
orderly shutdown test and a real backend failure prints the original
`libfranka` exception instead of only `terminate called without an active
exception`.
