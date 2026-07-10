# Remove Cartesian Velocity Failure Telemetry Design

## Goal

Remove the temporary 1 kHz Cartesian velocity failure telemetry added while
diagnosing Franka motion-generator discontinuities. Preserve all control,
safety, and exception-reporting behavior.

## Scope

Remove:

- the fixed-capacity Cartesian velocity diagnostic buffer;
- its `SharedMemory` member;
- the per-callback `Record` call;
- buffer clear, snapshot, and 100-frame debug dump on control failure;
- the dedicated C++ telemetry test and CMake target.

Keep:

- the original `robot.control failed` error with the exact libfranka reason;
- safe thread shutdown and read/control exception handling;
- libfranka Cartesian rate limiting;
- continuous 1 kHz cosine interpolation;
- `time_fraction: 0.5` and its configuration regression test;
- clutch-release braking in the Python policy;
- existing JSONL and Deoxys log files already stored on disk.

## Rationale

The buffer was diagnostic instrumentation, not part of motion generation. The
root causes are now established and grip-release braking passed hardware
acceptance. Removing the buffer eliminates unnecessary work in the 1 kHz
callback and prevents large failure dumps from rotating normal backend logs.

Exact libfranka exception logging remains available if a future fault occurs.

## Verification

Before deletion, run a source scan that fails while telemetry references are
present. After deletion, require no telemetry symbol or diagnostic-dump string
outside historical logs and documentation.

Build and run the remaining focused Deoxys tests:

- control exception handling;
- Cartesian velocity rate limiting;
- cosine Cartesian velocity interpolation;
- Cartesian velocity controller configuration.

Rebuild and install the backend, verify build/install Build IDs match, and run
the complete root repository pytest suite. No hardware motion is required
because this cleanup must not change the control command path.
