from __future__ import annotations

import argparse
import logging
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np

from dex_teleop.core.config import (
    ConfigError,
    apply_cli_overrides,
    load_config,
    resolve_repo_path,
)
from dex_teleop.core.loop import RateLimiter
from dex_teleop.diagnostics import ControlTracer, trace_event, trace_scope
from dex_teleop.devices.mock import MockControllerInput
from dex_teleop.devices.oculus import OculusDevice
from dex_teleop.core.types import ControllerAction, Twist
from dex_teleop.robots.franka import make_franka_robot
from dex_teleop.robots.mock import MockArm, MockGripper
from dex_teleop.robots.robot import Robot
from dex_teleop.teleop import CalibrationError, ClutchMapper
from dex_teleop.visualization import Controller3DVisualizer


logger = logging.getLogger(__name__)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Oculus to Franka teleoperation")
    parser.add_argument("--config", required=True)
    parser.add_argument("--duration", type=float)
    parser.add_argument("--rate-hz", type=float)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--dry-run", dest="dry_run", action="store_true", default=None)
    parser.add_argument("--real-robot", dest="dry_run", action="store_false")
    parser.add_argument(
        "--mock-controller",
        dest="mock_controller",
        action="store_true",
        default=None,
    )
    parser.add_argument(
        "--real-controller", dest="mock_controller", action="store_false"
    )
    parser.add_argument(
        "--print-commands",
        action="store_true",
        help="Print controller input, robot state, targets, and gripper commands each cycle",
    )
    parser.add_argument(
        "--visualize-controllers",
        "--visualize-controller",
        dest="visualize_controllers",
        action="store_true",
        help="Show live controller positions in a non-blocking 3D window",
    )
    parser.add_argument(
        "--record-trace",
        type=Path,
        help="Append a structured diagnostic control trace to this JSONL file",
    )
    parser.add_argument(
        "--trace-queue-size",
        type=int,
        default=4096,
        help="Maximum number of pending trace records (default: 4096)",
    )
    return parser.parse_args(argv)


def _make_device(cfg):
    if cfg["input"].get("mock_controller", False):
        return MockControllerInput(dt=1 / float(cfg["app"]["rate_hz"]))
    control_rate_hz = float(cfg["app"]["rate_hz"])
    velocity_cutoff_hz = min(5.0, 0.45 * control_rate_hz)
    return OculusDevice(
        ip=cfg["input"].get("ip"),
        port=int(cfg["input"].get("port", 5555)),
        stale_timeout=float(cfg["input"].get("stale_timeout", 0.15)),
        velocity_cutoff_hz=velocity_cutoff_hz,
    )


def _make_robots(cfg: dict[str, Any]) -> dict[str, Robot]:
    result = {}
    for arm_id, arm_cfg in cfg["arms"].items():
        has_gripper = arm_cfg.get("gripper") is not None
        if cfg["app"].get("dry_run", False):
            result[arm_id] = Robot(
                MockArm(), MockGripper() if has_gripper else None
            )
        else:
            if arm_cfg.get("type") != "franka":
                raise ConfigError(f"Unsupported arm type: {arm_cfg.get('type')!r}")
            path = resolve_repo_path(arm_cfg["interface_cfg"])
            if not path.exists():
                raise ConfigError(f"Interface config does not exist: {path}")
            result[arm_id] = make_franka_robot(
                str(path),
                with_gripper=has_gripper,
                state_timeout=float(arm_cfg.get("state_timeout", 0.2)),
                max_dt=float(arm_cfg.get("max_dt", 0.1)),
            )
    return result


def _make_mappers(cfg):
    safety = cfg["safety"]
    return {
        arm_id: ClutchMapper(
            position_scale=float(cfg["scales"]["position"]),
            rotation_scale=float(cfg["scales"]["rotation"]),
            max_translation_from_anchor_m=float(
                safety["max_translation_from_anchor_m"]
            ),
            max_rotation_from_anchor_rad=float(
                safety["max_rotation_from_anchor_rad"]
            ),
        )
        for arm_id in cfg["arms"]
    }


def _vector(values):
    return "[" + ", ".join(f"{float(value):+.4f}" for value in values) + "]"


def _span(tracer: ControlTracer | None, record_type: str, **metadata: Any):
    if tracer is None:
        return nullcontext()
    return tracer.span(record_type, **metadata)


def run(
    cfg: dict[str, Any],
    duration=None,
    *,
    print_commands=False,
    tracer: ControlTracer | None = None,
    visualize_controllers: bool = False,
    visualizer: Controller3DVisualizer | None = None,
) -> int:
    device = _make_device(cfg)
    robots = _make_robots(cfg)
    mappers = _make_mappers(cfg)
    if visualize_controllers and visualizer is None:
        visualizer = Controller3DVisualizer()
    period = 1 / float(cfg["app"]["rate_hz"])
    rate = RateLimiter(float(cfg["app"]["rate_hz"]))
    deadline = None if duration is None else time.monotonic() + float(duration)
    home_active = {arm_id: False for arm_id in robots}
    home_action_down = {arm_id: False for arm_id in robots}
    calibration_action_down = {arm_id: False for arm_id in robots}
    if tracer is not None:
        tracer.event(
            "run.configuration",
            config=cfg,
            duration_s=duration,
            configured_period_s=period,
        )
    try:
        with _span(tracer, "startup"):
            trace_event("device.start.request", device_type=type(device).__name__)
            device.start()
            trace_event("device.start.completed")
            for arm_id, robot in robots.items():
                with trace_scope(arm_id=arm_id):
                    trace_event(
                        "robot.connect.request",
                        arm_type=type(robot.arm).__name__,
                        end_effector_type=(
                            None
                            if robot.end_effector is None
                            else type(robot.end_effector).__name__
                        ),
                    )
                    robot.connect()
                    trace_event("robot.connect.completed")

        iteration = 0
        while deadline is None or time.monotonic() < deadline:
            with _span(
                tracer,
                "control_cycle",
                iteration=iteration,
                configured_period_s=period,
            ):
                trace_event("device.read.request")
                device_state = device.read()
                trace_event("device.read.result", state=device_state)
                if visualizer is not None:
                    visualizer.update(device_state)
                for arm_id, robot in robots.items():
                    binding = cfg["arms"][arm_id]
                    controller_name = binding["controller"]
                    controller = device_state.controllers.get(controller_name)
                    mapper = mappers[arm_id]
                    with trace_scope(
                        arm_id=arm_id,
                        controller_name=controller_name,
                    ):
                        trace_event(
                            "control.binding",
                            controller=controller,
                            controller_present=controller is not None,
                        )
                        home_signal = (
                            False
                            if controller is None
                            else controller.action(ControllerAction.HOME)
                        )
                        calibration_signal = (
                            False
                            if controller is None
                            else controller.action(ControllerAction.CALIBRATE)
                        )
                        home_triggered = False
                        calibration_triggered = False
                        if controller is not None:
                            home_triggered = (
                                home_signal and not home_action_down[arm_id]
                            )
                            calibration_triggered = (
                                calibration_signal
                                and not calibration_action_down[arm_id]
                            )
                            home_action_down[arm_id] = home_signal
                            calibration_action_down[arm_id] = calibration_signal
                        else:
                            home_action_down[arm_id] = False
                            calibration_action_down[arm_id] = False
                        trace_event(
                            "control.home",
                            action=ControllerAction.HOME.value,
                            signal=home_signal,
                            triggered=home_triggered,
                            active=home_active[arm_id],
                        )
                        clutch = (
                            controller is not None
                            and controller.button(cfg["controls"]["clutch_button"])
                        )
                        trace_event("control.clutch", engaged=clutch)
                        trace_event(
                            "control.calibration",
                            action=ControllerAction.CALIBRATE.value,
                            signal=calibration_signal,
                            triggered=calibration_triggered,
                            calibrated=mapper.calibration.is_calibrated,
                            pointer_direction=(
                                None
                                if controller is None
                                else controller.pointer_direction
                            ),
                        )
                        if calibration_signal:
                            mapper.release()
                            home_active[arm_id] = False
                            trace_event(
                                "teleop.clutch.released", reason="CALIBRATE"
                            )
                            trace_event("arm.stop.request", reason="CALIBRATE")
                            robot.arm.stop()
                            trace_event("arm.stop.completed", reason="CALIBRATE")
                            calibration_error = None
                            rotation = None
                            if calibration_triggered:
                                try:
                                    if clutch:
                                        raise CalibrationError(
                                            "release the grip/clutch before calibration"
                                        )
                                    if controller.pointer_direction is None:
                                        raise CalibrationError(
                                            "controller pointer direction is unavailable"
                                        )
                                    rotation = mapper.calibration.calibrate(
                                        controller.pointer_direction
                                    )
                                except CalibrationError as exc:
                                    calibration_error = str(exc)
                                    logger.warning(
                                        "Calibration failed for arm %s from %s: %s",
                                        arm_id,
                                        controller_name,
                                        exc,
                                    )
                                    trace_event(
                                        "teleop.calibration.failed",
                                        exception=exc,
                                    )
                                else:
                                    logger.info(
                                        "Calibrated arm %s from %s; R_robot_from_vr=%s",
                                        arm_id,
                                        controller_name,
                                        np.array2string(rotation, precision=4),
                                    )
                                    trace_event(
                                        "teleop.calibration.updated",
                                        robot_from_vr_rotation=rotation,
                                    )
                            if print_commands and calibration_triggered:
                                if calibration_error is None:
                                    print(
                                        f"[{iteration:04d}] arm={arm_id} "
                                        f"controller={controller_name} "
                                        f"action={ControllerAction.CALIBRATE.value} "
                                        f"R_robot_from_vr="
                                        f"{np.array2string(rotation, precision=4)}",
                                        flush=True,
                                    )
                                else:
                                    print(
                                        f"[{iteration:04d}] arm={arm_id} "
                                        f"controller={controller_name} "
                                        f"action={ControllerAction.CALIBRATE.value} "
                                        f"failed={calibration_error}",
                                        flush=True,
                                    )
                            continue

                        if home_triggered:
                            home_active[arm_id] = True
                        if home_active[arm_id]:
                            mapper.release()
                            trace_event("teleop.clutch.released", reason="HOME")
                            trace_event(
                                "arm.home.request",
                                action=ControllerAction.HOME.value,
                                dt=period,
                            )
                            robot.arm.home(dt=period)
                            home_completed = robot.arm.is_settled
                            home_active[arm_id] = not home_completed
                            trace_event(
                                "arm.home.completed",
                                action=ControllerAction.HOME.value,
                                target_reached=home_completed,
                                motion_state=robot.arm.motion_state.value,
                                is_settled=robot.arm.is_settled,
                            )
                            if print_commands:
                                q_cmd = getattr(
                                    robot.arm,
                                    "last_commanded_joint_positions",
                                    None,
                                )
                                print(
                                    f"[{iteration:04d}] arm={arm_id} "
                                    f"action={ControllerAction.HOME.value} "
                                    f"state={robot.arm.motion_state.value.upper()} "
                                    f"q_cmd={'n/a' if q_cmd is None else _vector(q_cmd)}",
                                    flush=True,
                                )
                        elif not clutch:
                            mapper.release()
                            trace_event("teleop.clutch.released")
                            trace_event("arm.stop.request")
                            robot.arm.stop()
                            trace_event("arm.stop.completed")
                            if print_commands:
                                print(
                                    f"[{iteration:04d}] arm={arm_id} clutch=OFF "
                                    f"controller={'missing' if controller is None else controller_name} "
                                    "action=HOLD",
                                    flush=True,
                                )
                        elif not mapper.calibration.is_calibrated:
                            mapper.release()
                            trace_event(
                                "teleop.clutch.released",
                                reason="CALIBRATION_REQUIRED",
                            )
                            trace_event(
                                "arm.stop.request", reason="CALIBRATION_REQUIRED"
                            )
                            robot.arm.stop()
                            trace_event(
                                "arm.stop.completed", reason="CALIBRATION_REQUIRED"
                            )
                            if print_commands:
                                print(
                                    f"[{iteration:04d}] arm={arm_id} clutch=ON "
                                    f"controller={controller_name} "
                                    "action=CALIBRATION_REQUIRED",
                                    flush=True,
                                )
                        else:
                            newly_anchored = mapper.anchor is None
                            trace_event("arm.state.request", purpose="teleop_mapping")
                            arm_state = robot.arm.get_state()
                            trace_event(
                                "arm.state.result",
                                purpose="teleop_mapping",
                                state=arm_state,
                            )
                            target = mapper.update(
                                controller.pose, arm_state.end_effector_pose
                            )
                            target_twist = None
                            if controller.twist is not None:
                                target_twist = (
                                    Twist(np.zeros(3), np.zeros(3))
                                    if newly_anchored
                                    else mapper.map_twist(controller.twist)
                                )
                            trace_event(
                                "teleop.target.result",
                                controller_pose=controller.pose,
                                controller_twist=controller.twist,
                                measured_end_effector_pose=arm_state.end_effector_pose,
                                target_pose=target,
                                target_twist=target_twist,
                            )
                            trace_event(
                                "arm.ik_joint_position.request",
                                pose=target,
                                twist_ref=target_twist,
                                delta=False,
                                dt=period,
                            )
                            robot.arm.ik_joint_position(
                                target,
                                twist_ref=target_twist,
                                dt=period,
                            )
                            trace_event("arm.ik_joint_position.completed")
                            if print_commands:
                                q_ik = getattr(
                                    robot.arm, "last_ik_joint_positions", None
                                )
                                q_cmd = getattr(
                                    robot.arm, "last_commanded_joint_positions", None
                                )
                                interface = getattr(robot.arm, "interface", None)
                                q_des = getattr(interface, "last_q_d", None)
                                print(
                                    f"[{iteration:04d}] arm={arm_id} clutch=ON "
                                    f"controller_pos={_vector(controller.pose.position)} "
                                    f"controller_v={'n/a' if controller.twist is None else _vector(controller.twist.linear)} "
                                    f"controller_w={'n/a' if controller.twist is None else _vector(controller.twist.angular)} "
                                    f"eef_pos={_vector(arm_state.end_effector_pose.position)} "
                                    f"target_pos={_vector(target.position)} "
                                    f"target_v={'n/a' if target_twist is None else _vector(target_twist.linear)} "
                                    f"target_w={'n/a' if target_twist is None else _vector(target_twist.angular)} "
                                    f"q_meas={_vector(arm_state.joint_positions)} "
                                    f"dq_meas={_vector(arm_state.joint_velocities)} "
                                    f"q_des={'n/a' if q_des is None else _vector(q_des)} "
                                    f"q_ik={'n/a' if q_ik is None else _vector(q_ik)} "
                                    f"q_cmd={'n/a' if q_cmd is None else _vector(q_cmd)}",
                                    flush=True,
                                )
                        if robot.end_effector is not None and controller is not None:
                            trigger = controller.axis(cfg["controls"]["gripper_axis"])
                            normalized = 1.0 - trigger
                            trace_event(
                                "end_effector.position.request",
                                raw_trigger=trigger,
                                normalized_position=normalized,
                            )
                            robot.end_effector.position(normalized)
                            trace_event("end_effector.position.completed")
                            if print_commands:
                                print(
                                    f"         gripper trigger={trigger:.3f} "
                                    f"normalized={normalized:.3f} "
                                    "(0=closed, 1=open)",
                                    flush=True,
                                )
                trace_event("rate.sleep.request")
                rate.sleep()
                trace_event("rate.sleep.completed")
            iteration += 1
    except KeyboardInterrupt:
        if tracer is not None:
            tracer.event("run.interrupted")
    finally:
        with _span(tracer, "shutdown"):
            for arm_id, robot in robots.items():
                with trace_scope(arm_id=arm_id):
                    trace_event("robot.close.request")
                    try:
                        robot.close()
                    except Exception as exc:
                        trace_event("robot.close.failed", exception=exc)
                        logger.warning("Failed to close %s: %s", arm_id, exc)
                    else:
                        trace_event("robot.close.completed")
            trace_event("device.stop.request")
            try:
                device.stop()
            except Exception as exc:
                trace_event("device.stop.failed", exception=exc)
                logger.warning("Failed to stop input device: %s", exc)
            else:
                trace_event("device.stop.completed")
            if visualizer is not None:
                visualizer.close()
    return 0


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    tracer: ControlTracer | None = None
    result = 1
    try:
        if args.record_trace is not None:
            tracer = ControlTracer(
                args.record_trace, queue_size=args.trace_queue_size
            )
            logger.info("Recording control trace to %s", tracer.path)
            tracer.event("session.start", arguments=vars(args))
        cfg = apply_cli_overrides(
            load_config(args.config),
            dry_run=args.dry_run,
            mock_controller=args.mock_controller,
            rate_hz=args.rate_hz,
        )
        result = run(
            cfg,
            args.duration,
            print_commands=args.print_commands,
            tracer=tracer,
            visualize_controllers=args.visualize_controllers,
        )
    except ConfigError as exc:
        if tracer is not None:
            tracer.event("session.config_error", exception=exc)
        print(f"Config error: {exc}")
        result = 2
    except Exception as exc:
        if tracer is not None:
            tracer.event("session.error", exception=exc)
        print(f"Teleop error: {exc}")
        result = 1
    finally:
        if tracer is not None:
            tracer.event("session.end", exit_code=result)
            tracer.close()
    return result


if __name__ == "__main__":
    raise SystemExit(main())
