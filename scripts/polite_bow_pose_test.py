#!/usr/bin/env python3
"""Edit and validate the Microduck polite-bow target pose before training.

The script has two modes:

1. Headless physical validation (default). It drives HOME -> BOW, holds the
   bow for several seconds, returns to HOME, and reports trunk orientation,
   foot-contact duty, forbidden ground contacts, and joint tracking error.
   BAM M6 is used by default, matching policy training and deployment rehearsal.

2. Interactive pose editing (``--edit``). Gravity and free-base motion are
   disabled so MuJoCo's Control sliders can be used to compose a candidate.
   Closing the viewer prints a complete ``BOW_TARGET_POSE`` dictionary. Always
   validate that dictionary in the physical mode afterwards.

Examples:

    # Training-aligned BAM validation, headless.
    uv run python scripts/polite_bow_pose_test.py

    # Watch the same validation in real time.
    uv run python scripts/polite_bow_pose_test.py --viewer

    # Validate a JSON candidate without editing the training configuration.
    uv run python scripts/polite_bow_pose_test.py \
        --pose-file polite_bow_pose.json --viewer

    # Edit with sliders, then copy the printed dictionary into the env cfg.
    uv run python scripts/polite_bow_pose_test.py --edit \
        --pose-file polite_bow_pose.json \
        --save-pose-file polite_bow_pose_edited.json

    # Diagnostic only: compare against legacy XML position actuators.
    uv run python scripts/polite_bow_pose_test.py --no-bam
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import math
from pathlib import Path
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np

from infer_policy import (
    BAM_KP_FW,
    BAM_VIN_MIN,
    load_bam_model,
    load_mujoco_with_bam,
)
from mjlab_microduck.tasks.microduck_polite_bow_env_cfg import BOW_TARGET_POSE


REPO_ROOT = Path(__file__).resolve().parent.parent
SCENE_XML = REPO_ROOT / "src/mjlab_microduck/robot/microduck/scene.xml"
TIMESTEP = 0.005


def actuator_joints(model: mujoco.MjModel) -> list[tuple[str, int]]:
    """Return actuated joint names and qpos addresses in actuator order."""
    result: list[tuple[str, int]] = []
    for actuator_id in range(model.nu):
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        joint_name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_JOINT, joint_id
        )
        if not joint_name or joint_name.startswith("passive_"):
            continue
        result.append((joint_name, int(model.jnt_qposadr[joint_id])))
    return result


def load_pose(path: Path | None) -> tuple[dict[str, float], str]:
    """Load a finite numeric pose from JSON, or copy the training cfg default."""
    if path is None:
        return dict(BOW_TARGET_POSE), "BOW_TARGET_POSE from env cfg"
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"pose file does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"invalid JSON in {path}: line {exc.lineno}, column {exc.colno}: {exc.msg}"
        ) from exc
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: top-level JSON value must be an object")

    pose: dict[str, float] = {}
    for name, value in raw.items():
        if not isinstance(name, str):
            raise ValueError(f"{path}: every joint name must be a string")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{path}: {name!r} must have a numeric angle in radians")
        angle = float(value)
        if not math.isfinite(angle):
            raise ValueError(f"{path}: {name!r} must be finite, got {angle}")
        pose[name] = angle
    return pose, str(path.resolve())


def pose_vector(
    model: mujoco.MjModel,
    joints: list[tuple[str, int]],
    pose: dict[str, float],
    source: str,
) -> np.ndarray:
    """Validate a complete named pose and return actuator-order joint targets."""
    expected = {name for name, _ in joints}
    provided = set(pose)
    missing = sorted(expected - provided)
    unknown = sorted(provided - expected)
    problems: list[str] = []
    if missing:
        problems.append(f"missing joints: {', '.join(missing)}")
    if unknown:
        problems.append(f"unknown/unactuated joints: {', '.join(unknown)}")
    if problems:
        raise ValueError(f"{source}: " + "; ".join(problems))

    values: list[float] = []
    for name, _ in joints:
        value = pose[name]
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"{source}: joint {name!r} was not found in the model")
        if model.jnt_limited[joint_id]:
            low, high = (float(v) for v in model.jnt_range[joint_id])
            if not low <= value <= high:
                raise ValueError(
                    f"{source}: {name}={value:.6f} rad is outside its hard "
                    f"limit [{low:.6f}, {high:.6f}]"
                )
        values.append(value)
    return np.asarray(values, dtype=np.float64)


def root_rpy_deg(data: mujoco.MjData) -> tuple[float, float, float]:
    """Floating-base quaternion [w,x,y,z] -> roll, pitch, yaw in degrees."""
    w, x, y, z = (float(v) for v in data.qpos[3:7])
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return tuple(math.degrees(v) for v in (roll, pitch, yaw))


def contact_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    floor_id: int,
    left_foot_id: int,
    right_foot_id: int,
) -> tuple[bool, bool, bool]:
    """Return left-foot, right-foot, and forbidden robot-ground contact."""
    left = False
    right = False
    forbidden = False
    foot_ids = {left_foot_id, right_foot_id}
    for contact in data.contact[: data.ncon]:
        geom1, geom2 = int(contact.geom1), int(contact.geom2)
        if floor_id not in (geom1, geom2):
            continue
        other = geom2 if geom1 == floor_id else geom1
        left |= other == left_foot_id
        right |= other == right_foot_id
        forbidden |= other not in foot_ids
    return left, right, forbidden


def reset_to_stand(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "STAND")
    if key_id < 0:
        raise RuntimeError(f"{SCENE_XML} has no STAND keyframe")
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)


def ordered_pose(
    joints: list[tuple[str, int]], values: np.ndarray
) -> dict[str, float]:
    """Convert actuator-order values to a stable, human-editable mapping."""
    return {
        joint_name: float(value)
        for (joint_name, _), value in zip(joints, values, strict=True)
    }


def print_pose(name: str, joints: list[tuple[str, int]], values: np.ndarray) -> None:
    """Print a pose in the Python syntax used by the environment config."""
    print(f"{name} = {{")
    for joint_name, value in ordered_pose(joints, values).items():
        print(f'    "{joint_name}": {value:.6f},')
    print("}")


def print_pose_json(joints: list[tuple[str, int]], values: np.ndarray) -> None:
    """Print a pose as valid JSON for direct reuse as a pose file."""
    print(json.dumps(ordered_pose(joints, values), indent=2, ensure_ascii=False))


def save_pose_json(
    path: Path, joints: list[tuple[str, int]], values: np.ndarray
) -> None:
    """Save an edited candidate as stable, pretty-printed JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(ordered_pose(joints, values), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def edit_pose(
    pose: dict[str, float], source: str, save_path: Path | None
) -> None:
    """Open a gravity-free slider editor and print the resulting pose."""
    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    data = mujoco.MjData(model)
    reset_to_stand(model, data)
    joints = actuator_joints(model)
    target = pose_vector(model, joints, pose, source)

    model.opt.gravity[:] = 0.0
    data.ctrl[: len(target)] = target
    locked_root = data.qpos[:7].copy()

    print("=== PoliteBow pose editor ===")
    print(f"Initial pose: {source}")
    print("Open the MuJoCo 'Control' panel and adjust the 14 sliders.")
    print("Gravity and the floating base are locked for editing only.")
    print("Close the viewer to print JSON and a copy-ready BOW_TARGET_POSE.\n")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            data.qpos[:7] = locked_root
            data.qvel[:6] = 0.0
            mujoco.mj_step(model, data)
            data.qpos[:7] = locked_root
            data.qvel[:6] = 0.0
            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(1.0 / 60.0)

    captured = np.asarray([data.qpos[address] for _, address in joints])
    print("\n=== Captured candidate (not yet gravity-validated) ===\n")
    print("JSON pose file:\n")
    print_pose_json(joints, captured)
    print("\nPython environment config:\n")
    print_pose("BOW_TARGET_POSE", joints, captured)
    if save_path is not None:
        save_pose_json(save_path, joints, captured)
        print(f"\nSaved JSON candidate to: {save_path.resolve()}")


def summarize(label: str, samples: list[dict[str, float]]) -> dict[str, float]:
    if not samples:
        raise RuntimeError(f"No samples collected for {label}")

    def mean(key: str) -> float:
        return float(np.mean([sample[key] for sample in samples]))

    def maximum(key: str) -> float:
        return float(np.max([sample[key] for sample in samples]))

    result = {
        "roll_mean": mean("roll"),
        "roll_abs_max": maximum("roll_abs"),
        "pitch_mean": mean("pitch"),
        "pitch_abs_max": maximum("pitch_abs"),
        "yaw_abs_max": maximum("yaw_abs"),
        "base_z_min": float(np.min([sample["base_z"] for sample in samples])),
        "left_contact": mean("left_contact"),
        "right_contact": mean("right_contact"),
        "forbidden_contact": mean("forbidden_contact"),
        "joint_error_rms": mean("joint_error_rms"),
        "torque_abs_max": maximum("torque_abs_max"),
    }
    print(f"\n[{label}]")
    print(
        f"  trunk roll mean/max:  {result['roll_mean']:+6.2f} / "
        f"{result['roll_abs_max']:5.2f} deg"
    )
    print(
        f"  trunk pitch mean/max: {result['pitch_mean']:+6.2f} / "
        f"{result['pitch_abs_max']:5.2f} deg"
    )
    print(f"  yaw abs max:           {result['yaw_abs_max']:5.2f} deg")
    print(f"  minimum trunk z:       {result['base_z_min']:.4f} m")
    print(
        "  foot contact duty:    "
        f"left={100*result['left_contact']:5.1f}%  "
        f"right={100*result['right_contact']:5.1f}%"
    )
    print(
        f"  forbidden contact:     {100*result['forbidden_contact']:5.1f}% of steps"
    )
    print(f"  joint tracking RMS:    {result['joint_error_rms']:.4f} rad")
    print(f"  peak actuator torque:  {result['torque_abs_max']:.4f} Nm")
    return result


def validate_pose(
    args: argparse.Namespace, pose: dict[str, float], source: str
) -> bool:
    """Run HOME -> BOW -> HOME and return whether the basic safety gate passes."""
    bam_ctrl = None
    if args.no_bam:
        model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
        model.opt.timestep = TIMESTEP
        data = mujoco.MjData(model)
        actuator_names = [name for name, _ in actuator_joints(model)]
        print("Using legacy XML position actuators (--no-bam diagnostic).")
    else:
        bam_model = load_bam_model(args.kp_fw, args.vin, None)
        vin_drop_gain = args.vin_drop_gain if args.vin_drop_gain > 0 else None
        model, data, bam_ctrl, actuator_names = load_mujoco_with_bam(
            str(SCENE_XML), bam_model, TIMESTEP, vin_drop_gain, BAM_VIN_MIN
        )

    reset_to_stand(model, data)
    joints = actuator_joints(model)
    if actuator_names != [name for name, _ in joints]:
        raise RuntimeError(
            "BAM actuator order does not match the robot joint order: "
            f"{actuator_names} != {[name for name, _ in joints]}"
        )

    home = np.asarray([data.qpos[address] for _, address in joints])
    bow = pose_vector(model, joints, pose, source)
    if bam_ctrl is not None:
        bam_ctrl.reset(data.qpos)

    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    left_foot_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "left_foot_collision"
    )
    right_foot_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "right_foot_collision"
    )
    if min(floor_id, left_foot_id, right_foot_id) < 0:
        raise RuntimeError("Required floor/foot collision geoms were not found")

    print(f"Pose source: {source}")
    phases = (
        ("home", args.home_s, home, home),
        ("bow_ramp", args.ramp_s, home, bow),
        ("bow_hold", args.hold_s, bow, bow),
        ("recovery", args.recovery_s, bow, home),
        ("stand_rest", args.rest_s, home, home),
    )
    hold_samples: list[dict[str, float]] = []
    rest_samples: list[dict[str, float]] = []

    viewer_context = (
        mujoco.viewer.launch_passive(model, data) if args.viewer else nullcontext(None)
    )
    with viewer_context as viewer:
        for phase_name, duration, start, end in phases:
            steps = max(1, round(duration / TIMESTEP))
            print(f"{phase_name:>10}: {duration:.2f} s ({steps} physics steps)")
            for step in range(steps):
                blend = 1.0 if steps == 1 else step / (steps - 1)
                target = start + blend * (end - start)
                if bam_ctrl is not None:
                    bam_ctrl.q_target[:] = target
                    bam_ctrl.update()
                else:
                    data.ctrl[: len(target)] = target
                mujoco.mj_step(model, data)

                if phase_name in ("bow_hold", "stand_rest"):
                    roll, pitch, yaw = root_rpy_deg(data)
                    left, right, forbidden = contact_state(
                        model, data, floor_id, left_foot_id, right_foot_id
                    )
                    actual = np.asarray([data.qpos[address] for _, address in joints])
                    sample = {
                        "roll": roll,
                        "roll_abs": abs(roll),
                        "pitch": pitch,
                        "pitch_abs": abs(pitch),
                        "yaw_abs": abs(yaw),
                        "base_z": float(data.qpos[2]),
                        "left_contact": float(left),
                        "right_contact": float(right),
                        "forbidden_contact": float(forbidden),
                        "joint_error_rms": float(
                            np.sqrt(np.mean((actual - target) ** 2))
                        ),
                        "torque_abs_max": float(np.max(np.abs(data.actuator_force))),
                    }
                    destination = (
                        hold_samples if phase_name == "bow_hold" else rest_samples
                    )
                    destination.append(sample)

                if viewer is not None:
                    viewer.sync()
                    time.sleep(TIMESTEP)

    hold = summarize("BOW HOLD", hold_samples)
    rest = summarize("RECOVERED STAND", rest_samples)
    print("\nTarget sent to the actuators:\n")
    print_pose("BOW_TARGET_POSE", joints, bow)

    checks = {
        "both feet hold >= 90%": (
            min(hold["left_contact"], hold["right_contact"]) >= 0.90
        ),
        "no forbidden hold contact": hold["forbidden_contact"] == 0.0,
        "hold roll max < 10 deg": hold["roll_abs_max"] < 10.0,
        "hold tilt remains < 45 deg": (
            max(hold["roll_abs_max"], hold["pitch_abs_max"]) < 45.0
        ),
        f"bow depth >= {args.min_bow_deg:g} deg": (
            abs(hold["pitch_mean"]) >= args.min_bow_deg
        ),
        "recovered stand tilt < 10 deg": (
            max(rest["roll_abs_max"], rest["pitch_abs_max"]) < 10.0
        ),
        "recovered joint RMS < 0.15 rad": rest["joint_error_rms"] < 0.15,
    }
    print("\n[CHECKS]")
    for label, passed in checks.items():
        print(f"  {'PASS' if passed else 'FAIL'}  {label}")
    passed = all(checks.values())
    print(f"\nOVERALL: {'PASS' if passed else 'FAIL'}")
    return passed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--edit", action="store_true", help="Open the gravity-free slider editor"
    )
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="Render physical validation in real time",
    )
    parser.add_argument(
        "--no-bam", action="store_true", help="Use legacy XML position actuators"
    )
    parser.add_argument(
        "--pose-file",
        type=Path,
        help="Load a complete 14-joint pose from a JSON file (angles in radians)",
    )
    parser.add_argument(
        "--save-pose-file",
        type=Path,
        help="With --edit, save the captured slider pose to this JSON file",
    )
    parser.add_argument("--home-s", type=float, default=0.5)
    parser.add_argument("--ramp-s", type=float, default=1.2)
    parser.add_argument("--hold-s", type=float, default=3.0)
    parser.add_argument("--recovery-s", type=float, default=1.2)
    parser.add_argument("--rest-s", type=float, default=1.2)
    parser.add_argument("--min-bow-deg", type=float, default=8.0)
    parser.add_argument("--kp-fw", type=float, default=BAM_KP_FW)
    parser.add_argument("--vin", type=float, default=7.4)
    parser.add_argument("--vin-drop-gain", type=float, default=0.1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.save_pose_file is not None and not args.edit:
        print(
            "ERROR: --save-pose-file can only be used together with --edit",
            file=sys.stderr,
        )
        return 2
    try:
        pose, source = load_pose(args.pose_file)
        if args.edit:
            edit_pose(pose, source, args.save_pose_file)
            return 0
        return 0 if validate_pose(args, pose, source) else 1
    except (RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
