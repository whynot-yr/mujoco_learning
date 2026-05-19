import argparse
import math
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "two_link_arm.xml"


def compute_pd_ctrl(angle, velocity, target_rad, kp, kd, ctrl_limit):
    error = target_rad - angle
    ctrl = kp * error - kd * velocity
    ctrl = float(np.clip(ctrl, -ctrl_limit, ctrl_limit))
    return error, ctrl


def print_state(
    step,
    shoulder_target_rad,
    shoulder_angle,
    shoulder_error,
    elbow_target_rad,
    elbow_angle,
    elbow_error,
    ctrl0,
    ctrl1,
    end_effector_pos,
):
    print(
        f"step={step:5d} | "
        f"shoulder target/current/error="
        f"{math.degrees(shoulder_target_rad):+6.1f}/"
        f"{math.degrees(shoulder_angle):+6.1f}/"
        f"{math.degrees(shoulder_error):+6.1f} deg | "
        f"elbow target/current/error="
        f"{math.degrees(elbow_target_rad):+6.1f}/"
        f"{math.degrees(elbow_angle):+6.1f}/"
        f"{math.degrees(elbow_error):+6.1f} deg | "
        f"ctrl=({ctrl0:+.3f}, {ctrl1:+.3f}) | "
        f"end_effector=({end_effector_pos[0]:+.3f}, {end_effector_pos[1]:+.3f}, {end_effector_pos[2]:+.3f})"
    )


def run_without_viewer(
    model,
    data,
    args,
    shoulder_qpos_adr,
    shoulder_qvel_adr,
    elbow_qpos_adr,
    elbow_qvel_adr,
    end_effector_id,
    shoulder_target_rad,
    elbow_target_rad,
):
    for step in range(args.steps):
        # 多关节控制时，qpos / qvel / ctrl 都是一一对应展开的。
        shoulder_angle = data.qpos[shoulder_qpos_adr]
        shoulder_velocity = data.qvel[shoulder_qvel_adr]
        elbow_angle = data.qpos[elbow_qpos_adr]
        elbow_velocity = data.qvel[elbow_qvel_adr]

        shoulder_error, shoulder_ctrl = compute_pd_ctrl(
            shoulder_angle,
            shoulder_velocity,
            shoulder_target_rad,
            args.kp,
            args.kd,
            args.ctrl_limit,
        )
        elbow_error, elbow_ctrl = compute_pd_ctrl(
            elbow_angle,
            elbow_velocity,
            elbow_target_rad,
            args.kp,
            args.kd,
            args.ctrl_limit,
        )

        data.ctrl[0] = shoulder_ctrl
        data.ctrl[1] = elbow_ctrl
        mujoco.mj_step(model, data)

        if step % args.print_every == 0 or step == args.steps - 1:
            print_state(
                step,
                shoulder_target_rad,
                shoulder_angle,
                shoulder_error,
                elbow_target_rad,
                elbow_angle,
                elbow_error,
                shoulder_ctrl,
                elbow_ctrl,
                data.xpos[end_effector_id],
            )


def run_with_viewer(
    model,
    data,
    args,
    shoulder_qpos_adr,
    shoulder_qvel_adr,
    elbow_qpos_adr,
    elbow_qvel_adr,
    end_effector_id,
    shoulder_target_rad,
    elbow_target_rad,
):
    with mujoco.viewer.launch_passive(model, data) as viewer:
        step = 0

        while viewer.is_running() and step < args.steps:
            step_start = time.time()

            shoulder_angle = data.qpos[shoulder_qpos_adr]
            shoulder_velocity = data.qvel[shoulder_qvel_adr]
            elbow_angle = data.qpos[elbow_qpos_adr]
            elbow_velocity = data.qvel[elbow_qvel_adr]

            shoulder_error, shoulder_ctrl = compute_pd_ctrl(
                shoulder_angle,
                shoulder_velocity,
                shoulder_target_rad,
                args.kp,
                args.kd,
                args.ctrl_limit,
            )
            elbow_error, elbow_ctrl = compute_pd_ctrl(
                elbow_angle,
                elbow_velocity,
                elbow_target_rad,
                args.kp,
                args.kd,
                args.ctrl_limit,
            )

            data.ctrl[0] = shoulder_ctrl
            data.ctrl[1] = elbow_ctrl
            mujoco.mj_step(model, data)

            if step % args.print_every == 0 or step == args.steps - 1:
                print_state(
                    step,
                    shoulder_target_rad,
                    shoulder_angle,
                    shoulder_error,
                    elbow_target_rad,
                    elbow_angle,
                    elbow_error,
                    shoulder_ctrl,
                    elbow_ctrl,
                    data.xpos[end_effector_id],
                )

            viewer.sync()

            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

            step += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shoulder_target_deg", type=float, default=45.0)
    parser.add_argument("--elbow_target_deg", type=float, default=-45.0)
    parser.add_argument("--kp", type=float, default=8.0)
    parser.add_argument("--kd", type=float, default=1.0)
    parser.add_argument("--ctrl_limit", type=float, default=3.0)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--no_viewer", action="store_true")
    parser.add_argument("--print_every", type=int, default=100)
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)

    shoulder_joint_id = model.joint("shoulder_joint").id
    elbow_joint_id = model.joint("elbow_joint").id

    shoulder_qpos_adr = model.jnt_qposadr[shoulder_joint_id]
    shoulder_qvel_adr = model.jnt_dofadr[shoulder_joint_id]
    elbow_qpos_adr = model.jnt_qposadr[elbow_joint_id]
    elbow_qvel_adr = model.jnt_dofadr[elbow_joint_id]
    end_effector_id = model.body("end_effector").id

    shoulder_target_rad = math.radians(args.shoulder_target_deg)
    elbow_target_rad = math.radians(args.elbow_target_deg)

    print("=" * 80)
    print("v0.7 PD Control Two-Link Arm")
    print(f"shoulder_target_deg : {args.shoulder_target_deg}")
    print(f"elbow_target_deg    : {args.elbow_target_deg}")
    print(f"kp                  : {args.kp}")
    print(f"kd                  : {args.kd}")
    print(f"ctrl_limit          : {args.ctrl_limit}")
    print("=" * 80)

    data.qpos[shoulder_qpos_adr] = math.radians(10.0)
    data.qpos[elbow_qpos_adr] = math.radians(10.0)
    mujoco.mj_forward(model, data)

    if args.no_viewer:
        run_without_viewer(
            model,
            data,
            args,
            shoulder_qpos_adr,
            shoulder_qvel_adr,
            elbow_qpos_adr,
            elbow_qvel_adr,
            end_effector_id,
            shoulder_target_rad,
            elbow_target_rad,
        )
    else:
        run_with_viewer(
            model,
            data,
            args,
            shoulder_qpos_adr,
            shoulder_qvel_adr,
            elbow_qpos_adr,
            elbow_qvel_adr,
            end_effector_id,
            shoulder_target_rad,
            elbow_target_rad,
        )


if __name__ == "__main__":
    main()
