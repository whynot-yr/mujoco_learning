import argparse
import math
import time
from pathlib import Path

import mujoco
import mujoco.viewer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "two_link_arm.xml"


def print_model_info(model, shoulder_qpos_adr, shoulder_qvel_adr, elbow_qpos_adr, elbow_qvel_adr):
    print("=" * 80)
    print("v0.5 Two-Link Arm Experiment")
    print(f"model path : {MODEL_PATH}")
    print(f"model.nq   : {model.nq}")
    print(f"model.nv   : {model.nv}")
    print(f"model.nu   : {model.nu}")
    print(f"shoulder qpos adr : {shoulder_qpos_adr}")
    print(f"shoulder qvel adr : {shoulder_qvel_adr}")
    print(f"elbow qpos adr    : {elbow_qpos_adr}")
    print(f"elbow qvel adr    : {elbow_qvel_adr}")
    print("=" * 80)


def print_state(step, data, shoulder_qpos_adr, shoulder_qvel_adr, elbow_qpos_adr, elbow_qvel_adr, end_effector_id):
    shoulder_angle_rad = data.qpos[shoulder_qpos_adr]
    elbow_angle_rad = data.qpos[elbow_qpos_adr]
    shoulder_vel = data.qvel[shoulder_qvel_adr]
    elbow_vel = data.qvel[elbow_qvel_adr]
    end_effector_pos = data.xpos[end_effector_id]

    print(
        f"step={step:5d} | "
        f"shoulder={math.degrees(shoulder_angle_rad):+7.2f} deg | "
        f"elbow={math.degrees(elbow_angle_rad):+7.2f} deg | "
        f"shoulder_vel={shoulder_vel:+.4f} | "
        f"elbow_vel={elbow_vel:+.4f} | "
        f"end_effector=({end_effector_pos[0]:+.3f}, {end_effector_pos[1]:+.3f}, {end_effector_pos[2]:+.3f}) | "
        f"ctrl=({data.ctrl[0]:+.3f}, {data.ctrl[1]:+.3f})"
    )


def run_without_viewer(
    model,
    data,
    steps,
    print_every,
    shoulder_qpos_adr,
    shoulder_qvel_adr,
    elbow_qpos_adr,
    elbow_qvel_adr,
    end_effector_id,
    shoulder_ctrl,
    elbow_ctrl,
):
    for step in range(steps):
        data.ctrl[0] = shoulder_ctrl
        data.ctrl[1] = elbow_ctrl
        mujoco.mj_step(model, data)

        if step % print_every == 0 or step == steps - 1:
            print_state(
                step,
                data,
                shoulder_qpos_adr,
                shoulder_qvel_adr,
                elbow_qpos_adr,
                elbow_qvel_adr,
                end_effector_id,
            )


def run_with_viewer(
    model,
    data,
    steps,
    print_every,
    shoulder_qpos_adr,
    shoulder_qvel_adr,
    elbow_qpos_adr,
    elbow_qvel_adr,
    end_effector_id,
    shoulder_ctrl,
    elbow_ctrl,
):
    with mujoco.viewer.launch_passive(model, data) as viewer:
        step = 0

        while viewer.is_running() and step < steps:
            step_start = time.time()

            # 多关节系统里，data.ctrl 的每个元素通常对应一个 actuator。
            data.ctrl[0] = shoulder_ctrl
            data.ctrl[1] = elbow_ctrl
            mujoco.mj_step(model, data)

            if step % print_every == 0 or step == steps - 1:
                print_state(
                    step,
                    data,
                    shoulder_qpos_adr,
                    shoulder_qvel_adr,
                    elbow_qpos_adr,
                    elbow_qvel_adr,
                    end_effector_id,
                )

            viewer.sync()

            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

            step += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shoulder_ctrl", type=float, default=0.5)
    parser.add_argument("--elbow_ctrl", type=float, default=-0.5)
    parser.add_argument("--no_viewer", action="store_true")
    parser.add_argument("--steps", type=int, default=2000)
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

    print_model_info(
        model,
        shoulder_qpos_adr,
        shoulder_qvel_adr,
        elbow_qpos_adr,
        elbow_qvel_adr,
    )
    print(f"end_effector body id : {end_effector_id}")
    print(f"shoulder ctrl        : {args.shoulder_ctrl}")
    print(f"elbow ctrl           : {args.elbow_ctrl}")

    # 手动给一个初始姿态，便于观察双关节系统如何共同运动。
    data.qpos[shoulder_qpos_adr] = math.radians(20.0)
    data.qpos[elbow_qpos_adr] = math.radians(-30.0)
    mujoco.mj_forward(model, data)

    if args.no_viewer:
        run_without_viewer(
            model,
            data,
            args.steps,
            args.print_every,
            shoulder_qpos_adr,
            shoulder_qvel_adr,
            elbow_qpos_adr,
            elbow_qvel_adr,
            end_effector_id,
            args.shoulder_ctrl,
            args.elbow_ctrl,
        )
    else:
        run_with_viewer(
            model,
            data,
            args.steps,
            args.print_every,
            shoulder_qpos_adr,
            shoulder_qvel_adr,
            elbow_qpos_adr,
            elbow_qvel_adr,
            end_effector_id,
            args.shoulder_ctrl,
            args.elbow_ctrl,
        )


if __name__ == "__main__":
    main()
