import argparse
import math
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "motor_pendulum.xml"


def compute_pd_ctrl(angle, velocity, target_rad, kp, kd, ctrl_limit):
    """PD 控制器：
    P 项根据角度误差给出“朝目标方向拉回去”的控制。
    D 项根据角速度给出阻尼，避免摆动太大。
    clip 用来把控制量限制在 actuator 能接受的范围内。
    """
    error = target_rad - angle
    ctrl = kp * error - kd * velocity
    ctrl = float(np.clip(ctrl, -ctrl_limit, ctrl_limit))
    return error, ctrl


def print_state(step, target_rad, angle, velocity, error, ctrl):
    print(
        f"step={step:5d} | "
        f"target={math.degrees(target_rad):+7.2f} deg | "
        f"current={math.degrees(angle):+7.2f} deg | "
        f"error={math.degrees(error):+7.2f} deg | "
        f"angular_velocity={velocity:+.4f} rad/s | "
        f"ctrl={ctrl:+.4f}"
    )


def run_without_viewer(model, data, args, qpos_adr, qvel_adr, target_rad):
    for step in range(args.steps):
        angle = data.qpos[qpos_adr]
        velocity = data.qvel[qvel_adr]
        error, ctrl = compute_pd_ctrl(angle, velocity, target_rad, args.kp, args.kd, args.ctrl_limit)

        data.ctrl[0] = ctrl
        mujoco.mj_step(model, data)

        if step % args.print_every == 0 or step == args.steps - 1:
            print_state(step, target_rad, angle, velocity, error, ctrl)


def run_with_viewer(model, data, args, qpos_adr, qvel_adr, target_rad):
    with mujoco.viewer.launch_passive(model, data) as viewer:
        step = 0

        while viewer.is_running() and step < args.steps:
            step_start = time.time()

            # 闭环控制的关键：每一步都根据“当前状态”重新计算一次 ctrl。
            angle = data.qpos[qpos_adr]
            velocity = data.qvel[qvel_adr]
            error, ctrl = compute_pd_ctrl(angle, velocity, target_rad, args.kp, args.kd, args.ctrl_limit)

            data.ctrl[0] = ctrl
            mujoco.mj_step(model, data)

            if step % args.print_every == 0 or step == args.steps - 1:
                print_state(step, target_rad, angle, velocity, error, ctrl)

            viewer.sync()

            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

            step += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_deg", type=float, default=45.0)
    parser.add_argument("--kp", type=float, default=5.0)
    parser.add_argument("--kd", type=float, default=0.5)
    parser.add_argument("--ctrl_limit", type=float, default=2.0)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--no_viewer", action="store_true")
    parser.add_argument("--print_every", type=int, default=100)
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)

    joint_id = model.joint("pendulum_hinge").id
    qpos_adr = model.jnt_qposadr[joint_id]
    qvel_adr = model.jnt_dofadr[joint_id]
    target_rad = math.radians(args.target_deg)

    print("=" * 80)
    print("v0.7 PD Control Pendulum")
    print(f"target_deg : {args.target_deg}")
    print(f"kp         : {args.kp}")
    print(f"kd         : {args.kd}")
    print(f"ctrl_limit : {args.ctrl_limit}")
    print("=" * 80)

    # 初始状态和目标状态不同，便于观察控制器如何逐步逼近目标。
    data.qpos[qpos_adr] = math.radians(-20.0)
    mujoco.mj_forward(model, data)

    if args.no_viewer:
        run_without_viewer(model, data, args, qpos_adr, qvel_adr, target_rad)
    else:
        run_with_viewer(model, data, args, qpos_adr, qvel_adr, target_rad)


if __name__ == "__main__":
    main()
