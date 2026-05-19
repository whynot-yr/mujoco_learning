import argparse
import math
import time
from pathlib import Path

import mujoco
import mujoco.viewer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "motor_pendulum.xml"


def actuator_name(model, actuator_id):
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)


def print_model_info(model):
    """说明 model 里有哪些 actuator，以及控制范围是什么。"""
    print("=" * 80)
    print("v0.4 Motor Pendulum Experiment")
    print(f"model path : {MODEL_PATH}")
    print(f"model.nu   : {model.nu}")

    for actuator_id in range(model.nu):
        name = actuator_name(model, actuator_id)
        ctrl_min = model.actuator_ctrlrange[actuator_id, 0]
        ctrl_max = model.actuator_ctrlrange[actuator_id, 1]
        is_limited = bool(model.actuator_ctrllimited[actuator_id])

        print(
            f"actuator[{actuator_id}] = {name}, "
            f"ctrllimited={is_limited}, ctrlrange=[{ctrl_min:.2f}, {ctrl_max:.2f}]"
        )

    print("=" * 80)


def print_state(step, data, qpos_adr, qvel_adr):
    angle_rad = data.qpos[qpos_adr]
    angle_deg = math.degrees(angle_rad)
    angular_velocity = data.qvel[qvel_adr]
    ctrl = data.ctrl[0]

    print(
        f"step={step:5d} | "
        f"angle={angle_rad:+.4f} rad ({angle_deg:+7.2f} deg) | "
        f"angular_velocity={angular_velocity:+.4f} rad/s | "
        f"ctrl={ctrl:+.4f}"
    )


def run_without_viewer(model, data, steps, print_every, qpos_adr, qvel_adr, ctrl_value):
    for step in range(steps):
        # data.ctrl 是控制输入向量。这里把第 0 个 actuator 的输入设成常数。
        data.ctrl[0] = ctrl_value

        # actuator 会把 ctrl 转换成作用在 joint 上的力矩，再进入动力学计算。
        mujoco.mj_step(model, data)

        if step % print_every == 0 or step == steps - 1:
            print_state(step, data, qpos_adr, qvel_adr)


def run_with_viewer(model, data, steps, print_every, qpos_adr, qvel_adr, ctrl_value):
    with mujoco.viewer.launch_passive(model, data) as viewer:
        step = 0

        while viewer.is_running() and step < steps:
            step_start = time.time()

            # model 是静态结构，data 是当前时刻的动态状态。
            # 这里每一步都给 data.ctrl 赋值，表示当前时刻的控制输入。
            data.ctrl[0] = ctrl_value
            mujoco.mj_step(model, data)

            if step % print_every == 0 or step == steps - 1:
                print_state(step, data, qpos_adr, qvel_adr)

            viewer.sync()

            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

            step += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ctrl", type=float, default=0.5)
    parser.add_argument("--no_viewer", action="store_true")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--print_every", type=int, default=100)
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)

    joint_id = model.joint("pendulum_hinge").id
    qpos_adr = model.jnt_qposadr[joint_id]
    qvel_adr = model.jnt_dofadr[joint_id]

    print_model_info(model)
    print(f"joint qpos adr : {qpos_adr}")
    print(f"joint qvel adr : {qvel_adr}")
    print(f"constant ctrl  : {args.ctrl}")

    # 给一个初始角度，避免一开始恰好处于平衡附近，看不出 actuator 的作用。
    data.qpos[qpos_adr] = math.radians(20.0)
    mujoco.mj_forward(model, data)

    if args.no_viewer:
        run_without_viewer(model, data, args.steps, args.print_every, qpos_adr, qvel_adr, args.ctrl)
    else:
        run_with_viewer(model, data, args.steps, args.print_every, qpos_adr, qvel_adr, args.ctrl)


if __name__ == "__main__":
    main()
