import argparse
import math
import time
from pathlib import Path

import mujoco
import mujoco.viewer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "pendulum.xml"


def print_joint_info(model, joint_id):
    """打印这个 hinge joint 在 qpos / qvel 中的地址。"""
    qpos_adr = model.jnt_qposadr[joint_id]
    qvel_adr = model.jnt_dofadr[joint_id]

    print("=" * 80)
    print("v0.3 Pendulum Experiment")
    print(f"model path      : {MODEL_PATH}")
    print(f"joint name      : pendulum_hinge")
    print(f"joint id        : {joint_id}")
    print(f"qpos address    : {qpos_adr}")
    print(f"qvel address    : {qvel_adr}")
    print(
        "joint range deg : "
        f"{math.degrees(model.jnt_range[joint_id, 0]):.1f} ~ "
        f"{math.degrees(model.jnt_range[joint_id, 1]):.1f}"
    )
    print(f"damping         : {model.dof_damping[qvel_adr]:.4f}")
    print(f"armature        : {model.dof_armature[qvel_adr]:.4f}")
    print("=" * 80)

    return qpos_adr, qvel_adr


def print_state(step, data, qpos_adr, qvel_adr):
    """打印当前角度和角速度。"""
    angle_rad = data.qpos[qpos_adr]
    angle_deg = math.degrees(angle_rad)
    angular_velocity = data.qvel[qvel_adr]

    print(
        f"step={step:5d} | "
        f"angle={angle_rad:+.4f} rad ({angle_deg:+7.2f} deg) | "
        f"angular_velocity={angular_velocity:+.4f} rad/s"
    )


def run_without_viewer(model, data, steps, qpos_adr, qvel_adr):
    """无图形界面时，直接推进固定步数，便于在服务器环境检查模型。"""
    for step in range(steps):
        mujoco.mj_step(model, data)

        if step % 100 == 0 or step == steps - 1:
            print_state(step, data, qpos_adr, qvel_adr)


def run_with_viewer(model, data, steps, qpos_adr, qvel_adr):
    """有 viewer 时，按接近真实时间的速度显示单摆运动。"""
    with mujoco.viewer.launch_passive(model, data) as viewer:
        step = 0

        while viewer.is_running() and step < steps:
            step_start = time.time()

            mujoco.mj_step(model, data)

            if step % 100 == 0 or step == steps - 1:
                print_state(step, data, qpos_adr, qvel_adr)

            viewer.sync()

            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

            step += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no_viewer", action="store_true")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--initial_angle_deg", type=float, default=30.0)
    args = parser.parse_args()

    # 1. model 是静态模型，里面存的是 XML 定义出来的结构和参数。
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))

    # 2. data 是动态状态，里面存的是 qpos / qvel / 接触状态等仿真过程数据。
    data = mujoco.MjData(model)

    # 3. 找到 pendulum_hinge 这个关节的 id。
    joint_id = model.joint("pendulum_hinge").id
    qpos_adr, qvel_adr = print_joint_info(model, joint_id)

    # 4. 手动设置初始角度。
    #    hinge joint 的 qpos 是一个标量，表示关节角度（单位是 rad）。
    initial_angle_rad = math.radians(args.initial_angle_deg)
    data.qpos[qpos_adr] = initial_angle_rad

    # 5. 改完 qpos 之后，需要调用 mj_forward，让 MuJoCo 重新计算几何位置等派生量。
    mujoco.mj_forward(model, data)

    print(
        f"initial angle   : {initial_angle_rad:+.4f} rad "
        f"({args.initial_angle_deg:+.2f} deg)"
    )

    if args.no_viewer:
        run_without_viewer(model, data, args.steps, qpos_adr, qvel_adr)
    else:
        run_with_viewer(model, data, args.steps, qpos_adr, qvel_adr)


if __name__ == "__main__":
    main()
