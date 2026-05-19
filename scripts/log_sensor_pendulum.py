import argparse
import csv
import math
import time
from pathlib import Path

import mujoco
import mujoco.viewer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "sensor_pendulum.xml"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "sensor_logs" / "pendulum_log.csv"


def get_sensor_slice(model, sensor_name):
    sensor_id = model.sensor(sensor_name).id
    adr = model.sensor_adr[sensor_id]
    dim = model.sensor_dim[sensor_id]
    return sensor_id, adr, dim


def print_state(step, time_value, angle, velocity, ctrl, sensor_angle, sensor_velocity, sensor_force):
    print(
        f"step={step:5d} | "
        f"time={time_value:7.4f} | "
        f"qpos={angle:+.4f} rad ({math.degrees(angle):+7.2f} deg) | "
        f"qvel={velocity:+.4f} rad/s | "
        f"ctrl={ctrl:+.4f} | "
        f"sensor_pos={sensor_angle:+.4f} | "
        f"sensor_vel={sensor_velocity:+.4f} | "
        f"sensor_force={sensor_force:+.4f}"
    )


def maybe_save_plot(csv_path, rows):
    """matplotlib 是可选依赖，导图失败也不影响主流程。"""
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"matplotlib 不可用，跳过 PNG 绘图：{exc}")
        return

    times = [row["time"] for row in rows]
    angles_deg = [math.degrees(row["qpos_angle"]) for row in rows]
    sensor_forces = [row["sensor_motor_force"] for row in rows]

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes[0].plot(times, angles_deg, label="angle (deg)")
    axes[0].set_ylabel("Angle (deg)")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(times, sensor_forces, label="motor force", color="tab:red")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Force")
    axes[1].grid(True)
    axes[1].legend()

    fig.tight_layout()

    png_path = csv_path.with_suffix(".png")
    fig.savefig(png_path)
    plt.close(fig)
    print(f"PNG 图已保存到: {png_path}")


def simulate(model, data, args, qpos_adr, qvel_adr, sensor_info):
    rows = []

    for step in range(args.steps):
        # 每一步都给 actuator 一个控制输入。
        data.ctrl[0] = args.ctrl
        mujoco.mj_step(model, data)
        # 再 forward 一次，确保当前 qpos / qvel 对应的 sensor 读数已经同步刷新。
        mujoco.mj_forward(model, data)

        sensor_joint_pos = data.sensordata[sensor_info["joint_pos"][1]]
        sensor_joint_vel = data.sensordata[sensor_info["joint_vel"][1]]
        sensor_motor_force = data.sensordata[sensor_info["motor_force"][1]]

        row = {
            "step": step,
            "time": data.time,
            "qpos_angle": float(data.qpos[qpos_adr]),
            "qvel_angle": float(data.qvel[qvel_adr]),
            "ctrl": float(data.ctrl[0]),
            "sensor_joint_pos": float(sensor_joint_pos),
            "sensor_joint_vel": float(sensor_joint_vel),
            "sensor_motor_force": float(sensor_motor_force),
        }
        rows.append(row)

        if step % args.print_every == 0 or step == args.steps - 1:
            print_state(
                step,
                row["time"],
                row["qpos_angle"],
                row["qvel_angle"],
                row["ctrl"],
                row["sensor_joint_pos"],
                row["sensor_joint_vel"],
                row["sensor_motor_force"],
            )

    return rows


def simulate_with_viewer(model, data, args, qpos_adr, qvel_adr, sensor_info):
    rows = []

    with mujoco.viewer.launch_passive(model, data) as viewer:
        step = 0

        while viewer.is_running() and step < args.steps:
            step_start = time.time()

            data.ctrl[0] = args.ctrl
            mujoco.mj_step(model, data)
            mujoco.mj_forward(model, data)

            sensor_joint_pos = data.sensordata[sensor_info["joint_pos"][1]]
            sensor_joint_vel = data.sensordata[sensor_info["joint_vel"][1]]
            sensor_motor_force = data.sensordata[sensor_info["motor_force"][1]]

            row = {
                "step": step,
                "time": data.time,
                "qpos_angle": float(data.qpos[qpos_adr]),
                "qvel_angle": float(data.qvel[qvel_adr]),
                "ctrl": float(data.ctrl[0]),
                "sensor_joint_pos": float(sensor_joint_pos),
                "sensor_joint_vel": float(sensor_joint_vel),
                "sensor_motor_force": float(sensor_motor_force),
            }
            rows.append(row)

            if step % args.print_every == 0 or step == args.steps - 1:
                print_state(
                    step,
                    row["time"],
                    row["qpos_angle"],
                    row["qvel_angle"],
                    row["ctrl"],
                    row["sensor_joint_pos"],
                    row["sensor_joint_vel"],
                    row["sensor_motor_force"],
                )

            viewer.sync()

            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

            step += 1

    return rows


def write_csv(csv_path, rows):
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "step",
        "time",
        "qpos_angle",
        "qvel_angle",
        "ctrl",
        "sensor_joint_pos",
        "sensor_joint_vel",
        "sensor_motor_force",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ctrl", type=float, default=0.5)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no_viewer", action="store_true")
    parser.add_argument("--print_every", type=int, default=100)
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)

    joint_id = model.joint("pendulum_hinge").id
    qpos_adr = model.jnt_qposadr[joint_id]
    qvel_adr = model.jnt_dofadr[joint_id]

    sensor_info = {
        "joint_pos": get_sensor_slice(model, "pendulum_joint_pos"),
        "joint_vel": get_sensor_slice(model, "pendulum_joint_vel"),
        "motor_force": get_sensor_slice(model, "pendulum_motor_force"),
    }

    print("=" * 80)
    print("v0.6 Sensor Pendulum Logging")
    print(f"model path : {MODEL_PATH}")
    print(f"output csv : {args.output}")
    print(f"nsensor    : {model.nsensor}")
    print(f"nsensordata: {model.nsensordata}")
    for key, (sensor_id, adr, dim) in sensor_info.items():
        print(f"{key:12s} -> sensor_id={sensor_id}, adr={adr}, dim={dim}")
    print("=" * 80)

    data.qpos[qpos_adr] = math.radians(20.0)
    mujoco.mj_forward(model, data)

    if args.no_viewer:
        rows = simulate(model, data, args, qpos_adr, qvel_adr, sensor_info)
    else:
        rows = simulate_with_viewer(model, data, args, qpos_adr, qvel_adr, sensor_info)

    write_csv(args.output, rows)
    print(f"CSV 已保存到: {args.output}")
    maybe_save_plot(args.output, rows)


if __name__ == "__main__":
    main()
