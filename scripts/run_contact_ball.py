import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "contact_ball.xml"


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--vx", type=float, default=1.5)
    parser.add_argument("--height", type=float, default=1.2)

    parser.add_argument("--friction", type=float, default=0.8)
    parser.add_argument("--timeconst", type=float, default=0.01)
    parser.add_argument("--dampratio", type=float, default=0.7)

    args = parser.parse_args()

    # 1. 加载模型
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)

    # 2. 找到 body / joint / geom 的 id
    ball_body_id = model.body("ball").id
    ball_joint_id = model.joint("ball_free").id

    floor_geom_id = model.geom("floor").id
    ball_geom_id = model.geom("ball_geom").id

    # 3. 动态修改摩擦和接触参数
    for geom_id in [floor_geom_id, ball_geom_id]:
        model.geom_friction[geom_id] = [args.friction, 0.05, 0.01]
        model.geom_solref[geom_id] = [args.timeconst, args.dampratio]

    # 4. 找到 freejoint 在 qpos / qvel 中的位置
    qpos_adr = model.jnt_qposadr[ball_joint_id]
    qvel_adr = model.jnt_dofadr[ball_joint_id]

    # freejoint 的 qpos 是 7 维：
    # [x, y, z, qw, qx, qy, qz]
    data.qpos[qpos_adr : qpos_adr + 3] = [-1.0, 0.0, args.height]
    data.qpos[qpos_adr + 3 : qpos_adr + 7] = [1.0, 0.0, 0.0, 0.0]

    # freejoint 的 qvel 是 6 维：
    # [vx, vy, vz, wx, wy, wz]
    data.qvel[qvel_adr : qvel_adr + 6] = [args.vx, 0.0, 0.0, 0.0, 0.0, 0.0]

    # 5. 根据手动设置的 qpos/qvel 更新一次状态
    mujoco.mj_forward(model, data)

    print("=" * 80)
    print("v0.2 Contact Ball Experiment")
    print(f"model path  : {MODEL_PATH}")
    print(f"init vx     : {args.vx}")
    print(f"init height : {args.height}")
    print(f"friction    : {args.friction}")
    print(f"solref      : {args.timeconst} {args.dampratio}")
    print("=" * 80)

    # 6. 启动 viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        step = 0

        while viewer.is_running():
            step_start = time.time()

            mujoco.mj_step(model, data)

            if step % 100 == 0:
                ball_x = data.xpos[ball_body_id, 0]
                ball_z = data.xpos[ball_body_id, 2]

                vx = data.qvel[qvel_adr + 0]
                vz = data.qvel[qvel_adr + 2]

                num_contacts = data.ncon

                print(
                    f"step={step:5d} | "
                    f"x={ball_x: .3f}, z={ball_z: .3f} | "
                    f"vx={vx: .3f}, vz={vz: .3f} | "
                    f"contacts={num_contacts}"
                )

            viewer.sync()

            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

            step += 1


if __name__ == "__main__":
    main()