import time
from pathlib import Path

import mujoco
import mujoco.viewer


# 1. 找到 XML 模型路径
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "falling_ball.xml"


# 2. 加载 MuJoCo 模型
model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))


# 3. 创建仿真数据对象
data = mujoco.MjData(model)


# 4. 找到 ball 这个 body 的 id，后面用来读取它的位置
ball_id = model.body("ball").id


# 5. 启动 viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    step = 0

    while viewer.is_running():
        step_start = time.time()

        # 6. 推进一步物理仿真
        mujoco.mj_step(model, data)

        # 7. 每 200 步打印一次小球高度
        if step % 200 == 0:
            ball_z = data.xpos[ball_id, 2]
            print(f"step = {step:5d}, ball z = {ball_z:.4f}")

        # 8. 同步 viewer 画面
        viewer.sync()

        # 9. 控制仿真速度，尽量接近真实时间
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

        step += 1