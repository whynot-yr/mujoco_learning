# v0.5 Body Tree And Two-Link Arm

## 1. body tree 是什么

MuJoCo 里的刚体通常不是平铺摆着的，而是组织成一棵树。

- 父 body 的运动会影响子 body
- 子 body 的位姿是相对父 body 定义的

机械臂非常适合用这个思路理解，因为“上一节连杆带着下一节连杆一起动”。

## 2. 为什么 link2 写在 link1 的 body 里面

因为 `link2` 是装在 `link1` 末端上的。

把 `link2` 写成 `link1` 的子 body，表示：

- `link2` 的根位置是相对 `link1` 定义的
- 当 `link1` 旋转时，`link2` 会跟着一起移动

这就是机械臂层级结构最核心的直觉。

## 3. shoulder_joint 和 elbow_joint 各自控制什么

- `shoulder_joint` 控制第一根连杆相对 base 的转动
- `elbow_joint` 控制第二根连杆相对 link1 的转动

所以：

- 肩关节决定大范围方向
- 肘关节决定末端更细的姿态调整

## 4. 为什么 qpos 中会有两个角度

因为这个模型有两个 hinge joint：

- `shoulder_joint`
- `elbow_joint`

每个 hinge joint 贡献 1 个角度，所以 `qpos` 里会有两个对应元素；`qvel` 里也会有两个对应角速度。

## 5. end_effector 的位置为什么可以从 data.xpos 读取

MuJoCo 在每一步仿真后，都会根据当前 `qpos` / `qvel` 计算各个 body 的世界坐标位置和姿态。

`data.xpos[body_id]` 就是某个 body 在世界坐标系下的位置。

因此只要找到 `end_effector` 的 body id，就能直接读出它当前的 `x y z` 位置。
