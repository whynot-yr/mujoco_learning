# v0.6 Sensor And Logging

## 1. sensor 是什么

`sensor` 是 MuJoCo 里专门用来“读状态或读输出”的元素。

你可以在 MJCF 里声明传感器，让 MuJoCo 在每一步仿真后自动把对应读数写到 `data.sensordata` 里。

## 2. data.sensordata 是什么

`data.sensordata` 是一个连续数组，里面按顺序存放所有传感器的输出值。

每个 sensor 会占用一段位置：

- `model.sensor_adr[sensor_id]`：这个 sensor 在 `sensordata` 里的起始地址
- `model.sensor_dim[sensor_id]`：这个 sensor 输出有几维

## 3. sensor 读数和直接读 qpos/qvel 有什么关系

这两者很多时候是相关的，但用途不同：

- 直接读 `qpos / qvel`：是在读系统内部状态
- 读 `sensor`：是在读“模型定义出来的观测通道”

比如 `jointpos` sensor 读到的值，本质上和对应关节的角度有关；`jointvel` 也是类似。但一旦模型更复杂，你可能还会有接触力、执行器输出、IMU、末端力等观测，它们统一放到 `sensordata` 会更方便。

## 4. 为什么要记录 CSV

因为控制和分析几乎都离不开数据记录。

把仿真过程写成 CSV 之后，你可以：

- 画曲线
- 对比不同参数
- 检查控制器是否稳定
- 做离线分析

## 5. 这和后续训练控制器、RL 或机器人实验有什么关系

无论是传统控制、强化学习，还是现实机器人实验，都会反复遇到三件事：

- 读取观测
- 施加控制
- 记录数据

如果你已经会从 `sensordata` 读信息、会记录 CSV，那么后面做：

- 控制器调参
- 轨迹分析
- RL rollout 记录
- 真机实验数据对比

都会更自然。
