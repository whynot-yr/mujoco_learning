# v0.4 Actuator Basics

## 1. actuator 和 joint 的区别

- `joint` 定义“这个 body 可以怎么运动”
- `actuator` 定义“你怎样对系统施加主动控制”

`joint` 更像自由度本身，`actuator` 更像作用在自由度上的驱动器。

## 2. motor actuator 是什么

`motor` 是一种最基础的 actuator。

它通常绑定到某个 joint 上，然后把控制输入 `data.ctrl` 转换成该 joint 上的广义力。在 hinge joint 上，可以直观理解成“施加一个关节力矩”。

## 3. data.ctrl 的含义

`data.ctrl` 是当前时刻的控制输入向量。

如果模型里有 1 个 actuator，那么：

- `data.ctrl[0]` 就是这个 actuator 的输入

如果模型里有多个 actuator，那么：

- `data.ctrl[0]`、`data.ctrl[1]`、`data.ctrl[2]` ... 分别对应不同 actuator

## 4. ctrlrange 的作用

`ctrlrange` 用来限制控制输入的取值范围。

比如：

```xml
ctrlrange="-2 2"
```

表示这个 actuator 接受的控制输入范围是 `[-2, 2]`。

这样做的直观作用是：

- 避免输入过大
- 让控制更接近真实执行器的能力限制
- 帮助你更稳定地做实验

## 5. gear 的直观理解

`gear` 可以先直观理解成“控制输入到实际作用力之间的放大倍数”。

- `gear` 大一些：同样的 `ctrl` 会产生更强的效果
- `gear` 小一些：同样的 `ctrl` 会更柔和

在这个实验里设成 `1`，先保持最简单。

## 6. 为什么加 actuator 后才可以主动控制系统

没有 actuator 时，系统只能按照自然动力学运动，比如受重力摆动。

加了 actuator 之后，Python 代码才能通过 `data.ctrl` 向模型注入控制输入，于是你可以主动“推它”“拉它”“让它往目标方向转”。
