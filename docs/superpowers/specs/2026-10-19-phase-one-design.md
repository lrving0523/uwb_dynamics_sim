# 阶段一：六自由度动力学与传感器仿真设计

## 目标与范围

实现用户任务书阶段一：从控制输入积分得到四旋翼真实运动，输出带准确时间戳的 IMU、原始 UWB 测距与 RPM 数据，附配置、测试、运行结果和图形。现有 Python 文件为空，已有配置数值必须保留并明确单位。本轮不实现估计器、学习网络、因子图或鲁棒算法。

## 坐标与物理约定

- 世界系为 ENU，机体系为 FLU，重力为 `[0,0,-g]`。
- 四元数为 Hamilton `wxyz`，表示机体系到世界系；角速度使用机体系 rad/s。
- 电机接口使用 RPM；已有 `kf=1e-5` 与 `km=2e-7` 解释为 N/(rad/s)² 和 N·m/(rad/s)²，先换算角速度后计算平方律。
- 十字构型四个电机依次位于前、左、后、右，偏航反力矩符号为 `+,-,+,-`。
- 气动力在机体系按相对气流的线性及二次阻力计算；未知外力在世界系独立生成并保存，二者不得混合。
- 刚体使用 RK4 积分并归一化四元数；电机一阶滞后在 RK4 各子时刻采用解析响应，控制命令在控制周期内保持不变。

## 模块及接口

- `simulator/math3d.py`：四元数与旋转矩阵。
- `simulator/motor_model.py`：`MotorModel(config)`、`wrench(rpm)`、`allocate(thrust, torque)`、`response(initial, target, dt)`。
- `simulator/drone.py`：`DroneState(position, velocity, quaternion, omega)`；`Drone(drone_config,motor_config,aero_config)`；`step(target_rpm,dt,external_force,time)`；`quantities(external_force,time)` 返回当前加速度和分解力。
- `simulator/controller.py`：位置 PD/积分控制、几何姿态及角速度控制，输出目标 RPM。
- `simulator/trajectory.py`：悬停、圆形、八字、变速、加减速参考，返回位置、速度、加速度与偏航。
- `simulator/wind_field.py`：恒定、阶跃、正弦、带固定节点的随机外力，查询不改变随机状态。
- 传感器独立构造、独立 RNG：IMU `sample(time,quaternion,acceleration,omega,gravity)`；UWB `sample(time,position,quaternion)`；RPM `sample(time,true_rpm)`。所有测量返回具名字典。
- `simulator/config.py`：安全加载与数值/形状/可执行性校验，兼容已有平铺传感器键。
- `simulator/runner.py`：时间事件调度；以各事件精确时刻推进动力学；保存动力学采样及各传感器自己的时间轴。
- `simulator/dataset.py`：独立真值与测量 NPZ、可读 CSV、完整解析配置、运行清单及 SHA256。
- `evaluation/plots.py`：从保存数据绘制中文说明及带 SI 单位的轨迹、IMU、UWB、RPM 和外力图，导出 PNG/PDF/SVG。

## 时间、随机性和数据

动力学最大步长由 `simulation.dt` 管理，控制和传感器可以采用任意有限正频率。采样时刻由整数序号除以频率生成，包含 t=0，只有恰好落在有效区间内的样本。所有事件按时间排序，观测只使用当前时刻真值，无插值未来状态。使用 SeedSequence 派生独立传感器与扰动 RNG。固定种子应逐数组复现；不把耗时或创建时间当作可复现数值结果。

保存 `truth.npz`、`imu.npz`、`uwb.npz`、`rpm.npz` 和 CSV；真值内含状态、参考、加速度、目标/真实 RPM、推力、气动力和外力。真值额外记录所有事件时刻及终止状态，因此时间轴可为非均匀，参考跟踪误差按时间积分加权。IMU 偏置真值仅存在真值文件，测量包不携带理想量。UWB 只输出带噪距离和基站顺序，不生成位置解算。

## 验收

单元测试覆盖悬停平衡、电机反分配和饱和/滞后、自由落体、恒角速度旋转、四元数合法性、IMU 静止/自由落体/旋转比力、偏置随机游走、含天线杆臂的 UWB 距离、RPM 噪声、任意频率时间轴、种子复现和配置错误。运行默认配置及五种轨迹、四种外扰的短实验，生成可重读数据和五组图；记录真实跟踪误差和约束，不声称已达到特定期刊的发表标准。

## 参考

- PX4 控制分配及串级控制资料：https://docs.px4.io/main/en/concept/control_allocation 与 https://docs.px4.io/main/en/flight_stack/controller_diagrams 。本实现是离线研究模型，不复用 PX4 坐标约定或固件代码。
- 四元数约定核对：Joan Solà, Quaternion kinematics for the error-state Kalman filter，https://arxiv.org/abs/1711.02508 。本轮仅使用刚体旋转运动学。
