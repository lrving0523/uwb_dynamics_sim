# 阶段三：物理与学习残余动力学增强

## 目标和边界

用户继续推进，阶段二已完成 209 项测试与真实运行。本阶段实现名义物理模型、TCN 气动残余力训练/独立测试/加载/在线推理及 Physics-EKF、Hybrid-EKF。所有说明与注释中文。本阶段不增广真实未知外扰状态，不开发因子图、鲁棒自适应或 NLOS 算法。

## 固定物理与接口

沿用 ENU/FLU、Hamilton wxyz 与右局部误差。状态仍为 p、v、q、ba、bg（15 维误差）。配置中真机与名义模型参数独立：名义默认质量 1.5 kg、kf=1e-5 N/(rad/s)²、线性/二次气动阻力为零，以研究未建模阻力。允许配置名义阻力和质量；不从真值读取这些参数。

`models/physics.py` 定义 `PhysicsModel(config)`，属性 mass、gravity；`force_body(quaternion,velocity,rpm)` 输出名义机体系力，`acceleration(...)` 输出世界加速度，`force_jacobians(...)` 返回对世界速度、右局部姿态和 RPM 的三组雅可比。力为 RPM 推力减去名义相对气流阻力；RPM 先换算为 rad/s。模型仅接收估计状态和测量 RPM。

`models/residual.py` 定义真实 PyTorch 因果 TCN，输入为最近 10 个 100 Hz 样本（每样本三轴测量陀螺+四 RPM，共 7 维），输出机体系三维气动残余力。提供批量训练、checkpoint 保存/加载、单窗口 predict、推理计时。配置网络宽度/卷积核/层数/优化器/种子；归一化仅在训练集拟合，验证集用于选 checkpoint，测试集仅用于最终误差报告。整次飞行按不同种子/轨迹划分，不随机拆重叠窗口。

`models/hybrid.py` 定义 `HybridModel(physics,residual_predictor)`，与 PhysicsModel 同 force_body/acceleration/force_jacobians 接口。`update_residual(window)` 仅根据测量历史更新持有的残余预测；满窗口前为零。物理+残余力输出满足两者求和，状态雅可比把持有预测视作给定条件输入。未训练模型与理想解析补偿不能进入正式 Hybrid 对比。

## 学习目标与证据

训练目标是离线真值机体系气动力减去名义气动项，明确不含世界系未知外力，也不把总加速度残差当成气动标签。训练/验证/测试飞行使用不同种子且保留其配置、观测、真值及来源清单。输入构造在整数 100 Hz 时刻取最近已到达 gyro/RPM，不读未来观测；标签真值仅在离线训练和评价使用。默认训练数据包含不同周期/幅度/方向的多条轨迹，留出独立测试。有限 0.1 秒 gyro/RPM 历史未必能唯一决定气动力，报告真实预测误差与模型限制，不保证网络改善全部场景。

正式实验保存测试残余力 RMSE、物理/混合加速度预测误差、短时速度/位置开环预测误差，以及真实训练曲线和独立测试图。模型 checkpoint 保存网络参数、训练集归一化、数据划分标识和训练种子；推理耗时单独保存。

## 动力学滤波结构及相关性

Physics-EKF、Hybrid-EKF 均以 RPM 名义/混合力驱动 p/v 预测，用已收到的陀螺传播姿态；加速度计比力仅作观测更新，不再同时用于运动传播。测距使用阶段二相同原始距离模型、初始化、先验、噪声和评价口径。

动力学力误差同时进入状态预测与 IMU 比力观测，不能独立注入两份不相关误差。内部短期辅助变量为当前陀螺样本噪声三维及当前 RPM 周期的模型力误差三维；保留均值、协方差与公开状态交叉项。测距或比力观测按完整联合 Gaussian 条件化；新陀螺/新 RPM 分别替换对应辅助量并保留另一辅助量的边缘分布。模型力误差不是持久随机游走未知外力状态。

模型力先验采用固定 `model_force_std`，加上测量 RPM 噪声经名义推力雅可比映射的协方差；两种增强滤波使用同一数值。学习误差作为给定测量历史下的固定条件误差近似处理，不宣称精确表达任意 TCN 历史输入噪声与状态的全部相关性。该建模假设必须写入方法和结果文档，不以结果残差自适应调权。

比力观测为 `f_nominal^B/m + ba + 当前模型力误差/m + n_acc`，其协方差仅注入独立的加速度计白噪声。陀螺噪声在同一样本分段传播时保留相关性。使用完整联合 Joseph 更新与右雅可比重置；公开状态/保存协方差仍为 15 维。误差状态与预测/测量雅可比需有限差分验证。

## 文件边界与接口

- 模型与训练：`models/physics.py`、`residual.py`、`hybrid.py`，`experiments/train_residual.py`，`tests/test_models.py`、`test_residual.py`。
- 滤波核心：`estimator/physics_ekf.py` 定义 `PhysicsEKF`；`estimator/hybrid_ekf.py` 定义 `HybridEKF`。构造 `(state,covariance,model,noise,initial_time=0.0)`；noise 包含 acc_std、gyro_std、acc_bias_rw_std、gyro_bias_rw_std、rpm_std、model_force_std。公开 state/P/time，`set_gyro(time,gyro)`、`set_rpm(time,rpm)`、`propagate_to(time)`、`update_accel(accel)`、`update_uwb(ranges,anchors,lever_arm,std)`。各 set 仅在当前时刻，新样本时间严格递增。
- 集成：`estimator/dynamics_pipeline.py`，`run_dynamics_estimators(measurements,settings,physics_model,hybrid_model)`，仅接受 imu/uwb/rpm。复用阶段二首帧 LS/杆臂初始协方差，100 Hz 因果 TCN 历史，输出 physics、hybrid 标准状态字典及各自 uwb_diagnostics、accel_diagnostics、timing，不访问真值。
- 配置：`config/hybrid.yaml` 保存名义物理、固定力误差、网络/训练与数据划分设置。
- 实验与图：`experiments/run_stage3.py`，`evaluation/dynamics_plots.py`；保留阶段二基线，所有方法在相同观测、同一 UWB 天线时间轴公平比较。推理只使用已训练 checkpoint，拒绝测试数据与训练/验证飞行标识重叠。

## 验收

保留全部 209 项既有测试；新增推力/阻力/雅可比、TCN 因果卷积及 save/load、训练和独立测试划分、输出不含未知外力标签、模型力共享相关性解析例、公开15维协方差PSD、无未来输入/前缀一致/真值隔离测试。实际训练并保存 checkpoint，实际运行 Physics/Hybrid 与既有 LS/ESKF，生成中文 PNG/PDF/SVG。结果无论优劣均报告，不做种子/轨迹挑选或理想残余替代训练网络。
