# 阶段二：UWB 单点定位与 UWB–IMU ESKF

## 目标

在阶段一保存的数据上实现 UWB 原始距离单点非线性最小二乘基线和 15 维误差状态滤波器，实际运行、验证与绘图。本轮不实现物理/混合动力学增强、扰动增广、学习模型、因子图或鲁棒自适应算法。全部新增注释和说明为中文。

## 坐标、状态及测量

继续使用世界 ENU、机体 FLU、Hamilton wxyz（机体到世界）。名义状态为 position、velocity、quaternion、acc_bias、gyro_bias，误差顺序为 `[δp,δv,δθ,δba,δbg]`。旋转误差右乘，`q_true=q_nominal⊗Exp(δθ)`。

ESKF 名义传播由当前已收到 IMU 样本的零阶保持驱动，使用中点姿态积分比力，绝不拿后一帧 IMU 回填过去。测距 `h_i=||p+R·l-a_i||`，雅可比位置块为单位视线，姿态块为 `-uᵀR[l]×`。联合距离更新采用固定对角测量协方差、Joseph 协方差更新、状态注入和右雅可比旋转误差重置。

## 异步输入的噪声

阶段一的 IMU 白噪声参数是每个实际样本的标准差。一个样本跨多个 UWB 事件使用时，不能将各段输入误差当作新独立噪声重复注入。滤波器公开状态仍是 15 维，但内部保留当前样本六维输入噪声的条件均值、协方差与状态的交叉协方差；在新样本到达时重置该短期辅助量。测距更新同时条件化该辅助量，后续传播使用其条件均值补偿，保持拆分传播时的相关性。偏置随机游走采用连续时间强度按实际 Δt 积分。

传播雅可比应与名义离散传播一致，使用 SO(3) 指数映射与右雅可比，接受有限差分测试。随机游走过程协方差可用保持半正定的数值求积近似，固定频率下不得把每样本噪声直接当作连续噪声密度。

## 初始化与可访问数据

ESKF 输入接口只含 IMU 与 UWB 测量及静态基站/杆臂。首个 UWB 时刻之前没有估计输出；初始化使用这一帧距离的 LS 天线位置及其局部协方差，减去独立配置的姿态先验下杆臂。首帧距离不再重复测量更新。位置与姿态初始协方差包含杆臂变换造成的交叉项。初始速度、姿态及偏置来自 `estimation` 的独立先验，禁止读取 `drone.initial_*` 或任何运动/偏置真值。初始朝向先验是实验已知条件，不能声称任意未知航向均可被观测。

在相同时间先完成过去区间传播、测距更新，再接收用于未来区间的新 IMU 样本。初始化所需 IMU 必须已经到达；若首个 UWB 早于首个 IMU，明确报错，不能使用未来样本。

## 公平评价

纯 UWB LS 无法独立获知姿态，输出天线位置；ESKF 输出机体状态及其估计天线位置。主要比较在相同 UWB 时间戳上的天线位置 RMSE/APE，不用真实姿态把 LS 天线位置伪装成机体位置。ESKF 另报告机体位置、速度、姿态、偏置误差和诊断 NIS/NEES。所有真值匹配位于独立离线评价层，缺失时刻明确报错或标记，不插值未来状态供估计器使用。单次 NEES/NIS 只作描述，不声称通过蒙特卡洛一致性检验。

## 文件与接口

- `estimator/uwb_ls.py`：`solve_uwb_position(anchors,ranges,std,initial=None,...)` 返回 position、covariance、residual、iterations、converged 等；只解单个时刻，不是滑动窗口优化。
- `estimator/eskf.py`：`NominalState`，`UWBIMUESKF(state,covariance,gravity,noise)`，包含 `time`、`P`、`state`；`set_imu(time,accel,gyro)`、`propagate_to(time)`、`update_uwb(ranges,anchors,lever_arm,std)`；输出创新、协方差与 NIS。
- `estimator/rotation.py`：SO(3) 指数/对数、右雅可比，复用阶段一四元数工具。
- `estimator/pipeline.py`：`run_estimators(measurements,settings)` 返回 `uwb_ls` 与 `eskf` 数组字典，严格不接受真值；每种输出含 time，ESKF 含 position、velocity、quaternion、acc_bias、gyro_bias、covariance、tag_position，LS 含 tag_position、covariance、converged；诊断单独数组带其自己的时间戳。
- `evaluation/metrics.py`：状态误差与轨迹指标，明确坐标、单位和样本时间；不修改估计结果。
- `evaluation/estimation_plots.py`：比较轨迹、相同时间的天线位置误差、ESKF 状态误差与一致性诊断图，保存 PNG/PDF/SVG。
- `experiments/run_stage2.py`：读取已保存数据，独立运行估计、保存结果、参数、指标与图；支持基于阶段一生成的不同轨迹验证。
- `config/estimation.yaml`：独立先验和滤波参数；测量标准差通过保存的阶段一解析配置提取，禁止传入真实偏置或真实初始状态。

## 验收

LS 精确恢复已知点、噪声与非共面几何、无效/退化几何明确错误。ESKF 通过静止/自由落体/旋转传播、离散雅可比与带杆臂测距雅可比有限差分、误差注入重置、Joseph 更新半正定、异步同样本噪声相关性、不同频率与无未来样本测试。完整运行默认数据和至少悬停/动态/外扰案例；验证更改真值不影响估计数组、数据前缀产生相同前缀输出，保存并重载结果。保留阶段一 100 项测试通过。

参考：Solà，Quaternion kinematics for the error-state Kalman filter，https://arxiv.org/abs/1711.02508 ，使用其右局部旋转误差和重置约定；本实现固定重力，误差状态为 15 维。
