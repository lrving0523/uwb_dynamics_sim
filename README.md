# 无人机 UWB–IMU 动力学仿真

已实现阶段一仿真、阶段二基础定位及阶段三动力学增强：四旋翼六自由度刚体、多频 IMU/UWB/RPM、原始距离单点最小二乘、15 维 UWB–IMU ESKF、Physics-EKF，以及实际训练的因果 TCN 气动残余与 Hybrid-EKF。`factors/`、`robust/` 保持后续阶段预留状态。

## 运行

在项目根目录使用 Python 3.12：

```powershell
python -m pip install -r requirements.txt
python main.py
python -m pytest -q --basetemp=results/pytest_tmp
python -m experiments.validate_stage1 --duration 8
```

已有依赖时直接运行即可。依赖版本固定；本地验证使用 Python 3.12.7、NumPy 1.26.4、PyYAML 6.0.1、Matplotlib 3.9.2、pytest 7.4.4。测试临时目录放在可写的 `results/` 中。

```powershell
python main.py --trajectory hover --disturbance none --output results/hover
python main.py --trajectory circle --disturbance step --duration 12 --seed 42 --output results/circle_step
python main.py --trajectory figure_eight --disturbance sine --output results/eight_sine
python main.py --trajectory variable_speed --disturbance random --no-plots --output results/random
python main.py --config results/phase1/config.resolved.yaml --output results/replay
```

默认保留原配置的 12 秒、种子 42、1.5 kg 无人机与八个基站，参考为八字。默认 `wind.enabled: false`，可通过 `--disturbance` 显式启用。批量验证包括五类轨迹及恒定、阶跃、正弦、随机外扰共九个实验；每个实验保存数据和五组 PNG 图，汇总写入 `results/stage1_validation/validation_report.json`。阶段一跟踪误差是实际飞行与参考的差值；定位结果由阶段二、三的独立实验入口生成。

## 物理与坐标约定

- 世界系为 ENU，z 向上，重力 `[0,0,-9.81] m/s²`；机体系为 FLU，推力沿机体 +z。
- 姿态为 Hamilton 四元数 `wxyz`，旋转方向从机体到世界；角速度在机体系，单位 rad/s。
- 十字构型电机顺序为前、左、后、右，位置为 `(L,0)`、`(0,L)`、`(-L,0)`、`(0,-L)`，偏航反力矩符号依次为 `+,-,+,-`。
- 电机输入/保存单位为 RPM，计算时转换为 `ω=rpm·2π/60`，单桨推力 `kf·ω²`，偏航反力矩 `±km·ω²`。原 `kf=1e-5`、`km=2e-7` 数值保留，单位为 N/(rad/s)²、N·m/(rad/s)²。
- 电机是一阶滞后，带上下限；每个 RK4 子时刻采用解析 RPM 响应，六自由度状态用 RK4 更新并归一化四元数。
- 机体系气动阻力为 `-c₁⊙v_rel-c₂⊙|v_rel|⊙v_rel`，相对气流由世界速度减去空气速度再旋转得到。`aerodynamics.wind_velocity` 是世界系空气速度；`wind` 节生成独立世界系未知外力。两项分别保存。
- 飞控使用当前真实状态反馈，以位置反馈、加速度前馈及几何姿态/角速度反馈产生目标转速。真实运动始终经电机和刚体积分产生。后续估计器只能读取独立测量包。

控制设计参考 [PX4 控制分配](https://docs.px4.io/main/en/concept/control_allocation) 和 [控制框图](https://docs.px4.io/main/en/flight_stack/controller_diagrams)；四元数运动学参考 [Solà 的文献](https://arxiv.org/abs/1711.02508)。本实现采用上面明确声明的坐标系。

## 配置与时间管理

参数通过 `config/sim.yaml` 管理，`simulator/config.py` 提供缺省值与校验。原平铺传感器键继续可用，也可以采用嵌套 `sensors.imu`、`sensors.rpm`、`sensors.uwb` 配置；同时存在时嵌套键优先。保存的解析配置包含实际使用的全部参数。

| 参数 | 含义 |
| --- | --- |
| `simulation.dt` | 最大动力学积分步长，秒 |
| `simulation.controller_hz` | 控制命令更新频率 |
| `sensors.imu_hz` / `rpm_hz` / `uwb_hz` | 独立传感器采样频率 |
| `imu_acc_std` / `imu_gyro_std` | 单次采样白噪声标准差，m/s² 和 rad/s |
| `imu_acc_bias` / `imu_gyro_bias` | 初始固定三轴偏置 |
| `imu_acc_bias_rw_std` / `imu_gyro_bias_rw_std` | 偏置随机游走强度，每 sqrt(s) 增量标准差 |
| `uwb.lever_arm` | 机体系天线偏移，米 |
| `wind.start_time` | 阶跃/正弦/随机外力起始时刻 |
| `wind.random_hz` | 随机外力节点频率，与传感器查询解耦 |

频率不必成整数比。传感器严格在 `t_k=k/f` 采样，包含 t=0，只保留仿真区间内的样本，不人为增加终点观测。动力学在控制、观测及外扰边界的实际时刻停止并生成当前观测；控制命令在周期之间保持。真值记录最大步长网格、所有事件与终止状态，因此时间轴可能非均匀，误差统计采用时间积分权重。外扰跳变时积分取左极限，观测取右连续值。

IMU 输出 `Rᵀ(a-g)+b_a+n_a` 和 `ω+b_g+n_g`；偏置从第二个实际样本开始按 `σ_rw·sqrt(Δt)` 演化。静止水平机体的 z 比力为 +g，自由落体比力为零。UWB 直接输出 `||p+R·l-a_i||+n_i`，不输出位置解算。RPM 噪声加在真实转速上，测量不裁剪，以保留高斯分布。

主种子经 `SeedSequence.spawn` 生成外扰、IMU、RPM、UWB 的独立随机数流，改变一种传感器噪声不改变其他流。随机外力由预生成的零均值高斯节点线性插值得到，查询不消耗随机数；它是时间相关外力，节点间并非白噪声。

## 数据与图

`results/phase1/` 默认包含：

| 文件 | 内容 |
| --- | --- |
| `truth.npz` / `truth.csv` | 时间、p/v/q/ω、世界加速度、参考、目标/真实 RPM、机体推力/力矩、机体气动力、世界外力、IMU 偏置真值 |
| `imu.npz` / `imu.csv` | 自有时间轴、机体比力与角速度观测 |
| `uwb.npz` / `uwb.csv` | 自有时间轴、原始距离；NPZ 另含基站与杆臂 |
| `rpm.npz` / `rpm.csv` | 自有时间轴、四电机带噪 RPM |
| `config.resolved.yaml` | 实际完整配置和种子，可直接重跑 |
| `manifest.json` | 字段形状、单位、坐标、依赖版本与数据/配置 SHA256 |
| `summary.json` | 样本数、跟踪误差、四元数误差和计算耗时 |
| `physical_validation.json` | 力平衡、旋转合法性与各通道测量噪声的实测统计 |
| `figures/` | 轨迹、IMU、UWB、三类 RPM、分解外力图；默认 PNG/PDF/SVG |

NPZ 使用无 pickle 数值数组，保存全部双精度；CSV 多轴字段以 `_0`、`_1` 等编号展开。基站静态几何在 UWB NPZ 和解析配置中保存。图使用原始样本，不平滑；RPM 分别标记目标、真实与测量，外力图明确坐标系。实验输出默认被 Git 忽略，避免大数据进入代码历史。

```python
from simulator.dataset import load_dataset

data = load_dataset('results/phase1')
imu_time = data['imu']['time']
ranges = data['uwb']['ranges']
```

## 验证与限制

测试覆盖悬停、电机分配/饱和/响应、自由落体、旋转积分、合法四元数、参考解析导数、各类外扰、IMU 比力/随机游走、旋转杆臂测距、噪声统计、非整数比时钟、离网格外扰边界、复现、保存重载和无噪声传感器逐时刻物理一致性。实际测试数和运行结果见 `docs/stage1-results.md`。

这是离线研究用的最小物理一致仿真。气动力是解析阻力，未包含桨叶挥舞、地效、旋翼相互干扰、传感器延时/丢包/带宽或地面接触。初始电机默认处于悬停转速；从零转速启动允许下落，但不模拟碰撞。控制器未建模板载状态估计噪声；极端参考、巨大扰动或推力饱和时不保证稳定。改变事件频率会改变 RK4 子步划分，真值可能出现积分误差量级的差异。

图形为通用科研输出，未针对特定期刊版式验收；程序可复现不等于已完成论文级实机模型验证。

## 阶段二：基础滤波定位

先运行阶段一生成默认数据，再读取同一份测距和 IMU：

```powershell
python main.py
python -m experiments.run_stage2 --dataset results/phase1 --output results/phase2
python -m pytest -q --basetemp=results/pytest_tmp
```

多案例验证复用阶段一批量实验数据，全部使用同一套估计先验：

```powershell
python -m experiments.validate_stage1 --duration 8
python -m experiments.validate_stage2
```

估计设置位于 `config/estimation.yaml`。初始姿态、速度、加速度计/陀螺偏置及标准差都是独立先验；没有从 `drone.initial_*` 或真实传感器偏置读取。测量噪声只提取保存配置中的标定标准差，零测距噪声实验采用显式固定标准差下限。默认姿态先验是水平、航向零，描述一个已知起飞朝向的实验条件；不承诺未知初始航向、任意先验误差都能收敛。

UWB LS 仅解算单时刻天线位置，基站至少四个且非共面，测距雅可比病态/天线恰好重合基站时明确报错。输出局部协方差采用固定 `σ²(JᵀJ)⁻¹`；不根据残差调整权重。它不输出机体姿态、速度或虚构的机体位置。

ESKF 的 15 维误差顺序为 `[δp,δv,δθ,δba,δbg]`，采用右乘旋转误差，名义四元数以指数映射更新，中点姿态用于比力积分。首帧距离 LS 初始化位置及协方差，减去姿态先验下的杆臂，并保留位置/姿态初始交叉协方差；这帧不再重复滤波更新。后续直接融合原始距离，使用固定协方差、Joseph 更新和右雅可比误差重置。

传播始终使用已经收到的 IMU 样本，同一时间先传播过去区间并更新 UWB，再装载用于未来区间的新 IMU。一个样本跨多个测距时刻时，内部保留当前六维输入噪声的短期条件均值、协方差和交叉项，防止把同一样本反复当成独立噪声；公开运动误差状态仍为 15 维。偏置随机游走以正权求积近似连续过程噪声积分，转动时属于数值近似。

公平位置比较使用全部相同 UWB 时刻的天线位置。ESKF 另报告机体位置、速度、姿态和偏置误差；纯 LS 不借助真实姿态消除杆臂。APE 使用位置误差范数，平移 RPE 明确定义为世界系 1 秒位移差；不把它混称为完整 SE(3) 相对姿态误差。单次 NIS/NEES 只报告描述统计，尚不作蒙特卡洛一致性结论。

`results/phase2/` 包含 `uwb_ls.npz`、`eskf.npz`、`diagnostics.npz`、`timing.npz`、解析估计设置、输入文件校验和、指标 JSON 和图。运行耗时独立存储，不要求与科学估计数组逐元素复现。具体实测结果和限制见 `docs/stage2-results.md`。

## 阶段三：物理与混合动力学增强

学习功能使用可选的 CPU PyTorch 环境。已有依赖和模型时可以直接运行定位对比：

```powershell
python -m pip install -r requirements-learning.txt
$env:MKL_THREADING_LAYER='SEQUENTIAL'
python main.py
python -m experiments.train_residual --config config/hybrid.yaml --output results/residual_training
python -m experiments.plot_training --training results/residual_training
python -m experiments.run_stage3 --dataset results/phase1 --output results/phase3
python -m experiments.validate_stage3
python -m pytest -q --basetemp=results/pytest_tmp
```

训练命令会生成 18 条完整飞行数据、训练 60 个 epoch，并只按验证集选择 checkpoint。默认训练/验证/测试为 12/3/3 条飞行，种子分别为 100–111、200–202、300–302。归一化只拟合训练集；训练标签是机体系气动力减名义气动力，不含独立未知外力。绘图入口可以从已保存结果重建训练曲线和三条测试飞行的残余力图，无需重训。

`config/hybrid.yaml` 明确给出独立名义物理参数、固定模型力误差标准差及网络/训练设置。默认名义阻力为零，真实仿真含阻力。TCN 输入是最近 10 个原始 100 Hz RPM 时刻的三轴测量陀螺与四 RPM，输出三轴机体系残余力；窗口未满时残余为零。目前混合入口要求原始 RPM 为 100 Hz，陀螺和 UWB 可异步，不把重复保持值冒充新 RPM 样本。正式入口会拒绝未训练模型、训练/验证飞行复用及不兼容的名义气动标签定义。

两种动力学滤波器都用 RPM 模型预测位置和速度、陀螺传播姿态，加速度计只作比力更新。内部保留当前陀螺噪声和模型力误差的联合条件分布，同一模型力误差同时进入预测和比力观测；公开误差状态仍为 15 维。TCN 全部历史测量噪声与状态的相关性采用固定条件误差近似，尚未完整增广。

默认 12 秒场景中，同一 121 个 UWB 时刻的天线位置 RMSE 为 LS 0.16900 m、ESKF 0.05960 m、Physics-EKF 0.05348 m、Hybrid-EKF 0.05231 m。三条独立测试飞行全部保留；变速案例中 Hybrid-EKF 略差于 Physics-EKF。单次结果不支持普遍优越性或统计一致性结论。

`results/residual_training/` 保存 checkpoint、归一化、飞行划分和指纹、原始训练源码快照、训练历史及独立测试预测；`results/phase3/` 保存四种方法的估计、两类创新、输入/模型校验和、指标及图。`results/stage3_validation/validation_report.json` 汇总全部独立测试飞行。完整方法、实测结果和近似边界见 [方法说明](docs/methods-dynamics.md) 与 [阶段三报告](docs/stage3-results.md)。

本机使用 PyTorch 2.7.1+cpu、单线程推理。Conda MKL 与 Torch 的 OpenMP DLL 冲突通过导入数值库前设置 `MKL_THREADING_LAYER=SEQUENTIAL` 处理；学习入口及测试收集已设置该缺省值。进程已经导入数值库时，应在新进程中运行以上命令。

## 下一阶段

阶段四计划增广世界系未知外扰状态，检查外力、加速度计偏置与模型残余的可辨识条件，再开展模型失配和蒙特卡洛实验。当前短期模型力误差辅助量不作为未知外力估计输出。
