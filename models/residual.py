"""真实 PyTorch 因果 TCN、仅训练集归一化与安全检查点。"""

from copy import deepcopy
from pathlib import Path
import random
from time import perf_counter

import numpy as np
import torch
from torch import nn
from torch.nn import functional as functional


class _CausalBlock(nn.Module):
    """左补零保证每个时刻只依赖当前及过去输入。"""

    def __init__(self, input_channels, output_channels, kernel_size, dilation):
        super().__init__()
        self.left_padding = (kernel_size-1)*dilation
        self.convolution = nn.Conv1d(input_channels, output_channels, kernel_size, dilation=dilation)
        self.skip = nn.Conv1d(input_channels, output_channels, 1) if input_channels != output_channels else nn.Identity()

    def forward(self, values):
        return functional.silu(self.convolution(functional.pad(values, (self.left_padding, 0)))+self.skip(values))


class ResidualTCN(nn.Module):
    """输入 B×T×7 陀螺/RPM 序列，输出 B×3 机体系气动残余力。"""

    def __init__(self, channels=32, layers=3, kernel_size=3):
        super().__init__()
        if any(isinstance(v, bool) or int(v) != v or v <= 0 for v in [channels, layers, kernel_size]):
            raise ValueError('TCN 通道数、层数和卷积核必须为正整数')
        self.architecture = {'channels': int(channels), 'layers': int(layers), 'kernel_size': int(kernel_size)}
        self.blocks = nn.Sequential(*[_CausalBlock(7 if i == 0 else channels, channels, kernel_size, 2**i) for i in range(layers)])
        self.output = nn.Conv1d(channels, 3, 1)

    def forward_sequence(self, values):
        """返回 B×T×3 全时刻输出，便于独立验证前缀因果性。"""
        if values.ndim != 3 or values.shape[2] != 7 or values.shape[1] < 1:
            raise ValueError('TCN 输入必须为 B×T×7')
        return self.output(self.blocks(values.transpose(1, 2))).transpose(1, 2)

    def forward(self, values):
        return self.forward_sequence(values)[:, -1]


class ResidualPredictor:
    """保存训练集统计量，并以 N 为单位提供固定 0.1 秒窗口推理。"""

    def __init__(self, network, input_mean, input_std, target_mean, target_std, metadata=None):
        self.network = network.cpu().eval()
        self.sequence_length = 10
        self.input_hz = 100.0
        self.input_mean = torch.as_tensor(input_mean, dtype=torch.float32).cpu().reshape(7)
        self.input_std = torch.as_tensor(input_std, dtype=torch.float32).cpu().reshape(7)
        self.target_mean = torch.as_tensor(target_mean, dtype=torch.float32).cpu().reshape(3)
        self.target_std = torch.as_tensor(target_std, dtype=torch.float32).cpu().reshape(3)
        for value in [self.input_mean, self.input_std, self.target_mean, self.target_std]:
            if not torch.isfinite(value).all():
                raise ValueError('归一化参数必须有限')
        if torch.any(self.input_std <= 0) or torch.any(self.target_std <= 0):
            raise ValueError('归一化标准差必须为正')
        self.metadata = deepcopy(metadata or {'trained': False})

    def predict_batch(self, windows, batch_size=512):
        """只根据测量窗口推理，不接收状态或真值。"""
        values = torch.as_tensor(np.asarray(windows), dtype=torch.float32)
        if values.ndim != 3 or values.shape[1:] != (10, 7) or not torch.isfinite(values).all():
            raise ValueError('预测输入必须为有限 B×10×7 窗口')
        if len(values) == 0:
            return np.empty((0, 3))
        outputs = []
        with torch.inference_mode():
            for chunk in values.split(batch_size):
                normalized = (chunk-self.input_mean)/self.input_std
                outputs.append(self.network(normalized)*self.target_std+self.target_mean)
        return torch.cat(outputs).numpy().astype(float)

    def predict(self, window):
        """返回三维机体系残余力，单位 N。"""
        values = np.asarray(window)
        if values.shape != (10, 7):
            raise ValueError('单窗口预测输入必须为 10×7')
        return self.predict_batch(values[None])[0]

    def save(self, path):
        """检查点仅含张量和基本类型，可用 weights_only=True 加载。"""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save({'schema_version': 1, 'architecture': self.network.architecture,
                    'sequence_length': self.sequence_length, 'input_hz': self.input_hz,
                    'state_dict': self.network.state_dict(), 'input_mean': self.input_mean,
                    'input_std': self.input_std, 'target_mean': self.target_mean,
                    'target_std': self.target_std, 'metadata': self.metadata}, destination)

    @classmethod
    def load(cls, path):
        """在 CPU 安全加载；正式实验另检查 metadata.trained 与数据来源。"""
        torch.set_num_threads(1)
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        if checkpoint.get('schema_version') != 1 or checkpoint['sequence_length'] != 10 or checkpoint['input_hz'] != 100:
            raise ValueError('不支持的 TCN 检查点结构或输入时钟')
        network = ResidualTCN(**checkpoint['architecture'])
        network.load_state_dict(checkpoint['state_dict'])
        return cls(network, checkpoint['input_mean'], checkpoint['input_std'],
                   checkpoint['target_mean'], checkpoint['target_std'], checkpoint['metadata'])

    def benchmark(self, window, repetitions=200, warmup=20):
        """记录完整单窗口归一化、网络及输出转换的平均/p95 毫秒耗时。"""
        for _ in range(warmup):
            self.predict(window)
        elapsed = []
        for _ in range(repetitions):
            start = perf_counter()
            self.predict(window)
            elapsed.append((perf_counter()-start)*1000)
        return {'mean_ms': float(np.mean(elapsed)), 'p95_ms': float(np.percentile(elapsed, 95)),
                'repetitions': repetitions, 'threads': torch.get_num_threads(), 'device': 'cpu'}


def train_residual_model(train_inputs, train_targets, val_inputs, val_targets, network_config, training_config, metadata):
    """真实小批量 Adam 训练；只按验证误差选模型，接口不接受测试集。"""
    from models.training_data import validate_split

    seed = int(training_config.get('seed', 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(int(training_config.get('threads', 1)))
    torch.use_deterministic_algorithms(True)
    if network_config.get('sequence_length', 10) != 10 or network_config.get('input_hz', 100) != 100:
        raise ValueError('当前网络固定使用 100 Hz、10 个样本')
    validate_split(metadata['data_split'])
    tx, ty, vx, vy = [torch.as_tensor(np.asarray(value), dtype=torch.float32) for value in [train_inputs, train_targets, val_inputs, val_targets]]
    for inputs, targets in [(tx, ty), (vx, vy)]:
        if inputs.ndim != 3 or inputs.shape[1:] != (10, 7) or targets.shape != (len(inputs), 3) or len(inputs) == 0:
            raise ValueError('训练/验证数据需要非空 N×10×7 输入及 N×3 标签')
        if not torch.isfinite(inputs).all() or not torch.isfinite(targets).all():
            raise ValueError('训练/验证数据必须有限')
    input_mean = tx.mean(dim=(0, 1))
    input_std = tx.std(dim=(0, 1), correction=0).clamp_min(1e-5)
    target_mean = ty.mean(dim=0)
    target_std = ty.std(dim=0, correction=0).clamp_min(.01)
    tx, vx = (tx-input_mean)/input_std, (vx-input_mean)/input_std
    ty_normalized = (ty-target_mean)/target_std
    network = ResidualTCN(**{key: network_config[key] for key in ['channels', 'layers', 'kernel_size'] if key in network_config})
    optimizer = torch.optim.Adam(network.parameters(), lr=float(training_config.get('learning_rate', .0015)),
                                 weight_decay=float(training_config.get('weight_decay', 1e-5)))
    epochs, batch_size = int(training_config.get('epochs', 60)), int(training_config.get('batch_size', 256))
    if epochs < 1 or batch_size < 1:
        raise ValueError('训练轮数与批量大小必须为正')
    best_error, best_state, best_epoch, history = float('inf'), None, 0, []
    for epoch in range(1, epochs+1):
        network.train()
        permutation = torch.randperm(len(tx))
        loss_sum, raw_squared_sum = 0., 0.
        for indices in permutation.split(batch_size):
            optimizer.zero_grad(set_to_none=True)
            normalized_error = network(tx[indices])-ty_normalized[indices]
            loss = torch.mean(normalized_error**2)
            raw_squared_sum += float(torch.sum((normalized_error.detach()*target_std)**2))
            loss.backward()
            nn.utils.clip_grad_norm_(network.parameters(), 5.)
            optimizer.step()
            loss_sum += float(loss.detach())*len(indices)
        network.eval()
        with torch.inference_mode():
            squared = sum(float(torch.sum((network(x)*target_std+target_mean-y)**2)) for x, y in zip(vx.split(batch_size), vy.split(batch_size)))
        validation_rmse = float(np.sqrt(squared/len(vx)))
        history.append({'epoch': epoch, 'train_loss': raw_squared_sum/len(tx),
                        'objective_loss': loss_sum/len(tx), 'val_force_rmse_3d_N': validation_rmse})
        if validation_rmse < best_error:
            best_error, best_epoch = validation_rmse, epoch
            best_state = {key: value.detach().clone() for key, value in network.state_dict().items()}
    network.load_state_dict(best_state)
    details = deepcopy(metadata)
    details.update(trained=True, seed=seed, completed_epochs=epochs, best_epoch=best_epoch,
                   best_validation_force_rmse_3d_N=best_error,
                   normalization_source='仅训练飞行窗口', input_features=['gyro_x', 'gyro_y', 'gyro_z', 'rpm_0', 'rpm_1', 'rpm_2', 'rpm_3'])
    return ResidualPredictor(network, input_mean, input_std, target_mean, target_std, details), history
