"""在测试收集导入 NumPy/Torch 前选择兼容的 MKL 线程层。"""

import os


# 不启用忽略重复 OpenMP 运行库错误的选项，也不改系统环境配置。
os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')
