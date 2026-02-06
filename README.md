Paper-MOBO: Batch Multi-Objective Bayesian Optimization Engine
1. 项目简介 (Overview)

Paper-MOBO 是一个轻量级、可复现的 批次多目标贝叶斯优化（Batch MOBO） 引擎，用于自动搜索设计变量空间，并通过用户定义的 objective_function() 评估目标函数，实现复杂系统（模拟 / 计算 / 模型）的自动优化。

该系统专为 单机多 GPU、长时间计算任务、论文级复现 场景设计，支持断点续跑与批次并行优化。

典型应用：

材料 / 分子模拟参数优化

机器学习模型超参数优化

物理 / 工程模拟黑箱优化

多目标设计空间搜索

2. 主要特性 (Key Features)

支持任意维度设计变量

支持任意数量目标函数（多目标优化）

基于 qEHVI 的批次多目标贝叶斯优化

单机多 GPU 并行执行目标函数

通用 objective_function() 抽象接口

自动断点续跑（resume）

CSV 数据库记录全部实验过程

适用于长时间运行任务（小时级模拟）

完全由配置文件驱动，无需修改源码

3. 项目结构 (Project Structure)
paper-mobo/
│
├── configs/
│   └── exp.yaml              # 实验配置文件
│
├── src/
│   ├── mobo.py               # BO核心（qEHVI）
│   ├── runner.py             # 多GPU并行执行
│   ├── pipeline.py           # 优化主流程
│   ├── config.py             # 读取配置
│   └── schema.py             # 变量/目标解析
│
├── user/
│   └── objective.py          # 用户定义目标函数
│
├── data/
│   └── input_output.csv      # 优化数据库（自动生成）
│
└── README.md

4. 快速开始 (Quick Start)
4.1 安装依赖
pip install -r requirements.txt


（需安装 PyTorch + BoTorch + gpytorch）

4.2 定义目标函数

编辑：

user/objective.py


示例：

def objective_function(x: dict, gpu_id=None, device=None) -> dict:
    """
    输入: 设计变量
    输出: 目标函数字典
    """

    # 示例（替换为真实模拟）
    f1 = -(x["x1"] - 0.3)**2
    f2 = -(x["x2"] - 0.7)**2

    return {
        "obj1": f1,
        "obj2": f2,
    }

4.3 运行优化
python -m src.pipeline --config configs/exp.yaml


程序将：

初始化数据库

生成初始样本

执行 BO 优化循环

多 GPU 并行运行目标函数

自动写入数据库

支持中断后继续运行

4.4 中断后继续

重新运行同一命令即可：

python -m src.pipeline --config configs/exp.yaml


系统会自动从数据库恢复。

5. 配置文件说明 (Configuration)

文件：

configs/exp.yaml

5.1 设计变量
design_variables:
  names: [x1, x2]
  bounds:
    lower: [0, 0]
    upper: [1, 1]


定义优化变量及范围。

5.2 目标函数
objectives:
  names: [obj1, obj2]


必须与 objective_function() 返回键一致。

5.3 贝叶斯优化参数
bo:
  q_batch: 2
  num_initial_samples: 6
  max_iterations: 100
  ref_margin: 0.05
  num_restarts: 10
  raw_samples: 128
  mc_samples: 128

5.4 GPU 设置
hardware:
  gpu_ids: [0, 1]
  mp_start_method: spawn

5.5 目标函数入口
objective:
  callable: "user.objective:objective_function"
  parallel: true
  timeout_seconds: 0
  max_retries: 1

6. 优化流程说明 (Optimization Workflow)

程序执行流程如下：

初始化数据库
自动创建 data/input_output.csv

初始采样
使用 LHS 或随机采样生成初始点

构建多目标 GP 模型

批次 BO 采样（qEHVI）
提议新设计点

多 GPU 并行执行目标函数

写入数据库
记录状态、结果、运行时间

断点续跑
未完成样本自动继续执行
