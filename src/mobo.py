"""
mobo.py
----------------------------------
Batch Multi-Objective Bayesian Optimization (qEHVI)

封装内容：
1) GP 模型构建
2) qEHVI acquisition
3) 批次样本提议

与原始 batchMOBO.py 逻辑等价，但模块化。
"""

from typing import List, Dict

import torch
import pandas as pd

from botorch.models import SingleTaskGP, ModelListGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from botorch.acquisition.multi_objective.logei import qLogExpectedHypervolumeImprovement
from botorch.utils.multi_objective.box_decompositions import NondominatedPartitioning
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.optim import optimize_acqf


# =====================================================
# GP MODEL
# =====================================================

def fit_mobo_model(
    df_train: pd.DataFrame,
    x_columns: List[str],
    y_columns: List[str],
) -> ModelListGP:
    """
    构建多目标 GP 模型（ModelListGP）

    Parameters
    ----------
    df_train : DataFrame
        已完成样本数据
    x_columns : list[str]
        设计变量列名
    y_columns : list[str]
        目标函数列名

    Returns
    -------
    ModelListGP
    """

    train_X = torch.tensor(
        df_train[x_columns].values,
        dtype=torch.double
    )

    train_Y = torch.tensor(
        df_train[y_columns].values,
        dtype=torch.double
    )

    # 每个目标一个 GP
    models = []
    for i in range(len(y_columns)):
        models.append(SingleTaskGP(train_X, train_Y[:, [i]]))

    model = ModelListGP(*models)

    mll = SumMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)

    model.eval()
    return model


# =====================================================
# qEHVI BATCH PROPOSAL
# =====================================================

def propose_batch_qehvi(
    df_train: pd.DataFrame,
    x_columns: List[str],
    y_columns: List[str],
    bounds: torch.Tensor,
    q: int,
    ref_margin: float = 0.05,
    num_restarts: int = 10,
    raw_samples: int = 128,
    mc_samples: int = 128,
) -> List[Dict]:
    """
    批次多目标贝叶斯优化（qEHVI）

    返回 q 个新样本

    Parameters
    ----------
    df_train : DataFrame
        已完成样本
    x_columns : 设计变量
    y_columns : 目标函数
    bounds : torch.Tensor shape(2, d)
        BO 搜索范围
    q : int
        batch size
    ref_margin : float
        hypervolume 参考点 margin
    """

    # ---------- 拟合 GP ----------
    model = fit_mobo_model(df_train, x_columns, y_columns)

    train_Y = torch.tensor(
        df_train[y_columns].values,
        dtype=torch.double
    )

    # ---------- 参考点 ----------
    y_min = train_Y.min(dim=0).values
    ref_point = (y_min - ref_margin * torch.abs(y_min)).tolist()

    partitioning = NondominatedPartitioning(
        ref_point=torch.tensor(ref_point, dtype=torch.double),
        Y=train_Y,
    )

    # ---------- MC sampler ----------
    sampler = SobolQMCNormalSampler(
        sample_shape=torch.Size([mc_samples])
    )

    # ---------- qEHVI ----------
    acqf = qLogExpectedHypervolumeImprovement(
        model=model,
        ref_point=ref_point,
        partitioning=partitioning,
        sampler=sampler,
    )

    # ---------- optimize acquisition ----------
    candidate, _ = optimize_acqf(
        acq_function=acqf,
        bounds=bounds,
        q=q,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
    )

    X_new = candidate.detach().cpu().numpy()

    sample_list = []
    for k in range(q):
        sample = {}
        for i, col in enumerate(x_columns):
            sample[col] = float(X_new[k, i])
        sample_list.append(sample)

    return sample_list
