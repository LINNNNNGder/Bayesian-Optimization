def objective_function(x: dict, gpu_id: int = None, device: str = None) -> dict:
    """
    x: 设计变量字典
    gpu_id: 原始GPU编号（可选）
    device: 形如 'cuda:0'（子进程已绑定单卡）
    return: 目标字典
    """
    # 这里写你的真实模拟/计算
    # 例如：run_external_simulation(x, device=device)
    return {"strain": 0.0, "elastic_modulus": 0.0}
