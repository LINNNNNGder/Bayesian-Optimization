import os
import time
import traceback
import importlib
import inspect
from typing import List, Dict, Tuple, Optional

import multiprocessing as mp


def _load_callable(callable_path: str):
    """
    callable_path 格式: "module.submodule:function_name"
    """
    if ":" not in callable_path:
        raise ValueError(f"Invalid objective.callable='{callable_path}'. Use 'module:function'.")
    mod_name, fn_name = callable_path.split(":", 1)
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, fn_name)
    if not callable(fn):
        raise TypeError(f"{callable_path} is not callable.")
    return fn


def _call_objective(fn, x: Dict, gpu_id: int):
    """
    兼容不同签名：
      f(x)
      f(x, gpu_id=...)
      f(x, device=...)
    """
    sig = inspect.signature(fn)
    kwargs = {}
    if "gpu_id" in sig.parameters:
        kwargs["gpu_id"] = gpu_id
    if "device" in sig.parameters:
        # 注意：这里传的是“可见设备后的 device=0”
        # 因为子进程里 CUDA_VISIBLE_DEVICES 已经映射到单卡
        kwargs["device"] = "cuda:0"
    return fn(x, **kwargs) if kwargs else fn(x)


def _worker(
    idx: int,
    x: Dict,
    gpu_id: int,
    callable_path: str,
    return_dict,
):
    """
    单任务 worker：绑定 GPU -> 动态 import objective -> 执行
    """
    try:
        # ---- 绑定 GPU（必须在 objective 导入前）----
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        # 可选：让一些库更稳
        os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

        # 动态加载（确保在设置 CUDA_VISIBLE_DEVICES 后 import）
        fn = _load_callable(callable_path)

        t0 = time.time()
        y = _call_objective(fn, x, gpu_id=gpu_id)
        t1 = time.time()

        if not isinstance(y, dict):
            raise TypeError("objective_function must return a dict of objectives.")

        y_out = dict(y)
        y_out["__ok__"] = True
        y_out["__runtime_sec__"] = float(t1 - t0)
        return_dict[idx] = y_out

    except Exception as e:
        tb = traceback.format_exc()
        return_dict[idx] = {
            "__ok__": False,
            "__error__": f"{repr(e)}\n{tb}",
        }


def evaluate_batch_multi_gpu(
    sample_batch: List[Dict],
    gpu_ids: List[int],
    callable_path: str,
    start_method: str = "spawn",
    parallel: bool = True,
    timeout_seconds: int = 0,
) -> List[Dict]:
    """
    批次执行（单机多GPU绑定）
    - sample_batch: list of x dict
    - gpu_ids: e.g. [0,1,2,3]
    - callable_path: "user.objective:objective_function"
    """

    if not parallel:
        # 单进程：使用“第一个gpu_id”映射
        results = []
        for x in sample_batch:
            manager = mp.Manager()
            d = manager.dict()
            _worker(0, x, gpu_ids[0], callable_path, d)
            results.append(d[0])
        return results

    ctx = mp.get_context(start_method)
    manager = ctx.Manager()
    return_dict = manager.dict()
    procs: List[Tuple[int, mp.Process]] = []

    for i, x in enumerate(sample_batch):
        gpu_id = gpu_ids[i % len(gpu_ids)]
        p = ctx.Process(
            target=_worker,
            args=(i, x, gpu_id, callable_path, return_dict),
        )
        p.start()
        procs.append((i, p))

    # join with optional timeout
    if timeout_seconds and timeout_seconds > 0:
        deadline = time.time() + timeout_seconds
        for i, p in procs:
            remaining = max(0.0, deadline - time.time())
            p.join(timeout=remaining)
    else:
        for _, p in procs:
            p.join()

    # kill any alive
    for i, p in procs:
        if p.is_alive():
            try:
                p.terminate()
            except Exception:
                pass
            return_dict[i] = {
                "__ok__": False,
                "__error__": f"Timeout after {timeout_seconds}s (process terminated).",
            }

    # return in order
    results = [return_dict[i] for i in range(len(sample_batch))]
    return results
