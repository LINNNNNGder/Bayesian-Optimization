# src/pipeline.py
# ------------------------------------------------------------
# Paper-grade (lightweight) Batch MOBO pipeline
# - Generic design variables & objectives (from YAML config)
# - Batch qEHVI proposal (BoTorch)
# - Single-machine multi-GPU execution via multiprocessing (spawn)
# - Long-running safe: pending/running/done/failed + resume
# - Generic objective entrypoint: "module:function" returning dict
# ------------------------------------------------------------

from __future__ import annotations

import os
import time
import argparse
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Any, Tuple

import numpy as np
import pandas as pd
import torch

from .config import load_config
from .schema import build_schema
from .mobo import propose_batch_qehvi
from .runner import evaluate_batch_multi_gpu


# =========================
# Helpers
# =========================

def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _ensure_dir_for_file(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def _as_list(x) -> List:
    return list(x) if isinstance(x, (list, tuple)) else [x]


def _coerce_float_dict(d: Dict[str, Any]) -> Dict[str, float]:
    out = {}
    for k, v in d.items():
        out[k] = float(v)
    return out


def _safe_int(x, default=0) -> int:
    try:
        if pd.isna(x):
            return default
        return int(x)
    except Exception:
        return default


def _build_bounds_tensor(cfg: Dict[str, Any], x_cols: List[str]) -> torch.Tensor:
    lower = cfg["design_variables"]["bounds"]["lower"]
    upper = cfg["design_variables"]["bounds"]["upper"]
    if len(lower) != len(x_cols) or len(upper) != len(x_cols):
        raise ValueError(
            f"Bounds length mismatch: got lower={len(lower)}, upper={len(upper)} "
            f"but x_cols={len(x_cols)}"
        )
    return torch.tensor([lower, upper], dtype=torch.double)


# =========================
# Sampling (initial)
# =========================

def latin_hypercube_sampling(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    """
    Simple LHS in [0,1]^d (no external dependency).
    """
    u = rng.random((n, d))
    a = np.linspace(0, 1, n, endpoint=False)
    b = a + 1.0 / n
    pts = np.zeros((n, d), dtype=float)
    for j in range(d):
        # stratified in each dim then permute strata
        strata = a + (b - a) * u[:, j]
        rng.shuffle(strata)
        pts[:, j] = strata
    return pts


def initial_samples(cfg: Dict[str, Any], x_cols: List[str], bounds: torch.Tensor, n: int) -> List[Dict[str, float]]:
    """
    Generate initial samples (LHS by default).
    """
    method = cfg.get("bo", {}).get("initial_sampling", "lhs").lower()
    seed = cfg.get("bo", {}).get("seed", None)
    rng = np.random.default_rng(seed)

    d = len(x_cols)
    if method == "random":
        unit = rng.random((n, d))
    else:
        unit = latin_hypercube_sampling(n=n, d=d, rng=rng)

    lower = bounds[0].cpu().numpy()
    upper = bounds[1].cpu().numpy()
    X = lower + unit * (upper - lower)

    samples: List[Dict[str, float]] = []
    for i in range(n):
        samples.append({x_cols[j]: float(X[i, j]) for j in range(d)})
    return samples


# =========================
# Database
# =========================

def init_database(db_path: str, x_cols: List[str], y_cols: List[str]) -> None:
    """
    Create DB CSV if missing. Adds status + retries + error + runtime fields.
    """
    _ensure_dir_for_file(db_path)
    if os.path.exists(db_path):
        return

    columns = (
        ["run_id", "status", "timestamp", "start_time", "end_time", "retries", "error", "runtime_sec"]
        + x_cols
        + y_cols
    )
    pd.DataFrame(columns=columns).to_csv(db_path, index=False)
    print(f"[INIT] Created database: {db_path}")


def read_db(db_path: str) -> pd.DataFrame:
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found: {db_path}")
    df = pd.read_csv(db_path)
    return df


def write_db(df: pd.DataFrame, db_path: str) -> None:
    df.to_csv(db_path, index=False)


def next_run_id(df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    if "run_id" not in df.columns:
        return 0
    mx = pd.to_numeric(df["run_id"], errors="coerce").max()
    if pd.isna(mx):
        return 0
    return int(mx) + 1


def ensure_required_columns(df: pd.DataFrame, x_cols: List[str], y_cols: List[str]) -> pd.DataFrame:
    """
    If DB exists from older version, ensure all required columns exist.
    """
    required = (
        ["run_id", "status", "timestamp", "start_time", "end_time", "retries", "error", "runtime_sec"]
        + x_cols
        + y_cols
    )
    for c in required:
        if c not in df.columns:
            df[c] = ""
    # defaults
    if "status" in df.columns:
        df["status"] = df["status"].replace("", np.nan).fillna("pending")
    if "retries" in df.columns:
        df["retries"] = pd.to_numeric(df["retries"], errors="coerce").fillna(0).astype(int)
    return df


def load_completed(df: pd.DataFrame, y_cols: List[str]) -> pd.DataFrame:
    """
    Completed means all objectives present and status done.
    """
    if df.empty:
        return df
    mask_obj = df[y_cols].notna().all(axis=1) if y_cols else pd.Series([True] * len(df))
    mask_done = df["status"].astype(str).str.lower().eq("done") if "status" in df.columns else pd.Series([True] * len(df))
    return df[mask_obj & mask_done].copy()


def load_pending(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    st = df["status"].astype(str).str.lower()
    return df[st.eq("pending")].copy()


def load_running(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    st = df["status"].astype(str).str.lower()
    return df[st.eq("running")].copy()


def mark_stale_running_as_pending(df: pd.DataFrame, stale_seconds: int) -> pd.DataFrame:
    """
    If a previous run crashed, some rows might be stuck in 'running'.
    Optionally re-queue them as 'pending' if start_time is older than stale_seconds.
    """
    if stale_seconds <= 0 or df.empty:
        return df
    st = df["status"].astype(str).str.lower()
    running_idx = df.index[st.eq("running")].tolist()
    if not running_idx:
        return df

    def parse_time(s):
        try:
            return datetime.fromisoformat(str(s))
        except Exception:
            return None

    now = datetime.now()
    for idx in running_idx:
        t = parse_time(df.at[idx, "start_time"])
        if t is None:
            # can't parse -> requeue to be safe
            df.at[idx, "status"] = "pending"
            df.at[idx, "error"] = "Recovered from unknown running state"
            continue
        age = (now - t).total_seconds()
        if age > stale_seconds:
            df.at[idx, "status"] = "pending"
            df.at[idx, "error"] = f"Recovered from stale running state (age={int(age)}s)"
    return df


def append_rows(df: pd.DataFrame, rows: List[Dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return df
    return pd.concat([df, pd.DataFrame(rows)], ignore_index=True)


# =========================
# Proposal
# =========================

def propose_next_batch(
    cfg: Dict[str, Any],
    df_done: pd.DataFrame,
    x_cols: List[str],
    y_cols: List[str],
    bounds: torch.Tensor,
) -> List[Dict[str, float]]:
    """
    If completed < num_initial_samples -> initial sampling
    else -> qEHVI batch proposal
    """
    q = int(cfg["bo"]["q_batch"])
    n_init = int(cfg["bo"]["num_initial_samples"])

    if len(df_done) < n_init:
        need = min(q, n_init - len(df_done))
        print(f"[PROPOSE] Initial sampling: generating {need} sample(s) (completed={len(df_done)}/{n_init})")
        return initial_samples(cfg, x_cols, bounds, n=need)

    # BO phase
    print(f"[PROPOSE] BO (qEHVI) proposing q={q} sample(s) using {len(df_done)} completed points")
    return propose_batch_qehvi(
        df_train=df_done,
        x_columns=x_cols,
        y_columns=y_cols,
        bounds=bounds,
        q=q,
        ref_margin=float(cfg["bo"].get("ref_margin", 0.05)),
        num_restarts=int(cfg["bo"].get("num_restarts", 10)),
        raw_samples=int(cfg["bo"].get("raw_samples", 128)),
        mc_samples=int(cfg["bo"].get("mc_samples", 128)),
    )


# =========================
# Main Pipeline
# =========================

def run_pipeline(cfg: Dict[str, Any]) -> None:
    x_cols, y_cols = build_schema(cfg)
    bounds = _build_bounds_tensor(cfg, x_cols)

    db_path = cfg["database"]["path"]
    init_database(db_path, x_cols, y_cols)

    gpu_ids = _as_list(cfg["hardware"]["gpu_ids"])
    start_method = str(cfg["hardware"].get("mp_start_method", "spawn"))
    parallel = bool(cfg["objective"].get("parallel", True))
    callable_path = str(cfg["objective"]["callable"])
    timeout_seconds = int(cfg["objective"].get("timeout_seconds", 0))
    max_retries = int(cfg["objective"].get("max_retries", 1))
    stale_running_seconds = int(cfg["objective"].get("stale_running_seconds", 0))  # optional

    max_iterations = int(cfg["bo"]["max_iterations"])

    print("[INFO] Pipeline started")
    print(f"  - DB: {db_path}")
    print(f"  - X ({len(x_cols)}): {x_cols}")
    print(f"  - Y ({len(y_cols)}): {y_cols}")
    print(f"  - GPUs: {gpu_ids} | mp_start_method={start_method} | parallel={parallel}")
    print(f"  - objective: {callable_path}")

    while True:
        df = read_db(db_path)
        df = ensure_required_columns(df, x_cols, y_cols)
        df = mark_stale_running_as_pending(df, stale_running_seconds)

        df_done = load_completed(df, y_cols)
        n_done = len(df_done)

        # stop criterion: number of completed points reaches max_iterations
        if n_done >= max_iterations:
            print(f"[DONE] Reached max_iterations={max_iterations} with completed={n_done}")
            write_db(df, db_path)
            break

        # If there are pending rows, run them first (resume behavior)
        pending = load_pending(df)
        if pending.empty:
            # Propose new rows
            new_samples = propose_next_batch(cfg, df_done, x_cols, y_cols, bounds)
            rid0 = next_run_id(df)
            rows = []
            for i, x in enumerate(new_samples):
                rows.append({
                    "run_id": rid0 + i,
                    "status": "pending",
                    "timestamp": _now_iso(),
                    "start_time": "",
                    "end_time": "",
                    "retries": 0,
                    "error": "",
                    "runtime_sec": "",
                    **_coerce_float_dict(x),
                    # objectives empty
                    **{k: np.nan for k in y_cols},
                })
            df = append_rows(df, rows)
            write_db(df, db_path)
            pending = load_pending(df)

        # Take up to q_batch pending rows to execute (keep it aligned with GPUs)
        q = int(cfg["bo"]["q_batch"])
        batch_df = pending.head(q).copy()

        # Mark as running + set start_time
        df.loc[batch_df.index, "status"] = "running"
        df.loc[batch_df.index, "start_time"] = _now_iso()
        df.loc[batch_df.index, "end_time"] = ""
        write_db(df, db_path)

        batch_x = batch_df[x_cols].to_dict(orient="records")
        batch_run_ids = batch_df["run_id"].tolist()

        print(f"[RUN] Executing batch size={len(batch_x)} | run_ids={batch_run_ids}")

        results = evaluate_batch_multi_gpu(
            sample_batch=batch_x,
            gpu_ids=gpu_ids,
            callable_path=callable_path,
            start_method=start_method,
            parallel=parallel,
            timeout_seconds=timeout_seconds,
        )

        # Reload DB to reduce race risks (single-writer anyway)
        df = read_db(db_path)
        df = ensure_required_columns(df, x_cols, y_cols)

        # Write back results
        for local_i, (row_idx, row) in enumerate(batch_df.iterrows()):
            r = results[local_i]
            ok = bool(r.get("__ok__", False))

            if ok:
                # validate objective keys
                missing = [k for k in y_cols if k not in r]
                if missing:
                    ok = False
                    r = {
                        "__ok__": False,
                        "__error__": f"objective_function returned missing keys: {missing}. Got keys={list(r.keys())}",
                    }

            if ok:
                for obj in y_cols:
                    df.at[row_idx, obj] = float(r[obj])
                df.at[row_idx, "status"] = "done"
                df.at[row_idx, "error"] = ""
                df.at[row_idx, "runtime_sec"] = r.get("__runtime_sec__", "")
            else:
                # failed -> retry or mark failed
                prev = _safe_int(df.at[row_idx, "retries"], default=0)
                new_retry = prev + 1
                df.at[row_idx, "retries"] = new_retry
                df.at[row_idx, "error"] = str(r.get("__error__", "unknown error"))

                if new_retry <= max_retries:
                    df.at[row_idx, "status"] = "pending"
                    print(f"[WARN] run_id={int(df.at[row_idx,'run_id'])} failed; retry {new_retry}/{max_retries}")
                else:
                    df.at[row_idx, "status"] = "failed"
                    print(f"[FAIL] run_id={int(df.at[row_idx,'run_id'])} failed; exceeded max_retries={max_retries}")

            df.at[row_idx, "end_time"] = _now_iso()

        write_db(df, db_path)

        # Brief progress
        df_done = load_completed(df, y_cols)
        print(f"[INFO] Progress: completed={len(df_done)}/{max_iterations}\n")

        # tiny sleep to reduce tight-loop file churn
        time.sleep(float(cfg.get("runtime", {}).get("loop_sleep_seconds", 0.5)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to configs/exp.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_pipeline(cfg)


if __name__ == "__main__":
    main()
