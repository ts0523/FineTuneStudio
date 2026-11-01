# -*- coding: utf-8 -*-
"""目标 → 训练方案：把「我想让模型学会什么」翻译成可执行的超参与时长预估。

时长模型（本机实测标定，i3-13100 / 8 线程 / 纯 CPU）：
    单个微批耗时 t_micro ≈ K * (params/135M) * (max_len/192)
    优化步耗时   t_step  ≈ t_micro * batch_size * grad_accum + 0.3s（优化器 + 数据搬运）
    K = 1.1s（scripts/train_lora.py 实测：batch=1, len=192, 135M 参数 ≈ 1.1s/微批）
预估误差约 ±25%；跑过一次之后会用实测 tokens/s 反向校准。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from . import dataset as ds

K_SECONDS = 1.1
BASE_PARAMS_M = 135.0
BASE_LEN = 192
STEP_OVERHEAD = 0.3
FALLBACK_TPS = 155.0     # 135M / 4 线程 / 本机实测：150~159 tokens/s
LOAD_SECONDS_PER_135M = 2.5

QUALITY_TIERS = {
    "smoke": ("冒烟验证", 2, {"epochs": 1.0, "max_len": 192, "lora_r": 8, "batch": 1, "accum": 1}),
    "usable": ("可用", 3, {"epochs": 3.0, "max_len": 256, "lora_r": 16, "batch": 2, "accum": 2}),
    "serious": ("认真", 5, {"epochs": 5.0, "max_len": 384, "lora_r": 32, "batch": 2, "accum": 4}),
}

TASK_PRESETS = {
    "客服问答": {"lr": 2e-4, "max_len": 384, "epochs": 3.0, "note": "问答对为主，序列偏长，注意截断"},
    "风格模仿": {"lr": 1e-4, "max_len": 512, "epochs": 4.0, "note": "风格类任务学习率要低，避免覆盖原有能力"},
    "信息抽取": {"lr": 2e-4, "max_len": 256, "epochs": 4.0, "note": "输出短、结构固定，小 r 就够"},
    "翻译": {"lr": 1.5e-4, "max_len": 384, "epochs": 3.0, "note": "双语数据要成对，长度对齐"},
    "代码补全": {"lr": 1e-4, "max_len": 512, "epochs": 3.0, "note": "长序列，CPU 上会明显变慢"},
    "自定义": {"lr": 2e-4, "max_len": 256, "epochs": 3.0, "note": "按默认 SFT 参数起步"},
}


@dataclass
class Plan:
    model: str = ""
    params_m: float = 135.0
    mode: str = "lora"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lr: float = 2e-4
    epochs: float = 3.0
    batch_size: int = 2
    grad_accum: int = 2
    max_len: int = 256
    val_ratio: float = 0.1
    steps: int = 0
    micro_steps: int = 0
    tokens: float = 0.0
    est_seconds: float = 0.0
    est_ram_gb: float = 0.0
    samples: int = 0
    avg_tokens: float = 0.0
    notes: list[str] = field(default_factory=list)
    quality: str = "usable"
    task: str = "客服问答"

    def est_text(self) -> str:
        from .paths import human_time

        return human_time(self.est_seconds)

    def to_train_cfg(self) -> dict:
        return {
            "mode": self.mode,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "grad_accum": self.grad_accum,
            "lr": self.lr,
            "max_len": self.max_len,
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "val_ratio": self.val_ratio,
            "full": self.mode == "full",
        }


def steps_per_epoch(n_train: int, batch: int, accum: int) -> int:
    eff = max(1, batch * accum)
    return max(1, math.ceil(n_train / eff))


def estimate_tps(params_m: float, max_len: int, samples: list[dict] | None = None,
                 threads: int = 4) -> float:
    """返回 tokens/s 估计：以 135M/4 线程实测为基准，按参数量与线程数缩放。

    实测（本机 i3-13100，4 线程）：len 192/256/384 下都在 150~159 tok/s，
    说明吞吐基本只跟参数量与线程数相关；8 线程没有提升（只有 4 个物理核）。
    """
    p = max(10.0, float(params_m))
    t = max(1, int(threads or 4))
    thread_factor = min(1.8, t / 4.0)
    base = FALLBACK_TPS * (BASE_PARAMS_M / p) * thread_factor
    base *= (max(64, max_len) / 256.0) ** 0.15
    if samples:
        tps = [s.get("tps") for s in samples if isinstance(s.get("tps"), (int, float)) and s["tps"] > 0]
        if tps:
            tps.sort()
            med = tps[len(tps) // 2]
            pm = [s.get("params_m") for s in samples
                  if isinstance(s.get("params_m"), (int, float)) and s["params_m"] > 0]
            if pm:
                med *= (sum(pm) / len(pm)) / p
            return float((med + base) / 2)
    return float(base)


def estimate(params_m: float, samples: int, avg_tokens: float, epochs: float,
             batch: int, accum: int, max_len: int, mode: str = "lora",
             hidden: int = 576, layers: int = 30,
             speed_samples: list[dict] | None = None, threads: int = 4) -> dict:
    steps = steps_per_epoch(samples, batch, accum)
    total_steps = max(1, int(round(steps * epochs)))
    tokens = samples * avg_tokens * epochs
    tps = estimate_tps(params_m, max_len, speed_samples, threads)
    train_seconds = tokens / max(1.0, tps)
    overhead = total_steps * STEP_OVERHEAD
    load = LOAD_SECONDS_PER_135M * (params_m / BASE_PARAMS_M)
    est = train_seconds + overhead + load
    # 内存：权重 fp32 + 梯度/优化器状态 + 激活
    weights = params_m * 1e6 * 4
    if mode == "full":
        mem = weights * 4.0
    else:
        mem = weights * 1.05 + params_m * 1e6 * 0.02
    act = batch * max_len * hidden * layers * 2 * (3 if mode == "full" else 1.6)
    ram_gb = (mem + act) / 1024 ** 3
    return {
        "steps": total_steps, "steps_per_epoch": steps, "tokens": tokens,
        "tps": tps, "est_seconds": est, "ram_gb": ram_gb,
        "train_seconds": train_seconds, "overhead_seconds": overhead + load,
    }


def make_plan(goal: str, task: str, quality: str, model_name: str, params_m: float,
              samples: int, avg_tokens: float, hidden: int = 576, layers: int = 30,
              speed_samples: list[dict] | None = None, mode: str = "lora",
              threads: int = 4) -> Plan:
    tier = QUALITY_TIERS.get(quality, QUALITY_TIERS["usable"])
    tname, epochs_hint, base = tier
    tp = TASK_PRESETS.get(task, TASK_PRESETS["自定义"])
    plan = Plan(model=model_name, params_m=params_m, task=task, quality=quality, mode=mode)
    plan.epochs = float(base["epochs"])
    plan.max_len = int(min(base["max_len"], tp["max_len"]))
    plan.lora_r = int(base["lora_r"])
    plan.lora_alpha = plan.lora_r * 2
    plan.batch_size = int(base["batch"])
    plan.grad_accum = int(base["accum"])
    plan.lr = float(tp["lr"])
    plan.samples = max(1, samples)
    plan.avg_tokens = max(1.0, avg_tokens)
    # 数据量小的时候多跑几轮才有意义，数据量大时减少轮数防止过拟合
    if samples < 60:
        plan.epochs = max(plan.epochs, 8.0 if samples < 30 else 5.0)
        plan.notes.append(f"样本只有 {samples} 条，已自动提高轮数到 {plan.epochs:.0f}；"
                          "建议至少 300 条，否则模型主要学到的是格式与人设")
    elif samples > 5000:
        plan.epochs = min(plan.epochs, 2.0)
        plan.notes.append("样本较多，轮数已下调到 2，避免过拟合")
    if params_m > 1000:
        plan.batch_size = 1
        plan.grad_accum = max(4, plan.grad_accum)
        plan.notes.append("模型超过 1B，CPU 上批量已降到 1 并提高梯度累积")
    est = estimate(params_m, plan.samples, plan.avg_tokens, plan.epochs, plan.batch_size,
                   plan.grad_accum, plan.max_len, plan.mode, hidden, layers, speed_samples,
                   threads)
    plan.steps = est["steps"]
    plan.tokens = est["tokens"]
    plan.est_seconds = est["est_seconds"]
    plan.est_ram_gb = est["ram_gb"]
    plan.notes.append(tp["note"])
    if plan.est_seconds > 3600:
        plan.notes.append(f"预计超过 1 小时（{plan.est_seconds/3600:.1f}h），可先选「冒烟验证」跑通链路")
    if not goal.strip():
        plan.notes.append("未填写目标描述；建议写清「模型要学会回答什么」，便于后续对照效果")
    return plan


def avg_tokens_from_stats(st: dict) -> float:
    """用数据统计里的粗估 token 数换算平均每条 token。"""
    if not st or not st.get("valid"):
        return 120.0
    return max(16.0, float(st.get("est_tokens", 0)) / max(1, st["valid"]))


def quick_stats(path) -> dict:
    st = ds.stats(path)
    return {"samples": st.get("valid", 0), "avg_tokens": avg_tokens_from_stats(st), "raw": st}
