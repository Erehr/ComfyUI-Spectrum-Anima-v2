# Solver-step schedule: which outer steps run the real model and which are forecast.

from __future__ import annotations

import math

WARMUP = "warmup"
WINDOW = "window"
FORECAST = "forecast"
TAIL = "tail"

PLAN_SYMBOLS = {WARMUP: "W", WINDOW: "A", FORECAST: "F", TAIL: "T"}


def plan_schedule(total_steps: int, warmup_steps: int, window_size: float, flex_window: float, tail_actual_steps: int) -> list[str]:
    # Spectrum's window rule: after warmup, every floor(window)-th step is actual and the window grows by flex_window after each one.
    plan = []
    window = float(window_size)
    cached = 0
    tail_start = total_steps - tail_actual_steps
    for step in range(total_steps):
        if step >= tail_start:
            plan.append(TAIL)
        elif step < warmup_steps:
            plan.append(WARMUP)
        elif (cached + 1) % max(1, math.floor(window)) == 0:
            plan.append(WINDOW)
            window = round(window + flex_window, 6)
            cached = 0
        else:
            plan.append(FORECAST)
            cached += 1
    return plan


def plan_string(plan: list[str]) -> str:
    return "".join(PLAN_SYMBOLS[p] for p in plan)
