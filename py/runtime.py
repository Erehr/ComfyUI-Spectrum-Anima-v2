# Per-sampling-run Spectrum state: outer solver-step tracking, per-branch feature history, decisions and debug statistics.

from __future__ import annotations

import logging
import math
import time

from dataclasses import dataclass, field
from typing import Any, Hashable, Optional

import torch

from .forecast import combine, relative_error, spectrum_weights
from .scheduler import FORECAST, plan_schedule, plan_string

LOG = logging.getLogger("spectrum_anima_v2")
PREFIX = "[Spectrum Anima v2]"

# Verified to call the model exactly once per solver step, at the scheduled sigma (tests/test_sampler_contract.py).
SUPPORTED_SAMPLERS = frozenset({
    "sample_euler",
    "sample_euler_cfg_pp",
    "sample_euler_ancestral",
    "sample_euler_ancestral_RF",
    "sample_euler_ancestral_cfg_pp",
    "sample_lms",
    "sample_dpmpp_2m",
    "sample_dpmpp_2m_cfg_pp",
    "sample_dpmpp_2m_sde",
    "sample_dpmpp_2m_sde_gpu",
    "sample_dpmpp_2m_sde_heun",
    "sample_dpmpp_2m_sde_heun_gpu",
    "sample_dpmpp_3m_sde",
    "sample_dpmpp_3m_sde_gpu",
    "sample_ddpm",
    "sample_lcm",
    "sample_ipndm",
    "sample_ipndm_v",
    "sample_deis",
    "sample_res_multistep",
    "sample_res_multistep_cfg_pp",
    "sample_res_multistep_ancestral",
    "sample_res_multistep_ancestral_cfg_pp",
    "sample_gradient_estimation",
    "sample_gradient_estimation_cfg_pp",
    "sample_er_sde",
    "sample_sa_solver",
    "sample_cfgpp_ud10_ab",
})

# SDE samplers evaluate the first step at a slightly clamped sigma (about 3e-5 below 1.0 for flow models).
SIGMA_TOLERANCE = 1e-4


@dataclass(frozen=True)
class SpectrumConfig:
    blend_weight: float = 0.5
    degree: int = 4
    ridge_lambda: float = 0.25
    warmup_steps: int = 4
    start_sigma: float = 0.7
    window_size: float = 2.0
    flex_window: float = 0.0
    tail_actual_steps: int = 3
    history_size: int = 16
    min_fit_points: int = 1
    adaptive_forecast: bool = False
    forecast_validation: bool = False
    forecast_error_threshold: float = 0.1
    offload_history: bool = True
    debug: bool = False

    def validate(self) -> "SpectrumConfig":
        checks = (
            (0.0 <= self.blend_weight <= 1.0, "blend_weight must be in [0, 1]"),
            (self.degree >= 1, "degree must be >= 1"),
            (self.ridge_lambda >= 0.0, "ridge_lambda must be >= 0"),
            (self.warmup_steps >= 0, "warmup_steps must be >= 0"),
            (0.0 < self.start_sigma <= 1.0, "start_sigma must be in (0, 1]"),
            (self.window_size >= 1.0, "window_size must be >= 1"),
            (self.flex_window >= 0.0, "flex_window must be >= 0"),
            (self.tail_actual_steps >= 0, "tail_actual_steps must be >= 0"),
            (self.min_fit_points >= 1, "min_fit_points must be >= 1"),
            (self.history_size >= max(2, self.min_fit_points), "history_size must be >= max(2, min_fit_points)"),
            (self.forecast_error_threshold > 0.0, "forecast_error_threshold must be > 0"),
        )
        for ok, message in checks:
            if not ok:
                raise ValueError(message)
        return self


def describe_sampler(sampler: Any) -> tuple[str, bool]:
    # Returns (display name, supported). SamplerSPEED is resolved to the solver it runs per segment.
    function = getattr(sampler, "sampler_function", None)
    name = getattr(function, "__name__", type(sampler).__name__)
    solver = name
    if name == "sample_speed":
        base = (getattr(sampler, "extra_options", None) or {}).get("base_sampler")
        solver = f"sample_{base}"
        name = f"sample_speed[{base}]"
    return name, solver in SUPPORTED_SAMPLERS


def _to_host(feature: torch.Tensor) -> torch.Tensor:
    if feature.device.type == "cpu":
        return feature.clone()
    try:
        host = torch.empty(feature.shape, dtype=feature.dtype, pin_memory=True)
    except RuntimeError:
        return feature.to("cpu")
    return host.copy_(feature, non_blocking=True)


class FeatureHistory:
    def __init__(self) -> None:
        self.coords: list[float] = []
        self.features: list[torch.Tensor] = []
        self.dtype: Optional[torch.dtype] = None
        self.last_error: Optional[float] = None

    def __len__(self) -> int:
        return len(self.coords)

    def add(self, coord: float, feature: torch.Tensor, size: int, offload: bool) -> None:
        self.dtype = feature.dtype
        self.coords.append(coord)
        self.features.append(_to_host(feature) if offload else feature.clone())
        if len(self.coords) > size:
            del self.coords[0]
            del self.features[0]


@dataclass
class StepContext:
    runtime: "SpectrumRuntime"
    run_id: int
    index: int
    sigma: float
    coord: float
    reason: str
    keys: set = field(default_factory=set)
    forecasts: int = 0
    actuals: int = 0
    notes: list = field(default_factory=list)
    errors: list = field(default_factory=list)

    @property
    def wants_forecast(self) -> bool:
        return self.reason == FORECAST


@dataclass
class RunStats:
    steps: int = 0
    actual_steps: int = 0
    forecast_steps: int = 0
    mixed_steps: int = 0
    forecast_seconds: float = 0.0
    validation_seconds: float = 0.0
    observe_seconds: float = 0.0
    errors: list = field(default_factory=list)


class SpectrumRuntime:
    #State for one patched model. Each sampling run starts from a clean slate and releases all history when it ends.

    def __init__(self, cfg: SpectrumConfig):
        self.cfg = cfg.validate()
        self.run_id = 0
        self.depth = 0
        self.last_stats: Optional[RunStats] = None
        self._clear()

    def _clear(self) -> None:
        self.active = False
        self.disabled_reason: Optional[str] = None
        self.sampler_name = ""
        self.latent_shape: Optional[tuple] = None
        self.sigmas: list[float] = []
        self.plan: list[str] = []
        self.last_forecast_index = -1
        self.sigma_first = 0.0
        self.sigma_span = 0.0
        self.next_step = 0
        self.current: Optional[StepContext] = None
        self.histories: dict[Hashable, FeatureHistory] = {}
        self.stats = RunStats()
        self.started = 0.0

    @property
    def total_steps(self) -> int:
        return max(len(self.sigmas) - 1, 0)

    def coord(self, sigma: float) -> float:
        return 2.0 * (sigma - self.sigma_first) / self.sigma_span - 1.0

    def _log(self, message: str, *args: Any) -> None:
        LOG.info(f"{PREFIX} run={self.run_id} " + message, *args)

    def disable(self, reason: str) -> None:
        if self.disabled_reason is None:
            self.disabled_reason = reason
            self.histories.clear()
            if self.cfg.debug:
                self._log("forecasting disabled: %s", reason)

    def start_run(self, sigmas: torch.Tensor, sampler: Any, latent_shape: Optional[tuple] = None) -> None:
        self._clear()
        self.latent_shape = tuple(latent_shape) if latent_shape is not None else None
        self.run_id += 1
        self.active = True
        self.started = time.perf_counter()
        self.sampler_name, supported = describe_sampler(sampler)
        self.sigmas = [float(s) for s in sigmas.detach().flatten().tolist()]
        total = self.total_steps
        cfg = self.cfg
        self.plan = plan_schedule(total, cfg.warmup_steps, cfg.window_size, cfg.flex_window, cfg.tail_actual_steps)
        self.last_forecast_index = max((i for i, reason in enumerate(self.plan) if reason == FORECAST and self.sigmas[i] <= cfg.start_sigma), default=-1)
        if total >= 2:
            self.sigma_first = self.sigmas[0]
            self.sigma_span = self.sigmas[total - 1] - self.sigmas[0]
        if not supported:
            self.disable(f"sampler {self.sampler_name} is not validated for solver-step tracking")
            LOG.warning("%s %s; running native Anima.", PREFIX, self.disabled_reason)
        elif total < 2 or abs(self.sigma_span) < 1e-12:
            self.disable("sigma schedule is too short")
        elif self.last_forecast_index < 0:
            self.disable("warmup/window/tail settings leave no forecast steps")
        if cfg.debug:
            self._log("sampler=%s steps=%d plan=%s (W warmup, A actual, F forecast, T tail)", self.sampler_name, total, plan_string(self.plan))

    def end_run(self) -> None:
        if self.cfg.debug:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            s = self.stats
            done = max(s.steps, 1)
            self._log(
                "summary: steps=%d actual=%d forecast=%d mixed=%d (forecast %.1f%%) wall=%.2fs forecast_time=%.3fs validation_time=%.3fs history_time=%.3fs%s%s",
                s.steps, s.actual_steps, s.forecast_steps, s.mixed_steps, 100.0 * s.forecast_steps / done,
                time.perf_counter() - self.started, s.forecast_seconds, s.validation_seconds, s.observe_seconds,
                f" forecast_error mean={sum(s.errors) / len(s.errors):.4f} max={max(s.errors):.4f} n={len(s.errors)}" if s.errors else "",
                f" disabled: {self.disabled_reason}" if self.disabled_reason else "",
            )
        self.last_stats = self.stats
        self._clear()

    def begin_step(self, sigma: float, model_options: dict, shape: Optional[tuple] = None) -> Optional[StepContext]:
        # Called once per outer solver step (one guider predict_noise call). Returns None when this step must run natively without recording.
        if not self.active or self.depth != 1 or self.current is not None:
            return None
        index = self.next_step
        self.next_step += 1
        if index >= self.total_steps:
            self.disable("sampler made more model evaluations than solver steps")
        if "multigpu_clones" in model_options:
            self.disable("multi-GPU sampling is not supported")
        if self.disabled_reason is not None:
            self._finish_step_stats(index, sigma, "ACTUAL", f"reason=disabled ({self.disabled_reason})")
            return None
        if shape is not None and self.latent_shape is not None and tuple(shape) != self.latent_shape:
            # Reduced-resolution steps (SamplerSPEED's early segments) are cheap to run and their forecasts measurably change composition.
            self._finish_step_stats(index, sigma, "ACTUAL", "reason=reduced_resolution")
            return None
        reason = self.plan[index]
        # High-noise steps decide composition; skipping them changed images far more than skipping later steps.
        if reason == FORECAST and self.sigmas[index] > self.cfg.start_sigma:
            reason = "high_sigma"
        if reason == FORECAST and not math.isclose(sigma, self.sigmas[index], rel_tol=SIGMA_TOLERANCE, abs_tol=SIGMA_TOLERANCE):
            reason = "sigma_mismatch"
        if index > self.last_forecast_index and not self.cfg.forecast_validation:
            self._finish_step_stats(index, sigma, "ACTUAL", f"reason={reason}")
            return None
        self.current = StepContext(self, self.run_id, index, sigma, self.coord(sigma), reason)
        return self.current

    def end_step(self, ctx: StepContext) -> None:
        if self.current is ctx:
            self.current = None
        if ctx.run_id != self.run_id or not self.active:
            return
        if ctx.forecasts and ctx.actuals:
            outcome = "MIXED"
        elif ctx.forecasts:
            outcome = "FORECAST"
        else:
            outcome = "ACTUAL"
        detail = f"reason={ctx.reason}" if outcome == "ACTUAL" else ("linear+chebyshev" if self.cfg.blend_weight > 0 else "linear")
        if ctx.notes:
            detail += " fallback=" + ",".join(ctx.notes)
        if ctx.errors:
            detail += " forecast_error=" + ",".join(f"{e:.4f}" for e in ctx.errors)
        self._finish_step_stats(ctx.index, ctx.sigma, outcome, detail)

    def _finish_step_stats(self, index: int, sigma: float, outcome: str, detail: str) -> None:
        s = self.stats
        s.steps += 1
        if outcome == "FORECAST":
            s.forecast_steps += 1
        elif outcome == "MIXED":
            s.mixed_steps += 1
        else:
            s.actual_steps += 1
        if self.cfg.debug:
            self._log("step=%d/%d sigma=%.6f %s %s", index, self.total_steps, sigma, outcome, detail)

    def enter_call(self, ctx: StepContext, key: Hashable) -> bool:
        """Registers one diffusion-model call of the current step. False means: run natively and record nothing."""
        if ctx is not self.current or ctx.run_id != self.run_id or self.disabled_reason is not None:
            return False
        if key in ctx.keys:
            self.disable("repeated model call for the same branch within one solver step")
            return False
        ctx.keys.add(key)
        return True

    def _forecast(self, history: FeatureHistory, coord: float, device: torch.device) -> torch.Tensor:
        cfg = self.cfg
        weights = spectrum_weights(history.coords, coord, cfg.degree, cfg.ridge_lambda, cfg.blend_weight)
        return combine(history.features, weights, device)

    def _sync(self) -> None:
        if self.cfg.debug and torch.cuda.is_available():
            torch.cuda.synchronize()

    def predict(self, ctx: StepContext, key: Hashable, device: torch.device) -> Optional[torch.Tensor]:
        if not ctx.wants_forecast:
            return None
        cfg = self.cfg
        history = self.histories.get(key)
        if history is None or len(history) < cfg.min_fit_points:
            ctx.notes.append("insufficient_history")
            return None
        if cfg.adaptive_forecast and history.last_error is None:
            ctx.notes.append("no_trust_data")
            return None
        # A feature error becomes a velocity error, which enters the denoised estimate x - sigma * v scaled by sigma.
        if cfg.adaptive_forecast and ctx.sigma * history.last_error > cfg.forecast_error_threshold:
            ctx.notes.append(f"low_trust({ctx.sigma * history.last_error:.4f})")
            return None
        self._sync()
        started = time.perf_counter()
        prediction = self._forecast(history, ctx.coord, device)
        finite = bool(torch.isfinite(prediction).all())
        self.stats.forecast_seconds += time.perf_counter() - started
        if not finite:
            ctx.notes.append("nonfinite")
            return None
        ctx.forecasts += 1
        return prediction.to(history.dtype)

    def observe(self, ctx: StepContext, key: Hashable, feature: Optional[torch.Tensor]) -> None:
        if ctx is not self.current or self.disabled_reason is not None:
            return
        if feature is None:
            self.disable("Anima final layer input was not observed during the model call")
            return
        cfg = self.cfg
        ctx.actuals += 1
        history = self.histories.setdefault(key, FeatureHistory())
        if (cfg.forecast_validation or cfg.adaptive_forecast) and len(history) >= cfg.min_fit_points:
            self._sync()
            started = time.perf_counter()
            history.last_error = relative_error(self._forecast(history, ctx.coord, feature.device), feature)
            self.stats.validation_seconds += time.perf_counter() - started
            ctx.errors.append(history.last_error)
            self.stats.errors.append(history.last_error)
        if ctx.index < self.last_forecast_index or cfg.forecast_validation:
            self._sync()
            started = time.perf_counter()
            history.add(ctx.coord, feature, cfg.history_size, cfg.offload_history)
            self._sync()
            self.stats.observe_seconds += time.perf_counter() - started
