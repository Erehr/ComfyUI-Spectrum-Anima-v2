#ComfyUI integration: wrappers that track outer solver steps and forecast Anima's final hidden feature.

from __future__ import annotations
from typing import Any

import torch
import comfy.ldm.common_dit
import comfy.patcher_extension

from comfy.ldm.anima.model import Anima
from .runtime import SpectrumConfig, SpectrumRuntime

WRAPPER_KEY = "spectrum_anima_v2"
STEP_CONTEXT_KEY = "spectrum_anima_v2_step"


def apply_spectrum(model: Any, cfg: SpectrumConfig) -> Any:
    inner = model.get_model_object("diffusion_model")
    if not isinstance(inner, Anima):
        raise ValueError(f"Spectrum Apply Anima v2 requires a native ComfyUI Anima model, got {type(inner).__name__}.")
    runtime = SpectrumRuntime(cfg)
    patched = model.clone()
    wrappers = (
        (comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, _outer_sample_wrapper(runtime)),
        (comfy.patcher_extension.WrappersMP.PREDICT_NOISE, _predict_noise_wrapper(runtime)),
        (comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, diffusion_model_wrapper),
    )
    for wrapper_type, wrapper in wrappers:
        patched.remove_wrappers_with_key(wrapper_type, WRAPPER_KEY)
        patched.add_wrapper_with_key(wrapper_type, WRAPPER_KEY, wrapper)
    return patched


def _outer_sample_wrapper(runtime: SpectrumRuntime):
    def outer_sample(executor, noise, latent_image, sampler, sigmas, *args, **kwargs):
        runtime.depth += 1
        try:
            if runtime.depth == 1:
                runtime.start_run(sigmas, sampler, noise.shape)
            return executor(noise, latent_image, sampler, sigmas, *args, **kwargs)
        finally:
            if runtime.depth == 1:
                runtime.end_run()
            runtime.depth -= 1

    outer_sample.runtime = runtime
    return outer_sample


def runtime_of(model: Any) -> SpectrumRuntime | None:
    wrappers = model.get_wrappers(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, WRAPPER_KEY)
    return wrappers[0].runtime if wrappers else None


def _predict_noise_wrapper(runtime: SpectrumRuntime):
    def predict_noise(executor, x, timestep, model_options={}, seed=None):
        ctx = runtime.begin_step(float(timestep.flatten()[0]), model_options, x.shape)
        if ctx is None:
            return executor(x, timestep, model_options, seed)
        options = model_options.copy()
        options["transformer_options"] = {**options.get("transformer_options", {}), STEP_CONTEXT_KEY: ctx}
        try:
            return executor(x, timestep, options, seed)
        finally:
            runtime.end_step(ctx)

    return predict_noise


def branch_key(inner: Any, x: torch.Tensor, transformer_options: dict) -> tuple:
    # History identity of one diffusion-model call: the exact batch of conds plus input layout.
    uuids = tuple(transformer_options.get("uuids") or ())
    cond_or_uncond = tuple(transformer_options.get("cond_or_uncond") or ())
    return id(inner), uuids, cond_or_uncond, tuple(x.shape), x.dtype, x.device


def diffusion_model_wrapper(executor, x, timesteps, context, fps=None, padding_mask=None, **kwargs):
    transformer_options = kwargs.get("transformer_options") or {}
    ctx = transformer_options.get(STEP_CONTEXT_KEY)
    if ctx is None:
        return executor(x, timesteps, context, fps, padding_mask, **kwargs)
    runtime = ctx.runtime
    inner = executor.class_obj
    key = branch_key(inner, x, transformer_options)
    if not runtime.enter_call(ctx, key):
        return executor(x, timesteps, context, fps, padding_mask, **kwargs)

    try:
        feature = runtime.predict(ctx, key, x.device)
        if feature is not None:
            return run_output_head(inner, feature, x, timesteps)
    except Exception as exc:
        runtime.disable(f"forecast failed with {type(exc).__name__}: {exc}")

    captured = []
    handle = inner.final_layer.register_forward_pre_hook(lambda module, args: captured.append(args[0].detach()))
    try:
        out = executor(x, timesteps, context, fps, padding_mask, **kwargs)
    finally:
        handle.remove()
    runtime.observe(ctx, key, captured[0] if len(captured) == 1 else None)
    return out


def run_output_head(inner: Any, feature: torch.Tensor, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
    # Native Anima epilogue (MiniTrainDIT._forward) applied to a forecast block-stack output, with the current timestep's AdaLN modulation.
    orig_shape = x.shape
    x = comfy.ldm.common_dit.pad_to_patch_size(x, (inner.patch_temporal, inner.patch_spatial, inner.patch_spatial))
    if timesteps.ndim == 1:
        timesteps = timesteps.unsqueeze(1)
    t_embedding, adaln_lora = inner.t_embedder[1](inner.t_embedder[0](timesteps).to(x.dtype))
    t_embedding = inner.t_embedding_norm(t_embedding)
    out = inner.final_layer(feature, t_embedding, adaln_lora_B_T_3D=adaln_lora)
    return inner.unpatchify(out)[:, :, : orig_shape[-3], : orig_shape[-2], : orig_shape[-1]]
