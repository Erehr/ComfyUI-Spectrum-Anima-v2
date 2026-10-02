import os

from comfy_api.latest import io
from .runtime import SpectrumConfig

DEFAULTS = SpectrumConfig()


class SpectrumApplyAnimaV2(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SpectrumApplyAnimaV2",
            display_name="Spectrum Apply Anima v2",
            category="sampling/spectrum",
            description="Speeds up Anima sampling: some steps skip the transformer and predict its result from earlier steps. The output stays very close to normal sampling but is not identical.",
            inputs=[
                io.Model.Input("model", tooltip="Anima model, after LoRAs and other model patches."),
                io.Float.Input("blend_weight", default=DEFAULTS.blend_weight, min=0.0, max=1.0, step=0.05, tooltip="Mix between the long-range spectral predictor (1) and a short linear extrapolation (0)."),
                io.Int.Input("degree", default=DEFAULTS.degree, min=1, max=16, tooltip="Complexity of the spectral predictor."),
                io.Float.Input("ridge_lambda", default=DEFAULTS.ridge_lambda, min=0.0, max=10.0, step=0.01, tooltip="Smoothing of the spectral predictor."),
                io.Int.Input("warmup_steps", default=DEFAULTS.warmup_steps, min=0, max=200, tooltip="First steps that always run the full model. Early steps decide composition, so a higher value keeps the image closer to normal sampling at the cost of speed."),
                io.Float.Input("start_sigma", default=DEFAULTS.start_sigma, min=0.05, max=1.0, step=0.05, tooltip="Steps are only predicted once the noise level (sigma, 1 = pure noise) is at or below this. Early high-noise steps decide composition; 1 removes the limit."),
                io.Float.Input("window_size", default=DEFAULTS.window_size, min=1.0, max=16.0, step=0.05, tooltip="After warmup, one step in every window_size runs the full model and the rest are predicted. 2 = every other step. 1 turns prediction off."),
                io.Float.Input("flex_window", default=DEFAULTS.flex_window, min=0.0, max=8.0, step=0.05, tooltip="Grows the window after each full step, predicting more steps later in the run."),
                io.Int.Input("tail_actual_steps", default=DEFAULTS.tail_actual_steps, min=0, max=200, tooltip="Last steps that always run the full model. They clean up fine detail and noise; keep at least 2."),
                io.Int.Input("min_fit_points", default=DEFAULTS.min_fit_points, min=1, max=64, advanced=True, tooltip="Full steps needed before a step can be predicted, counted again whenever the batch layout changes (for example when NAG stops). 1 is fastest, 2 is slightly closer to normal sampling."),
                io.Int.Input("history_size", default=DEFAULTS.history_size, min=2, max=64, advanced=True, tooltip="Full steps remembered for prediction."),
                io.Boolean.Input("offload_history", default=DEFAULTS.offload_history, advanced=True, tooltip="Keep remembered steps in system RAM instead of VRAM."),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, model, **settings) -> io.NodeOutput:
        from .anima import apply_spectrum

        debug = os.environ.get("SPECTRUM_ANIMA_DEBUG", "0") not in ("", "0")
        return io.NodeOutput(apply_spectrum(model, SpectrumConfig(**settings, debug=debug)))


NODE_CLASS_MAPPINGS = {"SpectrumApplyAnimaV2": SpectrumApplyAnimaV2}
NODE_DISPLAY_NAME_MAPPINGS = {"SpectrumApplyAnimaV2": "Spectrum Apply Anima v2"}
