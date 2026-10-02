# Spectrum forecasting math: Chebyshev ridge regression blended with local linear extrapolation.

from __future__ import annotations
from typing import Sequence

import torch

def chebyshev_basis(coords: torch.Tensor, degree: int) -> torch.Tensor:
    tau = coords.reshape(-1, 1)
    cols = [torch.ones_like(tau)]
    if degree >= 1:
        cols.append(tau)
    for _ in range(2, degree + 1):
        cols.append(2.0 * tau * cols[-1] - cols[-2])
    return torch.cat(cols, dim=1)


def linear_weights(coords: Sequence[float], target: float) -> torch.Tensor:
    weights = torch.zeros(len(coords), dtype=torch.float64)
    if len(coords) < 2 or abs(coords[-1] - coords[-2]) < 1e-12:
        weights[-1] = 1.0
        return weights
    k = (target - coords[-1]) / (coords[-1] - coords[-2])
    weights[-1] = 1.0 + k
    weights[-2] = -k
    return weights


def chebyshev_weights(coords: Sequence[float], target: float, degree: int, ridge_lambda: float) -> torch.Tensor:
    x = chebyshev_basis(torch.tensor(coords, dtype=torch.float64), degree)
    # A tiny floor keeps ridge_lambda=0 solvable when there are fewer points than basis functions (minimum-norm limit).
    gram = x.T @ x + max(float(ridge_lambda), 1e-10) * torch.eye(degree + 1, dtype=torch.float64)
    x_star = chebyshev_basis(torch.tensor([target], dtype=torch.float64), degree)
    return (x_star @ torch.linalg.solve(gram, x.T)).reshape(-1)


def spectrum_weights(coords: Sequence[float], target: float, degree: int, ridge_lambda: float, blend_weight: float) -> torch.Tensor:
    weights = (1.0 - blend_weight) * linear_weights(coords, target)
    if blend_weight > 0.0:
        weights += blend_weight * chebyshev_weights(coords, target, degree, ridge_lambda)
    return weights


def combine(features: Sequence[torch.Tensor], weights: torch.Tensor, device: torch.device) -> torch.Tensor:
    # Float32 weighted sum of history features on `device`; features may live on the host.
    out = None
    for weight, feature in zip(weights.tolist(), features):
        if weight == 0.0:
            continue
        feature = feature.to(device, non_blocking=True).float()
        if out is None:
            out = feature * weight
        else:
            out.add_(feature, alpha=weight)
    return out


def relative_error(prediction: torch.Tensor, actual: torch.Tensor) -> float:
    # Relative L2 error. Consumes `prediction` (float32 scratch tensor).
    denom = torch.linalg.vector_norm(actual, dtype=torch.float32).clamp_min(1e-12)
    return float(torch.linalg.vector_norm(prediction.sub_(actual)) / denom)
