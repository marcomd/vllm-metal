# SPDX-License-Identifier: Apache-2.0
"""Let Metal-served features past vLLM's V1 GPU-runner support check."""

from collections.abc import Callable
from functools import wraps
from typing import Any

# Feature label in vLLM's unsupported list -> whether Metal serves it for a config.
_ALLOWED: dict[str, Callable[[Any], bool]] = {}


def allow_v1_runner_feature(feature: str, allowed: Callable[[Any], bool]) -> None:
    """Drop ``feature`` from ``VllmConfig._get_v1_model_runner_unsupported_features``.

    vLLM reports features its V1 GPU runner lacks, and ``MetalModelRunner``
    serves some of them itself. One wrapper holds every rule, so independent
    bridges (DSpark, diffusion) share it instead of stacking their own.
    ``allowed(vllm_config)`` decides per config; a feature without a matching
    rule still fails upstream's check. Install from platform config
    validation, after ``VllmConfig`` is fully imported: importing
    ``vllm.config`` at plugin registration is circular.
    """
    from vllm.config import VllmConfig

    _ALLOWED[feature] = allowed
    original = getattr(VllmConfig, "_get_v1_model_runner_unsupported_features", None)
    if original is None or getattr(original, "_metal_v1_runner_guard", False):
        return

    @wraps(original)
    def unsupported_features(self: Any) -> list[str]:
        return [
            item
            for item in original(self)
            if not (item in _ALLOWED and _ALLOWED[item](self))
        ]

    unsupported_features._metal_v1_runner_guard = True  # type: ignore[attr-defined]
    VllmConfig._get_v1_model_runner_unsupported_features = unsupported_features
