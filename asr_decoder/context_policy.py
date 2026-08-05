# Copyright (c) 2024, Zhendong Peng (pzd17@tsinghua.org.cn)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum


class HotwordStrength(str, Enum):
    """User-facing contextual bias strength."""

    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"
    LOW_FALSE_ACTIVATION = "low_false_activation"


class DecodingQuality(str, Enum):
    """User-facing latency and search-quality trade-off."""

    LOW_LATENCY = "low_latency"
    BALANCED = "balanced"
    HIGH_ACCURACY = "high_accuracy"

    @property
    def search_settings(self) -> tuple[int, int, float]:
        return {
            self.LOW_LATENCY: (4, 8, 3.0),
            self.BALANCED: (8, 16, 4.0),
            self.HIGH_ACCURACY: (12, 24, 5.0),
        }[self]


@dataclass(frozen=True)
class AdaptiveGatingConfig:
    """Bounds for adapting context candidate gates from acoustic evidence."""

    enabled: bool = True
    reference_top_spread: float = 4.0
    min_acoustic_scale: float = 0.5
    max_acoustic_scale: float = 2.0
    min_entropy_confidence_factor: float = 0.65
    max_active_context_penalty: float = 0.35

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("enabled must be a bool")
        _require_finite_positive("reference_top_spread", self.reference_top_spread)
        _require_finite_positive("min_acoustic_scale", self.min_acoustic_scale)
        _require_finite_positive("max_acoustic_scale", self.max_acoustic_scale)
        if self.min_acoustic_scale > self.max_acoustic_scale:
            raise ValueError("acoustic scale limits must be ordered")
        _require_finite_range(
            "min_entropy_confidence_factor",
            self.min_entropy_confidence_factor,
            lower=0.0,
            upper=1.0,
            lower_inclusive=False,
        )
        _require_finite_range(
            "max_active_context_penalty",
            self.max_active_context_penalty,
            lower=0.0,
            upper=1.0,
            upper_inclusive=False,
        )


@dataclass(frozen=True)
class ContextPolicy:
    """Complete advanced policy for contextual scoring and candidate pruning."""

    completion_bonus: float = 3.5
    context_token_prune_threshold: float = 1.5
    max_injected_candidates: int = 8
    max_completed_contexts_per_token: int = 8
    gating: AdaptiveGatingConfig = field(default_factory=AdaptiveGatingConfig)

    def __post_init__(self) -> None:
        _require_finite_non_negative("completion_bonus", self.completion_bonus)
        _require_finite_non_negative("context_token_prune_threshold", self.context_token_prune_threshold)
        if not isinstance(self.max_injected_candidates, int) or isinstance(self.max_injected_candidates, bool):
            raise TypeError("max_injected_candidates must be an integer")
        if self.max_injected_candidates < 0:
            raise ValueError("max_injected_candidates must be non-negative")
        if not isinstance(self.max_completed_contexts_per_token, int) or isinstance(
            self.max_completed_contexts_per_token, bool
        ):
            raise TypeError("max_completed_contexts_per_token must be an integer")
        if self.max_completed_contexts_per_token < 1:
            raise ValueError("max_completed_contexts_per_token must be positive")
        if not isinstance(self.gating, AdaptiveGatingConfig):
            raise TypeError("gating must be an AdaptiveGatingConfig")

    @classmethod
    def conservative(cls) -> "ContextPolicy":
        return cls()

    @classmethod
    def balanced(cls) -> "ContextPolicy":
        return cls(
            completion_bonus=5.0,
            context_token_prune_threshold=2.5,
            max_injected_candidates=12,
        )

    @classmethod
    def aggressive(cls) -> "ContextPolicy":
        return cls(
            completion_bonus=7.0,
            context_token_prune_threshold=3.5,
            max_injected_candidates=16,
        )

    @classmethod
    def low_false_activation(cls) -> "ContextPolicy":
        """Prefer precision when a tokenizer has short, ambiguous word pieces.

        This preset is useful for English SentencePiece vocabularies where a
        hotword can share a short prefix with an ordinary word (for example,
        ``google`` and ``good``). It remains model-agnostic: the caller still
        supplies the model token IDs and, when available, word-boundary IDs.
        """

        return cls(
            completion_bonus=1.0,
            context_token_prune_threshold=0.5,
            max_injected_candidates=8,
        )


@dataclass(frozen=True)
class GatingFrameStats:
    acoustic_scale: float
    confidence_factor: float


class AdaptiveContextGate:
    """Adapt context candidate gates from the current acoustic frame only."""

    def __init__(self, config: AdaptiveGatingConfig) -> None:
        self.config = config

    def analyze_frame(self, top_log_probabilities: Sequence[float], blank_log_probability: float) -> GatingFrameStats:
        if not self.config.enabled or len(top_log_probabilities) < 2:
            return GatingFrameStats(1.0, 1.0)

        best = top_log_probabilities[0]
        finite = [value for value in top_log_probabilities if math.isfinite(value)]
        if len(finite) < 2:
            return GatingFrameStats(1.0, 1.0)

        spread = best - finite[-1]
        acoustic_scale = min(
            self.config.max_acoustic_scale,
            max(self.config.min_acoustic_scale, spread / self.config.reference_top_spread),
        )

        probabilities = [math.exp(value - best) for value in finite]
        total = sum(probabilities)
        normalized = [probability / total for probability in probabilities]
        entropy = -sum(probability * math.log(probability) for probability in normalized if probability > 0)
        normalized_entropy = entropy / math.log(len(normalized))
        confidence_factor = 1.0 - (1.0 - self.config.min_entropy_confidence_factor) * normalized_entropy
        if blank_log_probability == best:
            blank_margin = (best - finite[1]) / max(acoustic_scale, 1e-6)
            if blank_margin > 1.0:
                confidence_factor *= max(0.25, 1.0 / blank_margin)
        return GatingFrameStats(acoustic_scale, confidence_factor)

    def active_context_factor(self, active_candidate_count: int, maximum: int) -> float:
        if not self.config.enabled or active_candidate_count <= 1 or maximum <= 1:
            return 1.0
        pressure = min(1.0, math.log1p(active_candidate_count - 1) / math.log(maximum))
        return 1.0 - self.config.max_active_context_penalty * pressure


def policy_for_strength(strength: HotwordStrength) -> ContextPolicy:
    return {
        HotwordStrength.CONSERVATIVE: ContextPolicy.conservative,
        HotwordStrength.BALANCED: ContextPolicy.balanced,
        HotwordStrength.AGGRESSIVE: ContextPolicy.aggressive,
        HotwordStrength.LOW_FALSE_ACTIVATION: ContextPolicy.low_false_activation,
    }[strength]()


def _require_finite_positive(name: str, value: float) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{name} must be a number")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


def _require_finite_non_negative(name: str, value: float) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{name} must be a number")
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and non-negative")


def _require_finite_range(
    name: str,
    value: float,
    *,
    lower: float,
    upper: float,
    lower_inclusive: bool = True,
    upper_inclusive: bool = True,
) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{name} must be a number")
    lower_valid = value >= lower if lower_inclusive else value > lower
    upper_valid = value <= upper if upper_inclusive else value < upper
    if not math.isfinite(value) or not lower_valid or not upper_valid:
        raise ValueError(f"{name} is outside its valid finite range")
