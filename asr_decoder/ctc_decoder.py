# Copyright (c) 2021 Mobvoi Inc. (authors: Binbin Zhang)
#               2023 Tsinghua Univ. (authors: Xingchen Song)
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
from collections.abc import Collection
from numbers import Real
from operator import index
from typing import Optional, Union

import torch

from .context_graph import ContextGraph
from .context_policy import AdaptiveContextGate, ContextPolicy, DecodingQuality, HotwordStrength, policy_for_strength
from .greedy_search import GreedySearchMixin
from .prefix_beam_search import PrefixBeamSearchMixin
from .prefix_score import PrefixScore


class CTCDecoder(GreedySearchMixin, PrefixBeamSearchMixin):
    def __init__(
        self,
        contexts: Optional[list[str]] = None,
        symbol_table: Optional[dict[str, int]] = None,
        bpe_model: Optional[str] = None,
        context_token_ids: Optional[list[list[int]]] = None,
        context_policy: Optional[ContextPolicy] = None,
        blank_id: int = 0,
        hotword_strength: Optional[Union[HotwordStrength, str]] = None,
        decoding_quality: Union[DecodingQuality, str] = DecodingQuality.BALANCED,
        frame_shift_ms: Optional[float] = None,
        word_boundary_token_ids: Optional[Collection[int]] = None,
    ):
        if context_policy is not None and hotword_strength is not None:
            raise ValueError("provide context_policy or hotword_strength, not both")
        if context_policy is not None and not isinstance(context_policy, ContextPolicy):
            raise TypeError("context_policy must be a ContextPolicy")
        self.hotword_strength = None if context_policy is not None else HotwordStrength(hotword_strength or "balanced")
        self.decoding_quality = DecodingQuality(decoding_quality)
        self.context_policy = context_policy or policy_for_strength(self.hotword_strength)
        self.context_graph = None
        validated_context_graph = None
        if contexts is not None and context_token_ids is not None:
            raise ValueError("provide contexts or context_token_ids, not both")
        if isinstance(blank_id, bool):
            raise TypeError("blank_id must be an integer")
        try:
            blank_id = index(blank_id)
        except TypeError as error:
            raise TypeError("blank_id must be an integer") from error
        if blank_id < 0:
            raise ValueError("blank_id must be non-negative")
        if isinstance(frame_shift_ms, bool) or (frame_shift_ms is not None and not isinstance(frame_shift_ms, Real)):
            raise TypeError("frame_shift_ms must be a real number")
        if frame_shift_ms is not None and (not math.isfinite(frame_shift_ms) or frame_shift_ms <= 0):
            raise ValueError("frame_shift_ms must be finite and greater than zero")
        if word_boundary_token_ids is None:
            word_boundary_token_ids = frozenset()
        else:
            try:
                boundary_ids = []
                for token_id in word_boundary_token_ids:
                    if isinstance(token_id, bool):
                        raise TypeError
                    boundary_ids.append(index(token_id))
                word_boundary_token_ids = frozenset(boundary_ids)
            except TypeError as error:
                raise TypeError("word_boundary_token_ids must contain integers") from error
            if any(token_id < 0 for token_id in word_boundary_token_ids):
                raise ValueError("word_boundary_token_ids must be non-negative")
        if blank_id in word_boundary_token_ids:
            raise ValueError("word_boundary_token_ids must not include blank_id")
        if context_token_ids:
            validated_context_graph = ContextGraph.from_token_ids(context_token_ids, policy=self.context_policy)
        elif contexts:
            if symbol_table is None:
                raise ValueError("symbol_table is required when contexts are provided")
            validated_context_graph = ContextGraph(contexts, symbol_table, bpe_model, self.context_policy)
        if validated_context_graph is not None and blank_id in validated_context_graph.token_ids:
            raise ValueError("context token IDs must not contain blank_id")
        self._context_max_token_id = -1 if validated_context_graph is None else validated_context_graph.max_token_id
        if (
            validated_context_graph is not None
            and validated_context_graph.num_nodes > 1
            and self.context_policy.completion_bonus > 0
        ):
            self.context_graph = validated_context_graph
        self.blank_id = blank_id
        self._frame_shift_ms = None if frame_shift_ms is None else float(frame_shift_ms)
        self.word_boundary_token_ids = word_boundary_token_ids
        self.context_gate = AdaptiveContextGate(self.context_policy.gating)
        self.reset()

    def reset(self) -> None:
        self.processed_frames = 0
        self.gating_acoustic_scale_sum = 0.0
        self.gating_factor_sum = 0.0
        self.gating_frames = 0
        self.nonblank_gating_frames = 0
        self.evaluated_context_candidates = 0
        self.gathered_context_candidates = 0
        self.max_context_candidates_per_frame = 0
        self.context_candidate_limit_hits = 0
        self.boundary_suppressed_context_candidates = 0
        self._search_config = None
        self._decode_mode = None
        context_root = None if self.context_graph is None else self.context_graph.root
        self.greedy_tokens = []
        self.greedy_spans = []
        self.greedy_token_log_probabilities = []
        self.last_greedy_token = None
        self.hypotheses = [
            (
                tuple(),
                PrefixScore(
                    blank_score=0.0,
                    viterbi_blank_score=0.0,
                    context_state=context_root,
                ),
            )
        ]

    def create_stream(self):
        """Create an independent stream sharing only immutable policy and graph state."""
        stream = type(self).__new__(type(self))
        stream.context_graph = self.context_graph
        stream._context_max_token_id = self._context_max_token_id
        stream.blank_id = self.blank_id
        stream._frame_shift_ms = self.frame_shift_ms
        stream.word_boundary_token_ids = self.word_boundary_token_ids
        stream.hotword_strength = self.hotword_strength
        stream.decoding_quality = self.decoding_quality
        stream.context_policy = self.context_policy
        stream.context_gate = self.context_gate
        stream.reset()
        return stream

    @property
    def frame_shift_ms(self) -> Optional[float]:
        """Effective CTC output frame shift used for timestamp conversion."""
        return self._frame_shift_ms

    def _timestamps(self, spans):
        """Copy zero-based, half-open frame spans into the public timestamp schema."""
        return [
            {
                "start_frame": start,
                "end_frame": end,
                "start_ms": None if self.frame_shift_ms is None else start * self.frame_shift_ms,
                "end_ms": None if self.frame_shift_ms is None else end * self.frame_shift_ms,
            }
            for start, end in spans
        ]

    def _validate_log_probabilities(self, log_probabilities: torch.Tensor) -> None:
        if log_probabilities.ndim != 2:
            raise ValueError("log_probabilities must have shape (frames, vocabulary_size)")
        if not torch.is_floating_point(log_probabilities):
            raise TypeError("log_probabilities must use a floating-point dtype")
        vocabulary_size = log_probabilities.size(1)
        if vocabulary_size < 1:
            raise ValueError("vocabulary_size must be at least 1")
        if self.blank_id >= vocabulary_size:
            raise ValueError("blank_id must be smaller than vocabulary_size")
        if self._context_max_token_id >= vocabulary_size:
            raise ValueError("context token IDs must be smaller than vocabulary_size")

    def _lock_decode_mode(self, mode: str) -> None:
        if self._decode_mode is None:
            self._decode_mode = mode
        elif self._decode_mode != mode:
            raise ValueError(f"cannot switch from {self._decode_mode} to {mode} within a stream")
