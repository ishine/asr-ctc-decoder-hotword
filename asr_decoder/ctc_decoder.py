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
from collections import defaultdict
from collections.abc import Collection
from dataclasses import dataclass
from numbers import Real
from operator import index
from typing import Optional, Union

import torch

from .context_graph import ContextGraph
from .context_policy import AdaptiveContextGate, ContextPolicy, DecodingQuality, HotwordStrength, policy_for_strength
from .prefix_score import PrefixScore, append_alignment, materialize_alignment
from .utils import log_add


@dataclass(frozen=True)
class _SearchConfig:
    beam_size: int
    token_beam_size: int
    token_prune_threshold: float
    vocabulary_size: int


class CTCDecoder:
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

    def _resolve_search_config(
        self,
        beam_size: Optional[int],
        token_beam_size: Optional[int],
        vocabulary_size: int,
    ) -> _SearchConfig:
        quality_beam_size, quality_token_beam_size, token_prune_threshold = self.decoding_quality.search_settings
        if self._search_config is None:
            try:
                beam_size = quality_beam_size if beam_size is None else index(beam_size)
                token_beam_size = quality_token_beam_size if token_beam_size is None else index(token_beam_size)
            except TypeError as error:
                raise TypeError("beam sizes must be integers") from error
            resolved = _SearchConfig(
                beam_size=beam_size,
                token_beam_size=min(token_beam_size, vocabulary_size),
                token_prune_threshold=token_prune_threshold,
                vocabulary_size=vocabulary_size,
            )
            if resolved.beam_size < 1:
                raise ValueError("beam_size must be at least 1")
            if resolved.token_beam_size < 1:
                raise ValueError("token_beam_size must be at least 1")
            return resolved
        if vocabulary_size != self._search_config.vocabulary_size:
            raise ValueError("vocabulary_size cannot change within a stream")
        if beam_size is not None:
            try:
                beam_size = index(beam_size)
            except TypeError as error:
                raise TypeError("beam sizes must be integers") from error
        if token_beam_size is not None:
            try:
                token_beam_size = index(token_beam_size)
            except TypeError as error:
                raise TypeError("beam sizes must be integers") from error
        if beam_size is not None and beam_size != self._search_config.beam_size:
            raise ValueError("beam_size cannot change within a stream")
        requested_token_beam = None if token_beam_size is None else min(token_beam_size, vocabulary_size)
        if requested_token_beam is not None and requested_token_beam != self._search_config.token_beam_size:
            raise ValueError("token_beam_size cannot change within a stream")
        return self._search_config

    def _lock_decode_mode(self, mode: str) -> None:
        if self._decode_mode is None:
            self._decode_mode = mode
        elif self._decode_mode != mode:
            raise ValueError(f"cannot switch from {self._decode_mode} to {mode} within a stream")

    def _copy_context_state(self, prefix_score, next_score) -> None:
        if self.context_graph is not None and not next_score.has_context:
            next_score.context_state = prefix_score.context_state
            next_score.context_score_state = prefix_score.context_score_state
            next_score.has_context = True

    def _advance_context(self, prefix_score, next_score, token_id) -> None:
        if self.context_graph is None:
            return
        context_state = self.context_graph.forward_one_step(prefix_score.context_state, token_id)
        context_score_state = prefix_score.context_score_state.advance(
            self.context_graph.completed_matches(context_state),
        )
        if not next_score.has_context or context_score_state.best_score > next_score.context_score_state.best_score:
            next_score.context_score_state = context_score_state
            next_score.context_state = context_state
            next_score.has_context = True

    def _gating_diagnostics(self, search_config: _SearchConfig) -> dict:
        enabled = self.context_graph is not None and self.context_policy.gating.enabled
        return {
            "contextual_biasing": self.context_graph is not None,
            "adaptive_context_gating": enabled,
            "gating_frames": self.gating_frames if enabled else 0,
            "mean_gating_acoustic_scale": (
                self.gating_acoustic_scale_sum / self.gating_frames if enabled and self.gating_frames else None
            ),
            "mean_gating_factor": (
                self.gating_factor_sum / self.nonblank_gating_frames
                if enabled and self.nonblank_gating_frames
                else None
            ),
            "nonblank_gating_frames": self.nonblank_gating_frames if enabled else 0,
            "evaluated_context_candidates": self.evaluated_context_candidates,
            "gathered_context_candidates": self.gathered_context_candidates,
            "max_context_candidates_per_frame": self.max_context_candidates_per_frame,
            "context_candidate_limit_hits": self.context_candidate_limit_hits,
            "word_boundary_aware_context": bool(self.word_boundary_token_ids),
            "boundary_suppressed_context_candidates": self.boundary_suppressed_context_candidates,
            "beam_size": search_config.beam_size,
            "token_beam_size": search_config.token_beam_size,
        }

    def greedy_search(
        self,
        log_probabilities: torch.Tensor,
        finalize: bool = False,
        return_token_probabilities: bool = False,
    ):
        """Decode with frame-wise argmax followed by standard CTC collapse."""
        self._validate_log_probabilities(log_probabilities)
        frame_scores, frame_tokens = log_probabilities.max(dim=-1)
        scores = frame_scores.detach().cpu().tolist()
        tokens = frame_tokens.detach().cpu().tolist()
        if any(not math.isfinite(score) for score in scores):
            raise ValueError("every frame must contain at least one finite log probability")
        self._lock_decode_mode("greedy")
        for score, token_id in zip(scores, tokens):
            self.processed_frames += 1
            if token_id == self.blank_id:
                self.last_greedy_token = None
                continue
            if token_id == self.last_greedy_token:
                self.greedy_spans[-1] = (self.greedy_spans[-1][0], self.processed_frames)
                self.greedy_token_log_probabilities[-1] = max(
                    self.greedy_token_log_probabilities[-1],
                    score,
                )
                continue
            self.greedy_tokens.append(token_id)
            self.greedy_spans.append((self.processed_frames - 1, self.processed_frames))
            self.greedy_token_log_probabilities.append(score)
            self.last_greedy_token = token_id
        result = {
            "tokens": self.greedy_tokens.copy(),
            "timestamps": self._timestamps(self.greedy_spans),
        }
        if return_token_probabilities:
            result["probs"] = [math.exp(score) for score in self.greedy_token_log_probabilities]
        if finalize:
            self.reset()
        return result

    def prefix_beam_search(
        self,
        log_probabilities: torch.Tensor,
        beam_size: Optional[int] = None,
        finalize: bool = False,
        return_token_probabilities: bool = False,
        token_beam_size: Optional[int] = None,
        return_scores: bool = False,
        return_gating_diagnostics: bool = False,
    ):
        """Decode a chunk of CTC log probabilities with prefix beam search."""
        self._validate_log_probabilities(log_probabilities)
        search_config = self._resolve_search_config(beam_size, token_beam_size, log_probabilities.size(1))
        top_values, top_indices = log_probabilities.topk(search_config.token_beam_size, dim=-1)
        frame_values = top_values.detach().cpu().tolist()
        frame_indices = top_indices.detach().cpu().tolist()
        blank_values = log_probabilities[:, self.blank_id].detach().cpu().tolist()
        if any(not math.isfinite(values[0]) for values in frame_values):
            raise ValueError("every frame must contain at least one finite log probability")
        self._lock_decode_mode("prefix_beam")
        if self._search_config is None:
            self._search_config = search_config

        gating_top_values = None
        if self.context_graph is not None and self.context_policy.gating.enabled:
            gating_width = min(8, log_probabilities.size(1))
            if search_config.token_beam_size >= gating_width:
                gating_top_values = [values[:gating_width] for values in frame_values]
            else:
                gating_top_values = log_probabilities.topk(gating_width, dim=-1).values.detach().cpu().tolist()

        for frame_offset, (values, indices, blank_prob) in enumerate(zip(frame_values, frame_indices, blank_values)):
            self.processed_frames += 1
            best_prob = values[0]
            if gating_top_values is None:
                acoustic_scale = confidence_factor = 1.0
            else:
                gating_stats = self.context_gate.analyze_frame(gating_top_values[frame_offset], blank_prob)
                acoustic_scale = gating_stats.acoustic_scale
                confidence_factor = gating_stats.confidence_factor
                self.gating_acoustic_scale_sum += acoustic_scale
                self.gating_frames += 1

            active_tokens = []
            active_token_set = set()
            root_context_tokens = set()
            non_root_context_tokens = set()
            competing_count = 0
            candidate_limit_hit = False
            maximum = self.context_policy.max_injected_candidates
            if self.context_graph is not None and maximum:
                seen_context_states = set()
                for _, prefix_score in self.hypotheses:
                    context_state = prefix_score.context_state
                    state_id = context_state.node_id
                    if state_id in seen_context_states:
                        continue
                    seen_context_states.add(state_id)
                    active = self.context_graph.active_candidates(context_state, maximum)
                    competing_count = min(maximum + 1, competing_count + active.competing_count)
                    candidate_limit_hit = candidate_limit_hit or active.truncated
                    for token_id in active.tokens:
                        if context_state is self.context_graph.root:
                            root_context_tokens.add(token_id)
                        else:
                            non_root_context_tokens.add(token_id)
                        if token_id in active_token_set:
                            continue
                        if len(active_tokens) == maximum:
                            candidate_limit_hit = True
                            break
                        active_tokens.append(token_id)
                        active_token_set.add(token_id)
            if candidate_limit_hit:
                self.context_candidate_limit_hits += 1
            self.max_context_candidates_per_frame = max(self.max_context_candidates_per_frame, len(active_tokens))
            self.evaluated_context_candidates += len(active_tokens)

            active_factor = self.context_gate.active_context_factor(competing_count, maximum)
            gating_factor = acoustic_scale * confidence_factor * active_factor
            if gating_top_values is not None and blank_prob < best_prob:
                self.gating_factor_sum += gating_factor
                self.nonblank_gating_frames += 1
            token_threshold = search_config.token_prune_threshold
            context_threshold = self.context_policy.context_token_prune_threshold * gating_factor

            boundary_dominant = False
            blocked_context_tokens = set()
            if self.word_boundary_token_ids:
                boundary_best = max(
                    (value for value, token_id in zip(values, indices) if token_id in self.word_boundary_token_ids),
                    default=float("-inf"),
                )
                boundary_dominant = boundary_best >= best_prob - context_threshold
                if boundary_dominant:
                    blocked_context_tokens = (
                        (non_root_context_tokens - root_context_tokens) - self.word_boundary_token_ids - set(indices)
                    )
                    if blocked_context_tokens:
                        active_tokens = [
                            token_id for token_id in active_tokens if token_id not in blocked_context_tokens
                        ]
                        active_token_set.difference_update(blocked_context_tokens)
                        self.boundary_suppressed_context_candidates += len(blocked_context_tokens)

            candidate_scores = dict(zip(indices, values))
            candidate_scores[self.blank_id] = blank_prob
            for token_id in blocked_context_tokens:
                candidate_scores.pop(token_id, None)
            missing_tokens = [token_id for token_id in active_tokens if token_id not in candidate_scores]
            if missing_tokens:
                self.gathered_context_candidates += len(missing_tokens)
                context_indices = torch.tensor(missing_tokens, device=log_probabilities.device)
                context_values = (
                    log_probabilities[frame_offset].index_select(0, context_indices).detach().cpu().tolist()
                )
                for token_id, probability in zip(missing_tokens, context_values):
                    if probability >= best_prob - context_threshold:
                        candidate_scores[token_id] = probability

            candidates = sorted(
                ((probability, token_id) for token_id, probability in candidate_scores.items()),
                reverse=True,
            )
            next_hypotheses = defaultdict(PrefixScore)
            for probability, token_id in candidates:
                if probability < best_prob - token_threshold and token_id != self.blank_id:
                    continue
                for prefix, prefix_score in self.hypotheses:
                    last = prefix[-1] if prefix else None
                    if token_id == self.blank_id:
                        next_score = next_hypotheses[prefix]
                        next_score.blank_score = log_add(
                            next_score.blank_score,
                            prefix_score.acoustic_score() + probability,
                        )
                        viterbi = prefix_score.viterbi_score() + probability
                        if viterbi > next_score.viterbi_blank_score:
                            next_score.viterbi_blank_score = viterbi
                            next_score.blank_alignment = prefix_score.alignment()
                        self._copy_context_state(prefix_score, next_score)
                    elif token_id == last:
                        repeated = next_hypotheses[prefix]
                        repeated.non_blank_score = log_add(
                            repeated.non_blank_score,
                            prefix_score.non_blank_score + probability,
                        )
                        if repeated.viterbi_non_blank_score < prefix_score.viterbi_non_blank_score + probability:
                            repeated.viterbi_non_blank_score = prefix_score.viterbi_non_blank_score + probability
                            repeated.non_blank_alignment = prefix_score.non_blank_alignment.extend(
                                self.processed_frames,
                                probability,
                            )
                        self._copy_context_state(prefix_score, repeated)

                        extended_prefix = prefix + (token_id,)
                        after_blank = next_hypotheses[extended_prefix]
                        after_blank.non_blank_score = log_add(
                            after_blank.non_blank_score,
                            prefix_score.blank_score + probability,
                        )
                        if after_blank.viterbi_non_blank_score < prefix_score.viterbi_blank_score + probability:
                            after_blank.viterbi_non_blank_score = prefix_score.viterbi_blank_score + probability
                            after_blank.non_blank_alignment = append_alignment(
                                prefix_score.blank_alignment,
                                token_id,
                                self.processed_frames - 1,
                                probability,
                            )
                        self._advance_context(prefix_score, after_blank, token_id)
                    else:
                        extended_prefix = prefix + (token_id,)
                        next_score = next_hypotheses[extended_prefix]
                        next_score.non_blank_score = log_add(
                            next_score.non_blank_score,
                            prefix_score.acoustic_score() + probability,
                        )
                        if next_score.viterbi_non_blank_score < prefix_score.viterbi_score() + probability:
                            next_score.viterbi_non_blank_score = prefix_score.viterbi_score() + probability
                            next_score.non_blank_alignment = append_alignment(
                                prefix_score.alignment(),
                                token_id,
                                self.processed_frames - 1,
                                probability,
                            )
                        self._advance_context(prefix_score, next_score, token_id)

            ranked = sorted(
                (hypothesis for hypothesis in next_hypotheses.items() if math.isfinite(hypothesis[1].acoustic_score())),
                key=lambda hypothesis: hypothesis[1].total_score(),
                reverse=True,
            )
            self.hypotheses = ranked[: search_config.beam_size]

        hypotheses = sorted(self.hypotheses, key=lambda hypothesis: hypothesis[1].total_score(), reverse=True)
        alignments = [materialize_alignment(score.alignment()) for _, score in hypotheses]
        diagnostics = self._gating_diagnostics(search_config)
        if finalize:
            self.reset()
        response = {
            "tokens": [tokens for tokens, _, _ in alignments],
            "timestamps": [self._timestamps(spans) for _, spans, _ in alignments],
        }
        if return_token_probabilities:
            response["probs"] = [
                [math.exp(probability) for probability in log_probabilities] for _, _, log_probabilities in alignments
            ]
        if return_scores:
            response["scores"] = [
                {
                    "acoustic": score.acoustic_score(),
                    "contextual_bias": score.contextual_score(),
                    "total": score.total_score(),
                }
                for _, score in hypotheses
            ]
        if return_gating_diagnostics:
            response["gating"] = diagnostics
        return response
