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

"""Prefix beam search implementation used by :class:`CTCDecoder`.

The decoder facade owns stream configuration and validation.  This module
contains the search-specific mechanics so candidate selection, one-frame CTC
updates, and response materialization can be tested and changed independently.
"""

import math
from collections import defaultdict
from dataclasses import dataclass
from operator import index
from typing import Optional

import torch

from .prefix_score import PrefixScore, append_alignment, materialize_alignment
from .utils import log_add


@dataclass(frozen=True)
class _SearchConfig:
    beam_size: int
    token_beam_size: int
    token_prune_threshold: float
    vocabulary_size: int


@dataclass
class _ContextCandidates:
    """Context tokens collected from the distinct active automaton states."""

    tokens: list[int]
    root_tokens: set[int]
    non_root_tokens: set[int]
    competing_count: int


class PrefixBeamSearchMixin:
    """Search behavior mixed into ``CTCDecoder`` without owning its config."""

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
        try:
            beam_size = None if beam_size is None else index(beam_size)
            token_beam_size = None if token_beam_size is None else index(token_beam_size)
        except TypeError as error:
            raise TypeError("beam sizes must be integers") from error
        if beam_size is not None and beam_size != self._search_config.beam_size:
            raise ValueError("beam_size cannot change within a stream")
        requested_token_beam = None if token_beam_size is None else min(token_beam_size, vocabulary_size)
        if requested_token_beam is not None and requested_token_beam != self._search_config.token_beam_size:
            raise ValueError("token_beam_size cannot change within a stream")
        return self._search_config

    def _prepare_prefix_chunk(self, log_probabilities, beam_size, token_beam_size):
        """Validate and copy the small acoustic top-k needed by the CPU search."""
        self._validate_log_probabilities(log_probabilities)
        search_config = self._resolve_search_config(beam_size, token_beam_size, log_probabilities.size(1))
        top_values, top_indices = log_probabilities.topk(search_config.token_beam_size, dim=-1)
        frame_values = top_values.detach().cpu().tolist()
        frame_indices = top_indices.detach().cpu().tolist()
        blank_values = log_probabilities[:, self.blank_id].detach().cpu().tolist()
        if any(not math.isfinite(values[0]) for values in frame_values):
            raise ValueError("every frame must contain at least one finite log probability")

        gating_top_values = None
        if self.context_graph is not None and self.context_policy.gating.enabled:
            gating_width = min(8, log_probabilities.size(1))
            if search_config.token_beam_size >= gating_width:
                gating_top_values = [values[:gating_width] for values in frame_values]
            else:
                gating_top_values = log_probabilities.topk(gating_width, dim=-1).values.detach().cpu().tolist()
        return search_config, frame_values, frame_indices, blank_values, gating_top_values

    def _frame_gating(self, top_values, blank_probability):
        if top_values is None:
            return 1.0, 1.0
        stats = self.context_gate.analyze_frame(top_values, blank_probability)
        self.gating_acoustic_scale_sum += stats.acoustic_scale
        self.gating_frames += 1
        return stats.acoustic_scale, stats.confidence_factor

    def _collect_context_candidates(self) -> _ContextCandidates:
        maximum = self.context_policy.max_injected_candidates
        if self.context_graph is None or not maximum:
            return _ContextCandidates([], set(), set(), 0)

        tokens = []
        token_set = set()
        root_tokens = set()
        non_root_tokens = set()
        competing_count = 0
        truncated = False
        seen_states = set()
        for _, prefix_score in self.hypotheses:
            context_state = prefix_score.context_state
            if context_state.node_id in seen_states:
                continue
            seen_states.add(context_state.node_id)
            active = self.context_graph.active_candidates(context_state, maximum)
            competing_count = min(maximum + 1, competing_count + active.competing_count)
            truncated = truncated or active.truncated
            target = root_tokens if context_state is self.context_graph.root else non_root_tokens
            target.update(active.tokens)
            for token_id in active.tokens:
                if token_id in token_set:
                    continue
                if len(tokens) == maximum:
                    truncated = True
                    break
                tokens.append(token_id)
                token_set.add(token_id)

        if truncated:
            self.context_candidate_limit_hits += 1
        self.max_context_candidates_per_frame = max(self.max_context_candidates_per_frame, len(tokens))
        self.evaluated_context_candidates += len(tokens)
        return _ContextCandidates(tokens, root_tokens, non_root_tokens, competing_count)

    def _apply_word_boundary_guard(self, context_candidates, values, indices, best_probability, threshold):
        """Suppress an in-word continuation when acoustics prefer a word start."""
        if not self.word_boundary_token_ids:
            return set()
        boundary_best = max(
            (value for value, token_id in zip(values, indices) if token_id in self.word_boundary_token_ids),
            default=float("-inf"),
        )
        if boundary_best < best_probability - threshold:
            return set()

        blocked = (
            (context_candidates.non_root_tokens - context_candidates.root_tokens)
            - self.word_boundary_token_ids
            - set(indices)
        )
        if blocked:
            context_candidates.tokens[:] = [
                token_id for token_id in context_candidates.tokens if token_id not in blocked
            ]
            self.boundary_suppressed_context_candidates += len(blocked)
        return blocked

    def _frame_candidates(
        self,
        log_probabilities,
        frame_offset,
        values,
        indices,
        blank_probability,
        search_config,
        context_candidates,
        context_threshold,
    ):
        best_probability = values[0]
        blocked = self._apply_word_boundary_guard(
            context_candidates,
            values,
            indices,
            best_probability,
            context_threshold,
        )
        candidate_scores = dict(zip(indices, values))
        candidate_scores[self.blank_id] = blank_probability
        for token_id in blocked:
            candidate_scores.pop(token_id, None)

        missing_tokens = [token_id for token_id in context_candidates.tokens if token_id not in candidate_scores]
        if missing_tokens:
            self.gathered_context_candidates += len(missing_tokens)
            context_indices = torch.tensor(missing_tokens, device=log_probabilities.device)
            context_values = log_probabilities[frame_offset].index_select(0, context_indices).detach().cpu().tolist()
            for token_id, probability in zip(missing_tokens, context_values):
                if probability >= best_probability - context_threshold:
                    candidate_scores[token_id] = probability

        return [
            (probability, token_id)
            for probability, token_id in sorted(
                ((probability, token_id) for token_id, probability in candidate_scores.items()),
                reverse=True,
            )
            if probability >= best_probability - search_config.token_prune_threshold or token_id == self.blank_id
        ]

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

    def _update_blank(self, next_hypotheses, prefix, prefix_score, probability):
        next_score = next_hypotheses[prefix]
        next_score.blank_score = log_add(next_score.blank_score, prefix_score.acoustic_score() + probability)
        viterbi = prefix_score.viterbi_score() + probability
        if viterbi > next_score.viterbi_blank_score:
            next_score.viterbi_blank_score = viterbi
            next_score.blank_alignment = prefix_score.alignment()
        self._copy_context_state(prefix_score, next_score)

    def _update_repeated_token(self, next_hypotheses, prefix, prefix_score, token_id, probability):
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

    def _update_extended_token(self, next_hypotheses, prefix, prefix_score, token_id, probability):
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

    def _advance_prefix_frame(self, candidates, beam_size):
        next_hypotheses = defaultdict(PrefixScore)
        for probability, token_id in candidates:
            for prefix, prefix_score in self.hypotheses:
                if token_id == self.blank_id:
                    self._update_blank(next_hypotheses, prefix, prefix_score, probability)
                elif prefix and token_id == prefix[-1]:
                    self._update_repeated_token(next_hypotheses, prefix, prefix_score, token_id, probability)
                else:
                    self._update_extended_token(next_hypotheses, prefix, prefix_score, token_id, probability)

        ranked = sorted(
            (hypothesis for hypothesis in next_hypotheses.items() if math.isfinite(hypothesis[1].acoustic_score())),
            key=lambda hypothesis: hypothesis[1].total_score(),
            reverse=True,
        )
        self.hypotheses = ranked[:beam_size]

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

    def _prefix_response(self, search_config, return_token_probabilities, return_scores, return_gating_diagnostics):
        hypotheses = sorted(self.hypotheses, key=lambda hypothesis: hypothesis[1].total_score(), reverse=True)
        alignments = [materialize_alignment(score.alignment()) for _, score in hypotheses]
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
            response["gating"] = self._gating_diagnostics(search_config)
        return response

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
        prepared = self._prepare_prefix_chunk(log_probabilities, beam_size, token_beam_size)
        search_config, frame_values, frame_indices, blank_values, gating_top_values = prepared
        self._lock_decode_mode("prefix_beam")
        if self._search_config is None:
            self._search_config = search_config

        for frame_offset, (values, indices, blank_probability) in enumerate(
            zip(frame_values, frame_indices, blank_values)
        ):
            self.processed_frames += 1
            top_for_gating = None if gating_top_values is None else gating_top_values[frame_offset]
            acoustic_scale, confidence_factor = self._frame_gating(top_for_gating, blank_probability)
            context_candidates = self._collect_context_candidates()
            active_factor = self.context_gate.active_context_factor(
                context_candidates.competing_count,
                self.context_policy.max_injected_candidates,
            )
            gating_factor = acoustic_scale * confidence_factor * active_factor
            if gating_top_values is not None and blank_probability < values[0]:
                self.gating_factor_sum += gating_factor
                self.nonblank_gating_frames += 1
            context_threshold = self.context_policy.context_token_prune_threshold * gating_factor
            candidates = self._frame_candidates(
                log_probabilities,
                frame_offset,
                values,
                indices,
                blank_probability,
                search_config,
                context_candidates,
                context_threshold,
            )
            self._advance_prefix_frame(candidates, search_config.beam_size)

        response = self._prefix_response(
            search_config,
            return_token_probabilities,
            return_scores,
            return_gating_diagnostics,
        )
        if finalize:
            self.reset()
        return response
