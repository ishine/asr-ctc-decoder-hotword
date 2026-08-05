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

from collections.abc import Iterable
from typing import NamedTuple, Optional

from .context_graph import ContextState
from .utils import log_add


class AlignmentNode(NamedTuple):
    """One immutable token node in a persistent Viterbi alignment history."""

    parent: Optional["AlignmentNode"]
    token_id: int
    start_frame: int
    end_frame: int
    log_probability: float
    length: int

    def append(self, token_id: int, frame: int, log_probability: float) -> "AlignmentNode":
        return AlignmentNode(self, token_id, frame, frame + 1, log_probability, self.length + 1)

    def extend(self, end_frame: int, log_probability: float) -> "AlignmentNode":
        """Return an O(1) replacement for the current token without copying its ancestors."""
        return AlignmentNode(
            self.parent,
            self.token_id,
            self.start_frame,
            end_frame,
            max(self.log_probability, log_probability),
            self.length,
        )


def append_alignment(
    alignment: Optional[AlignmentNode],
    token_id: int,
    frame: int,
    log_probability: float,
) -> AlignmentNode:
    if alignment is None:
        return AlignmentNode(None, token_id, frame, frame + 1, log_probability, 1)
    return alignment.append(token_id, frame, log_probability)


def materialize_alignment(
    alignment: Optional[AlignmentNode],
) -> tuple[list[int], list[tuple[int, int]], list[float]]:
    """Expand one selected persistent history in O(number of output tokens)."""
    if alignment is None:
        return [], [], []
    tokens = [0] * alignment.length
    spans = [(0, 0)] * alignment.length
    log_probabilities = [0.0] * alignment.length
    node = alignment
    index = alignment.length - 1
    while node is not None:
        tokens[index] = node.token_id
        spans[index] = (node.start_frame, node.end_frame)
        log_probabilities[index] = node.log_probability
        node = node.parent
        index -= 1
    return tokens, spans, log_probabilities


class ContextScoreState:
    """Persistent online weighted-interval DP for completed contexts."""

    __slots__ = ("parent", "jumps", "length", "best_score")

    def __init__(
        self,
        parent: Optional["ContextScoreState"] = None,
        best_score: float = 0.0,
    ) -> None:
        self.parent = parent
        self.length = 0 if parent is None else parent.length + 1
        self.best_score = best_score
        if parent is None:
            self.jumps = ()
        else:
            jumps = [parent]
            level = 1
            while level - 1 < len(jumps[level - 1].jumps):
                jumps.append(jumps[level - 1].jumps[level - 1])
                level += 1
            self.jumps = tuple(jumps)

    def _ancestor(self, distance: int) -> tuple["ContextScoreState", int]:
        """Return an ancestor and the number of binary-lifting hops."""
        if distance < 0 or distance > self.length:
            raise ValueError("ancestor distance is outside the context history")
        node = self
        level = 0
        hops = 0
        while distance:
            if distance & 1:
                node = node.jumps[level]
                hops += 1
            distance >>= 1
            level += 1
        return node, hops

    def advance(self, matches: Iterable[tuple[int, float]]) -> "ContextScoreState":
        best_score = self.best_score
        for token_count, bonus in matches:
            predecessor, _ = self._ancestor(token_count - 1)
            best_score = max(best_score, predecessor.best_score + bonus)
        return ContextScoreState(self, best_score)


class PrefixScore:
    """Acoustic, alignment, and contextual state for one CTC prefix."""

    __slots__ = (
        "blank_score",
        "non_blank_score",
        "viterbi_blank_score",
        "viterbi_non_blank_score",
        "blank_alignment",
        "non_blank_alignment",
        "context_state",
        "context_score_state",
        "has_context",
    )

    def __init__(
        self,
        blank_score: float = float("-inf"),
        non_blank_score: float = float("-inf"),
        viterbi_blank_score: float = float("-inf"),
        viterbi_non_blank_score: float = float("-inf"),
        context_state: Optional[ContextState] = None,
        context_score_state: Optional[ContextScoreState] = None,
    ):
        self.blank_score = blank_score
        self.non_blank_score = non_blank_score
        self.viterbi_blank_score = viterbi_blank_score
        self.viterbi_non_blank_score = viterbi_non_blank_score
        self.blank_alignment = None
        self.non_blank_alignment = None
        self.context_state = context_state
        self.context_score_state = context_score_state or ContextScoreState()
        self.has_context = False

    def acoustic_score(self):
        return log_add(self.blank_score, self.non_blank_score)

    def viterbi_score(self):
        return max(self.viterbi_blank_score, self.viterbi_non_blank_score)

    def alignment(self) -> Optional[AlignmentNode]:
        """Select the history from the same Viterbi ending state as the score."""
        return (
            self.blank_alignment
            if self.viterbi_blank_score > self.viterbi_non_blank_score
            else self.non_blank_alignment
        )

    def contextual_score(self):
        return self.context_score_state.best_score

    def total_score(self):
        return self.acoustic_score() + self.contextual_score()
