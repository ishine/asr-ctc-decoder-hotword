# Copyright (c) 2023 Xiaomi Corp. (authors: Wei Kang)
#               2023 Kaixun Huang
#               2023 Chengdong Liang (liangchengdong@mail.nwpu.edu.cn)
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

from collections import deque
from dataclasses import dataclass
from operator import index
from typing import Optional

from .context_policy import ContextPolicy
from .utils import tokenize


@dataclass(frozen=True)
class ActiveContextCandidates:
    tokens: tuple[int, ...]
    competing_count: int
    truncated: bool


class ContextState:
    """One immutable-topology state in the contextual automaton."""

    __slots__ = (
        "node_id",
        "token",
        "transitions",
        "failure",
        "own_match",
        "output",
    )

    def __init__(self, node_id: int, token: int = -1):
        self.node_id = node_id
        self.token = token
        self.transitions = {}
        self.failure = None
        self.own_match = None
        self.output = None


class ContextGraph:
    """A compact Aho-Corasick graph over model token IDs."""

    def __init__(
        self,
        contexts: list[str],
        symbol_table: dict[str, int],
        bpe_model: Optional[str] = None,
        policy: Optional[ContextPolicy] = None,
    ):
        self._initialize(tokenize(contexts, symbol_table, bpe_model), policy)

    @classmethod
    def from_token_ids(
        cls,
        context_token_ids: list[list[int]],
        *,
        policy: Optional[ContextPolicy] = None,
    ) -> "ContextGraph":
        """Build a graph from token sequences produced by the ASR tokenizer."""
        graph = cls.__new__(cls)
        graph._initialize(context_token_ids, policy)
        return graph

    def _initialize(
        self,
        context_token_ids: list[list[int]],
        policy: Optional[ContextPolicy],
    ) -> None:
        self.policy = policy or ContextPolicy.conservative()
        validated_token_ids = self._validate_token_ids(context_token_ids)
        self.context_count = len(validated_token_ids)
        self.num_nodes = 1
        self.root = ContextState(0)
        self.root.failure = self.root
        self.states = [self.root]
        self._build_graph(validated_token_ids)

    @staticmethod
    def _validate_token_ids(context_token_ids: list[list[int]]) -> list[list[int]]:
        validated = []
        seen = set()
        for phrase in context_token_ids:
            tokens = []
            for token in phrase:
                if isinstance(token, bool):
                    raise TypeError("context token IDs must be integers")
                try:
                    token_id = index(token)
                except TypeError as error:
                    raise TypeError("context token IDs must be integers") from error
                if token_id < 0:
                    raise ValueError("context token IDs must be non-negative")
                tokens.append(token_id)
            key = tuple(tokens)
            if tokens and key not in seen:
                validated.append(tokens)
                seen.add(key)
        return validated

    def _build_graph(self, token_ids: list[list[int]]) -> None:
        for tokens in token_ids:
            node = self.root
            for token in tokens:
                if token not in node.transitions:
                    node.transitions[token] = ContextState(self.num_nodes, token)
                    self.num_nodes += 1
                    self.states.append(node.transitions[token])
                node = node.transitions[token]
            node.own_match = (len(tokens), self.policy.completion_bonus)

        self._fill_failure_and_matches()
        self.token_ids = frozenset(state.token for state in self.states if state is not self.root)
        self.max_token_id = max(self.token_ids, default=-1)
        self.transition_count = sum(len(state.transitions) for state in self.states)

    def _fill_failure_and_matches(self) -> None:
        queue = deque()
        for node in self.root.transitions.values():
            node.failure = self.root
            queue.append(node)
        while queue:
            current = queue.popleft()
            for token, node in current.transitions.items():
                failure = current.failure
                while failure is not self.root and token not in failure.transitions:
                    failure = failure.failure
                if token in failure.transitions:
                    failure = failure.transitions[token]
                node.failure = failure
                node.output = failure if failure.own_match is not None else failure.output
                queue.append(node)

    def forward_one_step(
        self,
        state: ContextState,
        token: int,
    ) -> ContextState:
        """Advance one token in the automaton."""
        while state is not self.root and token not in state.transitions:
            state = state.failure
        if token in state.transitions:
            state = state.transitions[token]
        return state

    def completed_matches(self, state: ContextState):
        """Yield longest completions first, bounded by the policy limit."""
        remaining = self.policy.max_completed_contexts_per_token
        if state.own_match is not None:
            yield state.own_match
            remaining -= 1
            if remaining == 0:
                return
        state = state.output
        while state is not None and remaining:
            yield state.own_match
            remaining -= 1
            state = state.output

    def active_candidates(self, state: ContextState, maximum: int) -> ActiveContextCandidates:
        """Return a bounded, lazily derived set of useful continuation tokens.

        Root transitions are injected only when the complete root branching
        fits. For a large dictionary, new phrase starts must first be supported
        by the ordinary acoustic top-k; continuations from active non-root
        states can still be injected.
        """
        if maximum <= 0:
            return ActiveContextCandidates((), 0, bool(state.transitions))
        candidates = []
        seen = set()
        competing_count = 0
        current = state
        while True:
            transitions = current.transitions
            competing_count = min(maximum + 1, competing_count + len(transitions))
            if current is self.root and len(transitions) > maximum - len(candidates):
                return ActiveContextCandidates(tuple(candidates), competing_count, True)
            for token in transitions:
                if token not in seen:
                    candidates.append(token)
                    seen.add(token)
                    if len(candidates) == maximum:
                        truncated = any(token not in seen for token in transitions) or current is not self.root
                        return ActiveContextCandidates(tuple(candidates), competing_count, truncated)
            if current is self.root:
                break
            current = current.failure
        return ActiveContextCandidates(tuple(candidates), competing_count, False)
