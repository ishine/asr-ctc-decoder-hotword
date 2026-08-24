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

"""Greedy CTC search implementation used by :class:`CTCDecoder`."""

import math

import torch


class GreedySearchMixin:
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
            self._consume_greedy_frame(score, token_id)
        result = self._greedy_response(return_token_probabilities)
        if finalize:
            self.reset()
        return result

    def _consume_greedy_frame(self, score, token_id) -> None:
        self.processed_frames += 1
        if token_id == self.blank_id:
            self.last_greedy_token = None
            return
        if token_id == self.last_greedy_token:
            self.greedy_spans[-1] = (self.greedy_spans[-1][0], self.processed_frames)
            self.greedy_token_log_probabilities[-1] = max(
                self.greedy_token_log_probabilities[-1],
                score,
            )
            return
        self.greedy_tokens.append(token_id)
        self.greedy_spans.append((self.processed_frames - 1, self.processed_frames))
        self.greedy_token_log_probabilities.append(score)
        self.last_greedy_token = token_id

    def _greedy_response(self, return_token_probabilities):
        result = {
            "tokens": self.greedy_tokens.copy(),
            "timestamps": self._timestamps(self.greedy_spans),
        }
        if return_token_probabilities:
            result["probs"] = [math.exp(score) for score in self.greedy_token_log_probabilities]
        return result
