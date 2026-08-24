#include "greedy_search.h"

#include <algorithm>
#include <cmath>

namespace asr_decoder::internal {
namespace {

void consume_frame(DecoderState& state, int token, double probability) {
  ++state.processed_frames;
  if (token == state.config.blank_id) {
    state.last_greedy_token = -1;
    return;
  }
  if (token == state.last_greedy_token) {
    state.greedy_spans.back().second = state.processed_frames;
    state.greedy_token_log_probabilities.back() =
        std::max(state.greedy_token_log_probabilities.back(), probability);
    return;
  }
  state.greedy_tokens.push_back(token);
  state.greedy_spans.push_back(
      {state.processed_frames - 1, state.processed_frames});
  state.greedy_token_log_probabilities.push_back(probability);
  state.last_greedy_token = token;
}

DecodeResult build_result(const DecoderState& state, GreedyOptions options) {
  DecodeResult result;
  result.tokens.push_back(state.greedy_tokens);
  std::vector<Timestamp> timestamps;
  timestamps.reserve(state.greedy_spans.size());
  for (const auto& [start, end] : state.greedy_spans) {
    Timestamp timestamp{start, end};
    if (state.config.frame_shift_ms > 0.0) {
      timestamp.start_ms = start * state.config.frame_shift_ms;
      timestamp.end_ms = end * state.config.frame_shift_ms;
    }
    timestamps.push_back(timestamp);
  }
  result.timestamps.push_back(std::move(timestamps));
  if (options.return_token_probabilities) {
    std::vector<double> probabilities;
    probabilities.reserve(state.greedy_token_log_probabilities.size());
    for (double probability : state.greedy_token_log_probabilities) {
      probabilities.push_back(std::exp(probability));
    }
    result.probabilities.push_back(std::move(probabilities));
  }
  return result;
}

}  // namespace

DecodeResult run_greedy_search(DecoderState& state, const float* values,
                               size_t frames, size_t vocabulary_size,
                               GreedyOptions options) {
  DecodeResult result;
  if (!state.config_error.empty()) return result;
  if (!validate_input(state, values, frames, vocabulary_size)) return result;
  if (!lock_decode_mode(state, DecodeMode::kGreedy)) return result;

  for (size_t frame = 0; frame < frames; ++frame) {
    const float* row = values + frame * vocabulary_size;
    size_t token = 0;
    for (size_t candidate = 1; candidate < vocabulary_size; ++candidate) {
      if (row[candidate] > row[token]) token = candidate;
    }
    consume_frame(state, static_cast<int>(token), row[token]);
  }
  result = build_result(state, options);
  if (options.finalize) state.reset();
  return result;
}

}  // namespace asr_decoder::internal
