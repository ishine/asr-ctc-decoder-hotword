#include "decoder_state.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace asr_decoder::internal {

DecoderState::DecoderState(DecoderConfig value)
    : config(std::move(value)),
      policy(
          config.context_policy.value_or(policy_for(config.hotword_strength))),
      graph(config.context_token_ids, policy) {
  if (config.context_policy &&
      config.hotword_strength != HotwordStrength::kBalanced) {
    config_error = "provide context_policy or hotword_strength, not both";
  }
  validate_policy(policy, config_error);
  if (config.blank_id < 0) config_error = "blank_id must be non-negative";
  if (!std::isfinite(config.frame_shift_ms) || config.frame_shift_ms < 0.0) {
    config_error = "frame_shift_ms must be finite and non-negative";
  }
  for (const auto& context : config.context_token_ids) {
    for (int token : context) {
      if (token < 0) {
        config_error = "hotword token IDs must be non-negative";
      } else if (token == config.blank_id) {
        config_error = "hotword token IDs must not contain blank_id";
      }
    }
  }
  for (int token : config.word_boundary_token_ids) {
    if (token < 0) {
      config_error = "word_boundary_token_ids must be non-negative";
    } else if (token == config.blank_id) {
      config_error = "word_boundary_token_ids must not include blank_id";
    } else {
      word_boundary_token_ids.insert(token);
    }
  }
  reset();
}

void DecoderState::reset() {
  processed_frames = 0;
  vocabulary_size = 0;
  beam_size = 0;
  token_beam_size = 0;
  decode_mode = DecodeMode::kNone;
  diagnostics = {};
  error.clear();
  greedy_tokens.clear();
  greedy_spans.clear();
  greedy_token_log_probabilities.clear();
  last_greedy_token = -1;
  hypotheses.clear();
  PrefixScore initial;
  initial.blank = 0.0;
  initial.viterbi_blank = 0.0;
  hypotheses.push_back({{}, std::move(initial)});
}

bool validate_input(DecoderState& state, const float* values, size_t frames,
                    size_t vocabulary_size) {
  state.error.clear();
  if ((!values && frames) || vocabulary_size == 0 ||
      static_cast<size_t>(state.config.blank_id) >= vocabulary_size ||
      (!state.graph.empty() &&
       static_cast<size_t>(state.graph.maximum_token()) >= vocabulary_size)) {
    state.error = "invalid log-probability matrix or token ID";
    return false;
  }
  for (size_t index = 0; index < frames * vocabulary_size; ++index) {
    if (std::isnan(values[index]) ||
        values[index] == std::numeric_limits<float>::infinity()) {
      state.error =
          "log probabilities must not contain NaN or positive infinity";
      return false;
    }
  }
  for (size_t frame = 0; frame < frames; ++frame) {
    bool has_finite_value = false;
    for (size_t token = 0; token < vocabulary_size; ++token) {
      has_finite_value |=
          std::isfinite(values[frame * vocabulary_size + token]);
    }
    if (!has_finite_value) {
      state.error = "every frame must contain a finite log probability";
      return false;
    }
  }
  return true;
}

bool lock_decode_mode(DecoderState& state, DecodeMode mode) {
  if (state.decode_mode == DecodeMode::kNone) {
    state.decode_mode = mode;
    return true;
  }
  if (state.decode_mode == mode) return true;
  state.error =
      mode == DecodeMode::kGreedy
          ? "cannot switch from prefix_beam to greedy within a stream"
          : "cannot switch from greedy to prefix_beam within a stream";
  return false;
}

}  // namespace asr_decoder::internal
