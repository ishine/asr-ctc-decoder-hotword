#include "decoder_state.h"

#include <algorithm>
#include <cmath>

namespace asr_decoder::internal {

DecoderState::DecoderState(DecoderConfig value)
    : config(std::move(value)),
      policy(policy_for(config.hotword_strength)),
      graph(config.context_token_ids, policy) {
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
  reset();
}

void DecoderState::reset() {
  processed_frames = 0;
  vocabulary_size = 0;
  beam_size = 0;
  token_beam_size = 0;
  error.clear();
  hypotheses.clear();
  PrefixScore initial;
  initial.blank = 0.0;
  initial.viterbi_blank = 0.0;
  hypotheses.push_back({{}, std::move(initial)});
}

}  // namespace asr_decoder::internal
