#pragma once

#include <string>
#include <utility>
#include <vector>

#include "asr_decoder/ctc_decoder.h"
#include "context_graph.h"
#include "prefix_score.h"

namespace asr_decoder::internal {

struct DecoderState {
  explicit DecoderState(DecoderConfig value);
  void reset();

  DecoderConfig config;
  Policy policy;
  ContextGraph graph;
  std::string config_error;
  std::string error;
  size_t processed_frames = 0;
  size_t vocabulary_size = 0;
  size_t beam_size = 0;
  size_t token_beam_size = 0;
  std::vector<std::pair<std::vector<int>, PrefixScore>> hypotheses;
};

}  // namespace asr_decoder::internal
