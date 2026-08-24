#pragma once

#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "asr_decoder/ctc_decoder.h"
#include "context_graph.h"
#include "prefix_score.h"

namespace asr_decoder::internal {

enum class DecodeMode { kNone, kGreedy, kPrefixBeam };

struct SearchDiagnostics {
  double gating_acoustic_scale_sum = 0.0;
  double gating_factor_sum = 0.0;
  size_t gating_frames = 0;
  size_t nonblank_gating_frames = 0;
  size_t evaluated_context_candidates = 0;
  size_t gathered_context_candidates = 0;
  size_t max_context_candidates_per_frame = 0;
  size_t context_candidate_limit_hits = 0;
  size_t boundary_suppressed_context_candidates = 0;
};

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
  DecodeMode decode_mode = DecodeMode::kNone;
  SearchDiagnostics diagnostics;
  std::unordered_set<int> word_boundary_token_ids;
  std::vector<int> greedy_tokens;
  std::vector<std::pair<size_t, size_t>> greedy_spans;
  std::vector<double> greedy_token_log_probabilities;
  int last_greedy_token = -1;
  std::vector<std::pair<std::vector<int>, PrefixScore>> hypotheses;

  bool context_enabled() const {
    return !graph.empty() && policy.completion_bonus > 0.0;
  }
};

bool validate_input(DecoderState& state, const float* values, size_t frames,
                    size_t vocabulary_size);
bool lock_decode_mode(DecoderState& state, DecodeMode mode);

}  // namespace asr_decoder::internal
