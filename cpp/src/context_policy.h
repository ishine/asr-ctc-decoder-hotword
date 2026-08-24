#pragma once

#include <cstddef>
#include <utility>
#include <vector>

#include "asr_decoder/ctc_decoder.h"

namespace asr_decoder::internal {

struct Policy {
  double completion_bonus = 5.0;
  double context_prune_threshold = 2.5;
  size_t max_injected_candidates = 12;
  size_t max_completed_contexts = 8;
};

struct SearchSettings {
  size_t beam = 8;
  size_t token_beam = 16;
  double prune_threshold = 4.0;
};

Policy policy_for(HotwordStrength strength);
SearchSettings settings_for(DecodingQuality quality);
double gating_factor(const std::vector<std::pair<int, double>>& top,
                     double blank);

}  // namespace asr_decoder::internal
