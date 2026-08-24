#pragma once

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "asr_decoder/ctc_decoder.h"

namespace asr_decoder::internal {

using Policy = ContextPolicy;

struct SearchSettings {
  size_t beam = 8;
  size_t token_beam = 16;
  double prune_threshold = 4.0;
};

Policy policy_for(HotwordStrength strength);
SearchSettings settings_for(DecodingQuality quality);
bool validate_policy(const Policy& policy, std::string& error);
std::pair<double, double> analyze_frame(
    const AdaptiveGatingConfig& config,
    const std::vector<std::pair<int, double>>& top, double blank);
double active_context_factor(const AdaptiveGatingConfig& config, size_t count,
                             size_t maximum);

}  // namespace asr_decoder::internal
