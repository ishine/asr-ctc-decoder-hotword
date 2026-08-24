#include "context_policy.h"

#include <algorithm>
#include <cmath>

namespace asr_decoder::internal {

Policy policy_for(HotwordStrength strength) {
  switch (strength) {
    case HotwordStrength::kConservative:
      return {3.5, 1.5, 8, 8};
    case HotwordStrength::kAggressive:
      return {7.0, 3.5, 16, 8};
    case HotwordStrength::kLowFalseActivation:
      return {1.0, 0.5, 8, 8};
    case HotwordStrength::kBalanced:
      return {};
  }
  return {};
}

SearchSettings settings_for(DecodingQuality quality) {
  switch (quality) {
    case DecodingQuality::kLowLatency:
      return {4, 8, 3.0};
    case DecodingQuality::kHighAccuracy:
      return {12, 24, 5.0};
    case DecodingQuality::kBalanced:
      return {};
  }
  return {};
}

double gating_factor(const std::vector<std::pair<int, double>>& top,
                     double blank) {
  const size_t count = std::min<size_t>(8, top.size());
  if (count < 2) return 1.0;
  const double best = top.front().second;
  const double spread = best - top[count - 1].second;
  const double acoustic = std::clamp(spread / 4.0, 0.5, 2.0);
  double total = 0.0;
  std::vector<double> probabilities;
  probabilities.reserve(count);
  for (size_t index = 0; index < count; ++index) {
    probabilities.push_back(std::exp(top[index].second - best));
    total += probabilities.back();
  }
  double entropy = 0.0;
  for (double probability : probabilities) {
    probability /= total;
    if (probability > 0.0) entropy -= probability * std::log(probability);
  }
  double confidence =
      1.0 - 0.35 * entropy / std::log(static_cast<double>(count));
  if (blank == best) {
    const double margin = (best - top[1].second) / std::max(acoustic, 1e-6);
    if (margin > 1.0) confidence *= std::max(0.25, 1.0 / margin);
  }
  return acoustic * confidence;
}

}  // namespace asr_decoder::internal
