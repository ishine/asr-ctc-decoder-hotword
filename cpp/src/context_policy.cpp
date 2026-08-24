#include "context_policy.h"

#include <algorithm>
#include <cmath>

namespace asr_decoder {

ContextPolicy ContextPolicy::conservative() { return {}; }

ContextPolicy ContextPolicy::balanced() {
  ContextPolicy policy;
  policy.completion_bonus = 5.0;
  policy.context_token_prune_threshold = 2.5;
  policy.max_injected_candidates = 12;
  return policy;
}

ContextPolicy ContextPolicy::aggressive() {
  ContextPolicy policy;
  policy.completion_bonus = 7.0;
  policy.context_token_prune_threshold = 3.5;
  policy.max_injected_candidates = 16;
  return policy;
}

ContextPolicy ContextPolicy::low_false_activation() {
  ContextPolicy policy;
  policy.completion_bonus = 1.0;
  policy.context_token_prune_threshold = 0.5;
  return policy;
}

namespace internal {

namespace {

bool finite_positive(double value) {
  return std::isfinite(value) && value > 0.0;
}

bool finite_non_negative(double value) {
  return std::isfinite(value) && value >= 0.0;
}

}  // namespace

Policy policy_for(HotwordStrength strength) {
  switch (strength) {
    case HotwordStrength::kConservative:
      return ContextPolicy::conservative();
    case HotwordStrength::kAggressive:
      return ContextPolicy::aggressive();
    case HotwordStrength::kLowFalseActivation:
      return ContextPolicy::low_false_activation();
    case HotwordStrength::kBalanced:
      return ContextPolicy::balanced();
  }
  return ContextPolicy::balanced();
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

bool validate_policy(const Policy& policy, std::string& error) {
  const auto& gating = policy.gating;
  if (!finite_non_negative(policy.completion_bonus) ||
      !finite_non_negative(policy.context_token_prune_threshold)) {
    error = "context scores must be finite and non-negative";
    return false;
  }
  if (policy.max_completed_contexts_per_token == 0) {
    error = "max_completed_contexts_per_token must be positive";
    return false;
  }
  if (!finite_positive(gating.reference_top_spread) ||
      !finite_positive(gating.min_acoustic_scale) ||
      !finite_positive(gating.max_acoustic_scale)) {
    error = "acoustic gating scales must be finite and positive";
    return false;
  }
  if (gating.min_acoustic_scale > gating.max_acoustic_scale) {
    error = "acoustic scale limits must be ordered";
    return false;
  }
  if (!std::isfinite(gating.min_entropy_confidence_factor) ||
      gating.min_entropy_confidence_factor <= 0.0 ||
      gating.min_entropy_confidence_factor > 1.0 ||
      !std::isfinite(gating.max_active_context_penalty) ||
      gating.max_active_context_penalty < 0.0 ||
      gating.max_active_context_penalty >= 1.0) {
    error = "gating factor is outside its valid range";
    return false;
  }
  return true;
}

std::pair<double, double> analyze_frame(
    const AdaptiveGatingConfig& config,
    const std::vector<std::pair<int, double>>& top, double blank) {
  if (!config.enabled) return {1.0, 1.0};
  const size_t count = std::min<size_t>(8, top.size());
  if (count < 2) return {1.0, 1.0};
  const double best = top.front().second;
  std::vector<double> finite;
  finite.reserve(count);
  for (size_t index = 0; index < count; ++index) {
    if (std::isfinite(top[index].second)) finite.push_back(top[index].second);
  }
  if (finite.size() < 2) return {1.0, 1.0};
  const double spread = best - finite.back();
  const double acoustic =
      std::clamp(spread / config.reference_top_spread,
                 config.min_acoustic_scale, config.max_acoustic_scale);
  double total = 0.0;
  std::vector<double> probabilities;
  probabilities.reserve(finite.size());
  for (double value : finite) {
    probabilities.push_back(std::exp(value - best));
    total += probabilities.back();
  }
  double entropy = 0.0;
  for (double probability : probabilities) {
    probability /= total;
    if (probability > 0.0) entropy -= probability * std::log(probability);
  }
  double confidence = 1.0 - (1.0 - config.min_entropy_confidence_factor) *
                                entropy /
                                std::log(static_cast<double>(finite.size()));
  if (blank == best) {
    const double margin = (best - finite[1]) / std::max(acoustic, 1e-6);
    if (margin > 1.0) confidence *= std::max(0.25, 1.0 / margin);
  }
  return {acoustic, confidence};
}

double active_context_factor(const AdaptiveGatingConfig& config, size_t count,
                             size_t maximum) {
  if (!config.enabled || count <= 1 || maximum <= 1) return 1.0;
  const double pressure =
      std::min(1.0, std::log1p(static_cast<double>(count - 1)) /
                        std::log(static_cast<double>(maximum)));
  return 1.0 - config.max_active_context_penalty * pressure;
}

}  // namespace internal
}  // namespace asr_decoder
