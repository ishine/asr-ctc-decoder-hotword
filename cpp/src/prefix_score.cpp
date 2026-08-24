#include "prefix_score.h"

#include <algorithm>
#include <cmath>
#include <utility>

namespace asr_decoder::internal {

double log_add(double first, double second) {
  if (first == kNegativeInfinity) return second;
  if (second == kNegativeInfinity) return first;
  const double maximum = std::max(first, second);
  return maximum +
         std::log(std::exp(first - maximum) + std::exp(second - maximum));
}

double PrefixScore::acoustic() const { return log_add(blank, nonblank); }
double PrefixScore::viterbi() const {
  return std::max(viterbi_blank, viterbi_nonblank);
}
double PrefixScore::context() const {
  return context_history.empty() ? 0.0 : context_history.back();
}
double PrefixScore::total() const { return acoustic() + context(); }
const std::vector<TokenSpan>& PrefixScore::alignment() const {
  return viterbi_blank > viterbi_nonblank ? blank_alignment
                                          : nonblank_alignment;
}

void copy_context(const PrefixScore& source, PrefixScore& destination,
                  bool enabled) {
  if (enabled && !destination.has_context) {
    destination.context_state = source.context_state;
    destination.context_history = source.context_history;
    destination.has_context = true;
  }
}

void advance_context(const ContextGraph* graph, const Policy& policy,
                     const PrefixScore& source, PrefixScore& destination,
                     int token) {
  if (!graph) return;
  const size_t state = graph->advance(source.context_state, token);
  std::vector<double> history = source.context_history;
  double score = history.back();
  for (size_t length : graph->matches(state)) {
    if (length <= history.size()) {
      score = std::max(
          score, history[history.size() - length] + policy.completion_bonus);
    }
  }
  history.push_back(score);
  if (!destination.has_context || score > destination.context()) {
    destination.context_state = state;
    destination.context_history = std::move(history);
    destination.has_context = true;
  }
}

}  // namespace asr_decoder::internal
