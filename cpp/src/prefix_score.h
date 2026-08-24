#pragma once

#include <cstddef>
#include <limits>
#include <vector>

#include "context_graph.h"

namespace asr_decoder::internal {

inline constexpr double kNegativeInfinity =
    -std::numeric_limits<double>::infinity();

double log_add(double first, double second);

struct TokenSpan {
  int token = 0;
  size_t start = 0;
  size_t end = 0;
  double log_probability = kNegativeInfinity;
};

struct PrefixScore {
  double blank = kNegativeInfinity;
  double nonblank = kNegativeInfinity;
  double viterbi_blank = kNegativeInfinity;
  double viterbi_nonblank = kNegativeInfinity;
  std::vector<TokenSpan> blank_alignment;
  std::vector<TokenSpan> nonblank_alignment;
  size_t context_state = 0;
  std::vector<double> context_history{0.0};
  bool has_context = false;

  double acoustic() const;
  double viterbi() const;
  double context() const;
  double total() const;
  const std::vector<TokenSpan>& alignment() const;
};

void copy_context(const PrefixScore& source, PrefixScore& destination,
                  bool enabled);
void advance_context(const ContextGraph* graph, const Policy& policy,
                     const PrefixScore& source, PrefixScore& destination,
                     int token);

}  // namespace asr_decoder::internal
