#pragma once

#include <cstddef>
#include <unordered_map>
#include <vector>

#include "context_policy.h"

namespace asr_decoder::internal {

class ContextGraph {
 public:
  ContextGraph(const std::vector<std::vector<int>>& contexts,
               const Policy& policy);

  bool empty() const { return states_.size() == 1; }
  int maximum_token() const { return maximum_token_; }
  size_t advance(size_t state, int token) const;
  std::vector<size_t> matches(size_t state) const;
  std::vector<int> candidates(size_t state, size_t maximum) const;

 private:
  struct State {
    std::unordered_map<int, size_t> transitions;
    std::vector<int> transition_order;
    size_t failure = 0;
    size_t match_length = 0;
    size_t output = 0;
  };

  void build_failures();

  std::vector<State> states_;
  size_t maximum_matches_ = 8;
  int maximum_token_ = 0;
};

}  // namespace asr_decoder::internal
