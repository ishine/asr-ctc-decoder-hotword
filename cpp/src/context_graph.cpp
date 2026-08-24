#include "context_graph.h"

#include <algorithm>
#include <queue>
#include <set>
#include <unordered_set>

namespace asr_decoder::internal {

ContextGraph::ContextGraph(const std::vector<std::vector<int>>& contexts,
                           const Policy& policy)
    : maximum_matches_(policy.max_completed_contexts_per_token) {
  states_.emplace_back();
  std::set<std::vector<int>> seen;
  for (const std::vector<int>& tokens : contexts) {
    if (tokens.empty() || !seen.insert(tokens).second) continue;
    size_t state = 0;
    for (int token : tokens) {
      maximum_token_ = std::max(maximum_token_, token);
      auto found = states_[state].transitions.find(token);
      if (found == states_[state].transitions.end()) {
        const size_t next = states_.size();
        states_.emplace_back();
        states_[state].transitions.emplace(token, next);
        states_[state].transition_order.push_back(token);
        state = next;
      } else {
        state = found->second;
      }
    }
    states_[state].match_length = tokens.size();
  }
  build_failures();
}

size_t ContextGraph::advance(size_t state, int token) const {
  while (state != 0 && !states_[state].transitions.count(token)) {
    state = states_[state].failure;
  }
  const auto found = states_[state].transitions.find(token);
  return found == states_[state].transitions.end() ? state : found->second;
}

std::vector<size_t> ContextGraph::matches(size_t state) const {
  std::vector<size_t> result;
  if (states_[state].match_length) {
    result.push_back(states_[state].match_length);
  }
  while (result.size() < maximum_matches_ && states_[state].output) {
    state = states_[state].output;
    if (states_[state].match_length) {
      result.push_back(states_[state].match_length);
    }
  }
  return result;
}

ActiveContextCandidates ContextGraph::active_candidates(size_t state,
                                                        size_t maximum) const {
  ActiveContextCandidates result;
  if (maximum == 0) {
    result.truncated = !states_[state].transitions.empty();
    return result;
  }
  std::unordered_set<int> seen;
  while (true) {
    const State& current = states_[state];
    result.competing_count = std::min(
        maximum + 1, result.competing_count + current.transitions.size());
    if (state == 0 &&
        current.transitions.size() > maximum - result.tokens.size()) {
      result.truncated = true;
      return result;
    }
    for (int token : current.transition_order) {
      if (seen.insert(token).second) {
        result.tokens.push_back(token);
        if (result.tokens.size() == maximum) {
          result.truncated =
              state != 0 || std::any_of(current.transition_order.begin(),
                                        current.transition_order.end(),
                                        [&seen](int candidate) {
                                          return !seen.count(candidate);
                                        });
          return result;
        }
      }
    }
    if (state == 0) break;
    state = current.failure;
  }
  return result;
}

void ContextGraph::build_failures() {
  std::queue<size_t> queue;
  for (const auto& transition : states_[0].transitions) {
    queue.push(transition.second);
  }
  while (!queue.empty()) {
    const size_t current = queue.front();
    queue.pop();
    const auto transitions = states_[current].transitions;
    for (const auto& [token, next] : transitions) {
      size_t failure = states_[current].failure;
      while (failure != 0 && !states_[failure].transitions.count(token)) {
        failure = states_[failure].failure;
      }
      const auto found = states_[failure].transitions.find(token);
      if (found != states_[failure].transitions.end()) {
        failure = found->second;
      }
      states_[next].failure = failure;
      states_[next].output =
          states_[failure].match_length ? failure : states_[failure].output;
      queue.push(next);
    }
  }
}

}  // namespace asr_decoder::internal
