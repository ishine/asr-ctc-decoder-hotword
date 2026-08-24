#include "asr_decoder/ctc_decoder.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <queue>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace asr_decoder {
namespace {

constexpr double kNegativeInfinity = -std::numeric_limits<double>::infinity();

double log_add(double first, double second) {
  if (first == kNegativeInfinity) return second;
  if (second == kNegativeInfinity) return first;
  const double maximum = std::max(first, second);
  return maximum +
         std::log(std::exp(first - maximum) + std::exp(second - maximum));
}

struct Policy {
  double completion_bonus = 5.0;
  double context_prune_threshold = 2.5;
  size_t max_injected_candidates = 12;
  size_t max_completed_contexts = 8;
};

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

struct SearchSettings {
  size_t beam = 8;
  size_t token_beam = 16;
  double prune_threshold = 4.0;
};

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

struct ContextState {
  std::unordered_map<int, size_t> transitions;
  std::vector<int> transition_order;
  size_t failure = 0;
  size_t match_length = 0;
  size_t output = 0;
};

class ContextGraph {
 public:
  ContextGraph(const std::vector<std::vector<int>>& contexts,
               const Policy& policy)
      : maximum_matches_(policy.max_completed_contexts) {
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

  bool empty() const { return states_.size() == 1; }
  int maximum_token() const { return maximum_token_; }

  size_t advance(size_t state, int token) const {
    while (state != 0 && !states_[state].transitions.count(token)) {
      state = states_[state].failure;
    }
    const auto found = states_[state].transitions.find(token);
    return found == states_[state].transitions.end() ? state : found->second;
  }

  std::vector<size_t> matches(size_t state) const {
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

  std::vector<int> candidates(size_t state, size_t maximum) const {
    std::vector<int> result;
    std::unordered_set<int> seen;
    while (true) {
      for (int token : states_[state].transition_order) {
        if (seen.insert(token).second) {
          result.push_back(token);
          if (result.size() == maximum) return result;
        }
      }
      if (state == 0) break;
      state = states_[state].failure;
    }
    return result;
  }

 private:
  void build_failures() {
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

  std::vector<ContextState> states_;
  size_t maximum_matches_ = 8;
  int maximum_token_ = 0;
};

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

  double acoustic() const { return log_add(blank, nonblank); }
  double viterbi() const { return std::max(viterbi_blank, viterbi_nonblank); }
  double context() const {
    return context_history.empty() ? 0.0 : context_history.back();
  }
  double total() const { return acoustic() + context(); }
  const std::vector<TokenSpan>& alignment() const {
    return viterbi_blank > viterbi_nonblank ? blank_alignment
                                            : nonblank_alignment;
  }
};

std::vector<std::pair<int, double>> top_k(const float* row, size_t size,
                                          size_t count) {
  std::vector<std::pair<int, double>> result;
  result.reserve(size);
  for (size_t token = 0; token < size; ++token) {
    result.emplace_back(static_cast<int>(token), row[token]);
  }
  const size_t kept = std::min(count, result.size());
  std::partial_sort(result.begin(), result.begin() + kept, result.end(),
                    [](const auto& left, const auto& right) {
                      if (left.second != right.second) {
                        return left.second > right.second;
                      }
                      return left.first > right.first;
                    });
  result.resize(kept);
  return result;
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

}  // namespace

struct CTCDecoder::Impl {
  explicit Impl(DecoderConfig value)
      : config(std::move(value)),
        policy(policy_for(config.hotword_strength)),
        graph(config.context_token_ids, policy) {
    if (config.blank_id < 0) config_error = "blank_id must be non-negative";
    if (!std::isfinite(config.frame_shift_ms) || config.frame_shift_ms < 0.0) {
      config_error = "frame_shift_ms must be finite and non-negative";
    }
    for (const auto& context : config.context_token_ids) {
      for (int token : context) {
        if (token < 0) {
          config_error = "hotword token IDs must be non-negative";
        } else if (token == config.blank_id) {
          config_error = "hotword token IDs must not contain blank_id";
        }
      }
    }
    reset();
  }

  void reset() {
    processed_frames = 0;
    vocabulary_size = 0;
    beam_size = 0;
    token_beam_size = 0;
    error.clear();
    hypotheses.clear();
    PrefixScore initial;
    initial.blank = 0.0;
    initial.viterbi_blank = 0.0;
    hypotheses.push_back({{}, std::move(initial)});
  }

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

CTCDecoder::CTCDecoder(DecoderConfig config)
    : impl_(std::make_unique<Impl>(std::move(config))) {}
CTCDecoder::~CTCDecoder() = default;
CTCDecoder::CTCDecoder(CTCDecoder&&) noexcept = default;
CTCDecoder& CTCDecoder::operator=(CTCDecoder&&) noexcept = default;

bool CTCDecoder::valid() const {
  return impl_ && impl_->config_error.empty() && impl_->error.empty();
}
const char* CTCDecoder::error() const {
  if (!impl_) return "decoder is not initialized";
  return impl_->config_error.empty() ? impl_->error.c_str()
                                     : impl_->config_error.c_str();
}
void CTCDecoder::reset() {
  if (impl_) impl_->reset();
}

DecodeResult CTCDecoder::prefix_beam_search(const float* values, size_t frames,
                                            size_t vocabulary_size,
                                            DecodeOptions options) {
  DecodeResult result;
  if (!impl_ || !impl_->config_error.empty()) return result;
  impl_->error.clear();
  if ((!values && frames) || vocabulary_size == 0 ||
      static_cast<size_t>(impl_->config.blank_id) >= vocabulary_size ||
      (!impl_->graph.empty() &&
       static_cast<size_t>(impl_->graph.maximum_token()) >= vocabulary_size)) {
    impl_->error = "invalid log-probability matrix or token ID";
    return result;
  }
  if (impl_->vocabulary_size && impl_->vocabulary_size != vocabulary_size) {
    impl_->error = "vocabulary_size changed during a stream";
    return result;
  }
  impl_->vocabulary_size = vocabulary_size;
  SearchSettings settings = settings_for(impl_->config.decoding_quality);
  if (options.beam_size) settings.beam = options.beam_size;
  if (options.token_beam_size) settings.token_beam = options.token_beam_size;
  settings.token_beam = std::min(settings.token_beam, vocabulary_size);
  if (settings.beam == 0 || settings.token_beam == 0) {
    impl_->error = "beam sizes must be positive";
    return result;
  }
  if (impl_->beam_size && impl_->beam_size != settings.beam) {
    impl_->error = "beam_size changed during a stream";
    return result;
  }
  if (impl_->token_beam_size && impl_->token_beam_size != settings.token_beam) {
    impl_->error = "token_beam_size changed during a stream";
    return result;
  }
  for (size_t index = 0; index < frames * vocabulary_size; ++index) {
    if (std::isnan(values[index]) ||
        values[index] == std::numeric_limits<float>::infinity()) {
      impl_->error =
          "log probabilities must not contain NaN or positive infinity";
      return result;
    }
  }
  for (size_t frame = 0; frame < frames; ++frame) {
    bool has_finite_value = false;
    for (size_t token = 0; token < vocabulary_size; ++token) {
      has_finite_value |=
          std::isfinite(values[frame * vocabulary_size + token]);
    }
    if (!has_finite_value) {
      impl_->error = "every frame must contain a finite log probability";
      return result;
    }
  }
  impl_->beam_size = settings.beam;
  impl_->token_beam_size = settings.token_beam;

  const ContextGraph* graph = impl_->graph.empty() ? nullptr : &impl_->graph;
  for (size_t frame = 0; frame < frames; ++frame) {
    const float* row = values + frame * vocabulary_size;
    auto acoustic = top_k(row, vocabulary_size, settings.token_beam);
    if (acoustic.empty()) continue;
    const double best = acoustic.front().second;
    const double blank = row[impl_->config.blank_id];
    const double threshold =
        impl_->policy.context_prune_threshold * gating_factor(acoustic, blank);

    std::map<int, double> candidates;
    for (const auto& [token, probability] : acoustic) {
      if (probability >= best - settings.prune_threshold ||
          token == impl_->config.blank_id) {
        candidates[token] = probability;
      }
    }
    candidates[impl_->config.blank_id] = blank;
    if (graph) {
      std::unordered_set<int> injected;
      for (const auto& hypothesis : impl_->hypotheses) {
        for (int token :
             graph->candidates(hypothesis.second.context_state,
                               impl_->policy.max_injected_candidates)) {
          if (injected.size() == impl_->policy.max_injected_candidates) break;
          injected.insert(token);
        }
      }
      for (int token : injected) {
        const double probability = row[token];
        if (probability >= best - threshold) candidates[token] = probability;
      }
    }

    std::map<std::vector<int>, PrefixScore> next;
    ++impl_->processed_frames;
    for (const auto& [token, probability] : candidates) {
      for (const auto& [prefix, source] : impl_->hypotheses) {
        if (token == impl_->config.blank_id) {
          PrefixScore& destination = next[prefix];
          destination.blank =
              log_add(destination.blank, source.acoustic() + probability);
          const double viterbi = source.viterbi() + probability;
          if (viterbi > destination.viterbi_blank) {
            destination.viterbi_blank = viterbi;
            destination.blank_alignment = source.alignment();
          }
          copy_context(source, destination, graph != nullptr);
          continue;
        }

        if (!prefix.empty() && prefix.back() == token) {
          PrefixScore& repeated = next[prefix];
          repeated.nonblank =
              log_add(repeated.nonblank, source.nonblank + probability);
          if (source.viterbi_nonblank + probability >
              repeated.viterbi_nonblank) {
            repeated.viterbi_nonblank = source.viterbi_nonblank + probability;
            repeated.nonblank_alignment = source.nonblank_alignment;
            if (!repeated.nonblank_alignment.empty()) {
              TokenSpan& span = repeated.nonblank_alignment.back();
              span.end = impl_->processed_frames;
              span.log_probability =
                  std::max(span.log_probability, probability);
            }
          }
          copy_context(source, repeated, graph != nullptr);

          std::vector<int> extended_prefix = prefix;
          extended_prefix.push_back(token);
          PrefixScore& after_blank = next[extended_prefix];
          after_blank.nonblank =
              log_add(after_blank.nonblank, source.blank + probability);
          if (source.viterbi_blank + probability >
              after_blank.viterbi_nonblank) {
            after_blank.viterbi_nonblank = source.viterbi_blank + probability;
            after_blank.nonblank_alignment = source.blank_alignment;
            after_blank.nonblank_alignment.push_back(
                {token, impl_->processed_frames - 1, impl_->processed_frames,
                 probability});
          }
          advance_context(graph, impl_->policy, source, after_blank, token);
          continue;
        }

        std::vector<int> extended_prefix = prefix;
        extended_prefix.push_back(token);
        PrefixScore& destination = next[extended_prefix];
        destination.nonblank =
            log_add(destination.nonblank, source.acoustic() + probability);
        if (source.viterbi() + probability > destination.viterbi_nonblank) {
          destination.viterbi_nonblank = source.viterbi() + probability;
          destination.nonblank_alignment = source.alignment();
          destination.nonblank_alignment.push_back(
              {token, impl_->processed_frames - 1, impl_->processed_frames,
               probability});
        }
        advance_context(graph, impl_->policy, source, destination, token);
      }
    }
    impl_->hypotheses.assign(std::make_move_iterator(next.begin()),
                             std::make_move_iterator(next.end()));
    std::sort(impl_->hypotheses.begin(), impl_->hypotheses.end(),
              [](const auto& left, const auto& right) {
                if (left.second.total() != right.second.total()) {
                  return left.second.total() > right.second.total();
                }
                return left.first < right.first;
              });
    if (impl_->hypotheses.size() > settings.beam) {
      impl_->hypotheses.resize(settings.beam);
    }
  }

  for (const auto& [prefix, score] : impl_->hypotheses) {
    const auto& alignment = score.alignment();
    std::vector<int> tokens;
    std::vector<Timestamp> timestamps;
    std::vector<double> probabilities;
    for (const TokenSpan& span : alignment) {
      tokens.push_back(span.token);
      Timestamp timestamp{span.start, span.end};
      if (impl_->config.frame_shift_ms > 0.0) {
        timestamp.start_ms = span.start * impl_->config.frame_shift_ms;
        timestamp.end_ms = span.end * impl_->config.frame_shift_ms;
      }
      timestamps.push_back(timestamp);
      if (options.return_token_probabilities) {
        probabilities.push_back(std::exp(span.log_probability));
      }
    }
    result.tokens.push_back(std::move(tokens));
    result.timestamps.push_back(std::move(timestamps));
    if (options.return_token_probabilities) {
      result.probabilities.push_back(std::move(probabilities));
    }
    result.scores.push_back({score.acoustic(), score.context(), score.total()});
  }
  if (options.finalize) impl_->reset();
  return result;
}

}  // namespace asr_decoder
