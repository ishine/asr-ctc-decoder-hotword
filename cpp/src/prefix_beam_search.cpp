#include "prefix_beam_search.h"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <limits>
#include <map>
#include <unordered_set>

namespace asr_decoder::internal {
namespace {

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

bool prepare_search(DecoderState& state, const float* values, size_t frames,
                    size_t vocabulary_size, DecodeOptions options,
                    SearchSettings& settings) {
  state.error.clear();
  if ((!values && frames) || vocabulary_size == 0 ||
      static_cast<size_t>(state.config.blank_id) >= vocabulary_size ||
      (!state.graph.empty() &&
       static_cast<size_t>(state.graph.maximum_token()) >= vocabulary_size)) {
    state.error = "invalid log-probability matrix or token ID";
    return false;
  }
  if (state.vocabulary_size && state.vocabulary_size != vocabulary_size) {
    state.error = "vocabulary_size changed during a stream";
    return false;
  }
  settings = settings_for(state.config.decoding_quality);
  if (options.beam_size) settings.beam = options.beam_size;
  if (options.token_beam_size) settings.token_beam = options.token_beam_size;
  settings.token_beam = std::min(settings.token_beam, vocabulary_size);
  if (settings.beam == 0 || settings.token_beam == 0) {
    state.error = "beam sizes must be positive";
    return false;
  }
  if (state.beam_size && state.beam_size != settings.beam) {
    state.error = "beam_size changed during a stream";
    return false;
  }
  if (state.token_beam_size && state.token_beam_size != settings.token_beam) {
    state.error = "token_beam_size changed during a stream";
    return false;
  }
  for (size_t index = 0; index < frames * vocabulary_size; ++index) {
    if (std::isnan(values[index]) ||
        values[index] == std::numeric_limits<float>::infinity()) {
      state.error =
          "log probabilities must not contain NaN or positive infinity";
      return false;
    }
  }
  for (size_t frame = 0; frame < frames; ++frame) {
    bool has_finite_value = false;
    for (size_t token = 0; token < vocabulary_size; ++token) {
      has_finite_value |=
          std::isfinite(values[frame * vocabulary_size + token]);
    }
    if (!has_finite_value) {
      state.error = "every frame must contain a finite log probability";
      return false;
    }
  }
  state.vocabulary_size = vocabulary_size;
  state.beam_size = settings.beam;
  state.token_beam_size = settings.token_beam;
  return true;
}

void advance_frame(DecoderState& state, const float* row,
                   size_t vocabulary_size, const SearchSettings& settings) {
  const ContextGraph* graph = state.graph.empty() ? nullptr : &state.graph;
  auto acoustic = top_k(row, vocabulary_size, settings.token_beam);
  if (acoustic.empty()) return;
  const double best = acoustic.front().second;
  const double blank = row[state.config.blank_id];
  const double threshold =
      state.policy.context_prune_threshold * gating_factor(acoustic, blank);

  std::map<int, double> candidates;
  for (const auto& [token, probability] : acoustic) {
    if (probability >= best - settings.prune_threshold ||
        token == state.config.blank_id) {
      candidates[token] = probability;
    }
  }
  candidates[state.config.blank_id] = blank;
  if (graph) {
    std::unordered_set<int> injected;
    for (const auto& hypothesis : state.hypotheses) {
      for (int token :
           graph->candidates(hypothesis.second.context_state,
                             state.policy.max_injected_candidates)) {
        if (injected.size() == state.policy.max_injected_candidates) break;
        injected.insert(token);
      }
    }
    for (int token : injected) {
      const double probability = row[token];
      if (probability >= best - threshold) candidates[token] = probability;
    }
  }

  std::map<std::vector<int>, PrefixScore> next;
  ++state.processed_frames;
  for (const auto& [token, probability] : candidates) {
    for (const auto& [prefix, source] : state.hypotheses) {
      if (token == state.config.blank_id) {
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
        if (source.viterbi_nonblank + probability > repeated.viterbi_nonblank) {
          repeated.viterbi_nonblank = source.viterbi_nonblank + probability;
          repeated.nonblank_alignment = source.nonblank_alignment;
          if (!repeated.nonblank_alignment.empty()) {
            TokenSpan& span = repeated.nonblank_alignment.back();
            span.end = state.processed_frames;
            span.log_probability = std::max(span.log_probability, probability);
          }
        }
        copy_context(source, repeated, graph != nullptr);

        std::vector<int> extended_prefix = prefix;
        extended_prefix.push_back(token);
        PrefixScore& after_blank = next[extended_prefix];
        after_blank.nonblank =
            log_add(after_blank.nonblank, source.blank + probability);
        if (source.viterbi_blank + probability > after_blank.viterbi_nonblank) {
          after_blank.viterbi_nonblank = source.viterbi_blank + probability;
          after_blank.nonblank_alignment = source.blank_alignment;
          after_blank.nonblank_alignment.push_back(
              {token, state.processed_frames - 1, state.processed_frames,
               probability});
        }
        advance_context(graph, state.policy, source, after_blank, token);
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
            {token, state.processed_frames - 1, state.processed_frames,
             probability});
      }
      advance_context(graph, state.policy, source, destination, token);
    }
  }
  state.hypotheses.assign(std::make_move_iterator(next.begin()),
                          std::make_move_iterator(next.end()));
  std::sort(state.hypotheses.begin(), state.hypotheses.end(),
            [](const auto& left, const auto& right) {
              if (left.second.total() != right.second.total()) {
                return left.second.total() > right.second.total();
              }
              return left.first < right.first;
            });
  if (state.hypotheses.size() > settings.beam) {
    state.hypotheses.resize(settings.beam);
  }
}

DecodeResult build_result(const DecoderState& state, DecodeOptions options) {
  DecodeResult result;
  for (const auto& hypothesis : state.hypotheses) {
    const PrefixScore& score = hypothesis.second;
    const auto& alignment = score.alignment();
    std::vector<int> tokens;
    std::vector<Timestamp> timestamps;
    std::vector<double> probabilities;
    for (const TokenSpan& span : alignment) {
      tokens.push_back(span.token);
      Timestamp timestamp{span.start, span.end};
      if (state.config.frame_shift_ms > 0.0) {
        timestamp.start_ms = span.start * state.config.frame_shift_ms;
        timestamp.end_ms = span.end * state.config.frame_shift_ms;
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
  return result;
}

}  // namespace

DecodeResult run_prefix_beam_search(DecoderState& state, const float* values,
                                    size_t frames, size_t vocabulary_size,
                                    DecodeOptions options) {
  DecodeResult result;
  if (!state.config_error.empty()) return result;
  SearchSettings settings;
  if (!prepare_search(state, values, frames, vocabulary_size, options,
                      settings)) {
    return result;
  }
  for (size_t frame = 0; frame < frames; ++frame) {
    advance_frame(state, values + frame * vocabulary_size, vocabulary_size,
                  settings);
  }
  result = build_result(state, options);
  if (options.finalize) state.reset();
  return result;
}

}  // namespace asr_decoder::internal
