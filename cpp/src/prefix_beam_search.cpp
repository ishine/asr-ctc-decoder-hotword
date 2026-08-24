#include "prefix_beam_search.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace asr_decoder::internal {
namespace {

using Candidate = std::pair<int, double>;

struct ContextCandidates {
  std::vector<int> tokens;
  std::unordered_set<int> root_tokens;
  std::unordered_set<int> non_root_tokens;
  size_t competing_count = 0;
};

std::vector<Candidate> top_k(const float* row, size_t size, size_t count) {
  std::vector<Candidate> result;
  result.reserve(size);
  for (size_t token = 0; token < size; ++token) {
    result.emplace_back(static_cast<int>(token), row[token]);
  }
  const size_t kept = std::min(count, result.size());
  std::partial_sort(result.begin(), result.begin() + kept, result.end(),
                    [](const Candidate& left, const Candidate& right) {
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
  if (!validate_input(state, values, frames, vocabulary_size)) return false;
  if (state.vocabulary_size && state.vocabulary_size != vocabulary_size) {
    state.error = "vocabulary_size changed during a stream";
    return false;
  }

  settings = settings_for(state.config.decoding_quality);
  if (state.vocabulary_size) {
    settings.beam = state.beam_size;
    settings.token_beam = state.token_beam_size;
    if (options.beam_size && options.beam_size != state.beam_size) {
      state.error = "beam_size changed during a stream";
      return false;
    }
    const size_t requested_token_beam =
        std::min(options.token_beam_size, vocabulary_size);
    if (options.token_beam_size &&
        requested_token_beam != state.token_beam_size) {
      state.error = "token_beam_size changed during a stream";
      return false;
    }
  } else {
    if (options.beam_size) settings.beam = options.beam_size;
    if (options.token_beam_size) settings.token_beam = options.token_beam_size;
    settings.token_beam = std::min(settings.token_beam, vocabulary_size);
  }
  if (settings.beam == 0 || settings.token_beam == 0) {
    state.error = "beam sizes must be positive";
    return false;
  }

  return true;
}

ContextCandidates collect_context_candidates(DecoderState& state,
                                             const ContextGraph* graph) {
  ContextCandidates result;
  const size_t maximum = state.policy.max_injected_candidates;
  if (!graph || maximum == 0) return result;

  std::unordered_set<int> token_set;
  std::unordered_set<size_t> seen_states;
  bool truncated = false;
  for (const auto& hypothesis : state.hypotheses) {
    const size_t context_state = hypothesis.second.context_state;
    if (!seen_states.insert(context_state).second) continue;
    ActiveContextCandidates active =
        graph->active_candidates(context_state, maximum);
    result.competing_count =
        std::min(maximum + 1, result.competing_count + active.competing_count);
    truncated |= active.truncated;
    auto& target =
        context_state == 0 ? result.root_tokens : result.non_root_tokens;
    target.insert(active.tokens.begin(), active.tokens.end());
    for (int token : active.tokens) {
      if (token_set.count(token)) continue;
      if (result.tokens.size() == maximum) {
        truncated = true;
        break;
      }
      result.tokens.push_back(token);
      token_set.insert(token);
    }
  }
  if (truncated) ++state.diagnostics.context_candidate_limit_hits;
  state.diagnostics.max_context_candidates_per_frame = std::max(
      state.diagnostics.max_context_candidates_per_frame, result.tokens.size());
  state.diagnostics.evaluated_context_candidates += result.tokens.size();
  return result;
}

std::unordered_set<int> apply_word_boundary_guard(
    DecoderState& state, const std::vector<Candidate>& acoustic,
    ContextCandidates& context, double best, double threshold) {
  std::unordered_set<int> blocked;
  if (state.word_boundary_token_ids.empty()) return blocked;

  double boundary_best = -std::numeric_limits<double>::infinity();
  std::unordered_set<int> acoustic_tokens;
  for (const auto& [token, probability] : acoustic) {
    acoustic_tokens.insert(token);
    if (state.word_boundary_token_ids.count(token)) {
      boundary_best = std::max(boundary_best, probability);
    }
  }
  if (boundary_best < best - threshold) return blocked;

  for (int token : context.non_root_tokens) {
    if (!context.root_tokens.count(token) &&
        !state.word_boundary_token_ids.count(token) &&
        !acoustic_tokens.count(token)) {
      blocked.insert(token);
    }
  }
  if (!blocked.empty()) {
    context.tokens.erase(
        std::remove_if(context.tokens.begin(), context.tokens.end(),
                       [&blocked](int token) { return blocked.count(token); }),
        context.tokens.end());
    state.diagnostics.boundary_suppressed_context_candidates += blocked.size();
  }
  return blocked;
}

std::vector<Candidate> frame_candidates(DecoderState& state, const float* row,
                                        const std::vector<Candidate>& acoustic,
                                        ContextCandidates& context,
                                        const SearchSettings& settings,
                                        double context_threshold) {
  const double best = acoustic.front().second;
  const auto blocked = apply_word_boundary_guard(state, acoustic, context, best,
                                                 context_threshold);
  std::map<int, double> scores;
  for (const auto& [token, probability] : acoustic) {
    scores[token] = probability;
  }
  scores[state.config.blank_id] = row[state.config.blank_id];
  for (int token : blocked) scores.erase(token);

  for (int token : context.tokens) {
    if (scores.count(token)) continue;
    ++state.diagnostics.gathered_context_candidates;
    const double probability = row[token];
    if (probability >= best - context_threshold) scores[token] = probability;
  }

  std::vector<Candidate> candidates;
  for (const auto& [token, probability] : scores) {
    if (probability >= best - settings.prune_threshold ||
        token == state.config.blank_id) {
      candidates.emplace_back(token, probability);
    }
  }
  std::sort(candidates.begin(), candidates.end(),
            [](const Candidate& left, const Candidate& right) {
              if (left.second != right.second) {
                return left.second > right.second;
              }
              return left.first > right.first;
            });
  return candidates;
}

void advance_frame(DecoderState& state, const float* row,
                   size_t vocabulary_size, const SearchSettings& settings) {
  const ContextGraph* graph = state.context_enabled() ? &state.graph : nullptr;
  const auto acoustic = top_k(row, vocabulary_size, settings.token_beam);
  if (acoustic.empty()) return;

  double acoustic_scale = 1.0;
  double confidence_factor = 1.0;
  const bool gating_enabled = graph && state.policy.gating.enabled;
  if (gating_enabled) {
    const size_t gating_width = std::min<size_t>(8, vocabulary_size);
    std::vector<Candidate> gating;
    if (settings.token_beam >= gating_width) {
      gating.assign(acoustic.begin(), acoustic.begin() + gating_width);
    } else {
      gating = top_k(row, vocabulary_size, gating_width);
    }
    const auto stats =
        analyze_frame(state.policy.gating, gating, row[state.config.blank_id]);
    acoustic_scale = stats.first;
    confidence_factor = stats.second;
    state.diagnostics.gating_acoustic_scale_sum += acoustic_scale;
    ++state.diagnostics.gating_frames;
  }

  ContextCandidates context = collect_context_candidates(state, graph);
  const double active_factor =
      active_context_factor(state.policy.gating, context.competing_count,
                            state.policy.max_injected_candidates);
  const double gating_factor =
      acoustic_scale * confidence_factor * active_factor;
  if (gating_enabled && row[state.config.blank_id] < acoustic.front().second) {
    state.diagnostics.gating_factor_sum += gating_factor;
    ++state.diagnostics.nonblank_gating_frames;
  }
  const double context_threshold =
      state.policy.context_token_prune_threshold * gating_factor;
  const auto candidates = frame_candidates(state, row, acoustic, context,
                                           settings, context_threshold);

  std::map<std::vector<int>, PrefixScore> next;
  std::vector<std::vector<int>> insertion_order;
  auto destination_for = [&next, &insertion_order](
                             const std::vector<int>& prefix) -> PrefixScore& {
    auto [found, inserted] = next.try_emplace(prefix);
    if (inserted) insertion_order.push_back(prefix);
    return found->second;
  };

  ++state.processed_frames;
  for (const auto& [token, probability] : candidates) {
    for (const auto& [prefix, source] : state.hypotheses) {
      if (token == state.config.blank_id) {
        PrefixScore& destination = destination_for(prefix);
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
        PrefixScore& repeated = destination_for(prefix);
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
        PrefixScore& after_blank = destination_for(extended_prefix);
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
      PrefixScore& destination = destination_for(extended_prefix);
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

  state.hypotheses.clear();
  for (const auto& prefix : insertion_order) {
    auto found = next.find(prefix);
    if (std::isfinite(found->second.acoustic())) {
      state.hypotheses.push_back({prefix, std::move(found->second)});
    }
  }
  std::stable_sort(state.hypotheses.begin(), state.hypotheses.end(),
                   [](const auto& left, const auto& right) {
                     return left.second.total() > right.second.total();
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

  if (options.return_gating_diagnostics) {
    const bool enabled = state.context_enabled() && state.policy.gating.enabled;
    GatingDiagnostics diagnostics;
    diagnostics.contextual_biasing = state.context_enabled();
    diagnostics.adaptive_context_gating = enabled;
    diagnostics.gating_frames = enabled ? state.diagnostics.gating_frames : 0;
    if (enabled && state.diagnostics.gating_frames) {
      diagnostics.mean_gating_acoustic_scale =
          state.diagnostics.gating_acoustic_scale_sum /
          state.diagnostics.gating_frames;
    }
    if (enabled && state.diagnostics.nonblank_gating_frames) {
      diagnostics.mean_gating_factor = state.diagnostics.gating_factor_sum /
                                       state.diagnostics.nonblank_gating_frames;
    }
    diagnostics.nonblank_gating_frames =
        enabled ? state.diagnostics.nonblank_gating_frames : 0;
    diagnostics.evaluated_context_candidates =
        state.diagnostics.evaluated_context_candidates;
    diagnostics.gathered_context_candidates =
        state.diagnostics.gathered_context_candidates;
    diagnostics.max_context_candidates_per_frame =
        state.diagnostics.max_context_candidates_per_frame;
    diagnostics.context_candidate_limit_hits =
        state.diagnostics.context_candidate_limit_hits;
    diagnostics.word_boundary_aware_context =
        !state.word_boundary_token_ids.empty();
    diagnostics.boundary_suppressed_context_candidates =
        state.diagnostics.boundary_suppressed_context_candidates;
    diagnostics.beam_size = state.beam_size;
    diagnostics.token_beam_size = state.token_beam_size;
    result.gating = diagnostics;
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
  if (!lock_decode_mode(state, DecodeMode::kPrefixBeam)) return result;
  if (state.vocabulary_size == 0) {
    state.vocabulary_size = vocabulary_size;
    state.beam_size = settings.beam;
    state.token_beam_size = settings.token_beam;
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
