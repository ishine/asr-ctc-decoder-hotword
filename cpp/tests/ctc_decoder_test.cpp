#include "asr_decoder/ctc_decoder.h"

#include <cmath>
#include <cstdio>
#include <limits>
#include <numeric>
#include <vector>

#define CHECK(expression)                                                   \
  do {                                                                      \
    if (!(expression)) {                                                    \
      std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, \
                   #expression);                                            \
      return 1;                                                             \
    }                                                                       \
  } while (false)

int main() {
  const std::vector<float> values{-5.0f, -0.1f, -2.0f, -5.0f, -2.0f, -0.2f};
  asr_decoder::DecoderConfig config;
  config.context_token_ids = {{1, 2}};
  asr_decoder::CTCDecoder decoder(config);
  CHECK(decoder.valid());
  asr_decoder::DecodeOptions options;
  options.finalize = true;
  options.return_token_probabilities = true;
  const auto result = decoder.prefix_beam_search(values.data(), 2, 3, options);
  CHECK(!result.tokens.empty());
  CHECK((result.tokens[0] == std::vector<int>{1, 2}));
  CHECK(result.scores[0].contextual_bias == 5.0);

  asr_decoder::CTCDecoder stream(config);
  stream.prefix_beam_search(values.data(), 1, 3);
  const auto streamed =
      stream.prefix_beam_search(values.data() + 3, 1, 3, options);
  CHECK(streamed.tokens == result.tokens);
  CHECK(streamed.timestamps == result.timestamps);

  const std::vector<float> ambiguous{-5.0f, -0.1f, -0.2f};
  asr_decoder::DecodeOptions final_options;
  final_options.finalize = true;
  asr_decoder::CTCDecoder baseline({});
  const auto baseline_result =
      baseline.prefix_beam_search(ambiguous.data(), 1, 3, final_options);
  CHECK((baseline_result.tokens[0] == std::vector<int>{1}));

  asr_decoder::DecoderConfig biased_config;
  biased_config.context_token_ids = {{2}};
  biased_config.frame_shift_ms = 40.0;
  asr_decoder::CTCDecoder biased_decoder(biased_config);
  const auto biased_result =
      biased_decoder.prefix_beam_search(ambiguous.data(), 1, 3, final_options);
  CHECK((biased_result.tokens[0] == std::vector<int>{2}));
  CHECK(biased_result.timestamps[0][0].start_ms == 0.0);
  CHECK(biased_result.timestamps[0][0].end_ms == 40.0);

  asr_decoder::DecoderConfig blank_config;
  blank_config.context_token_ids = {{1}};
  blank_config.hotword_strength = asr_decoder::HotwordStrength::kAggressive;
  asr_decoder::CTCDecoder blank_decoder(blank_config);
  const std::vector<float> strong_blank{0.0f, -8.0f};
  const auto blank_result = blank_decoder.prefix_beam_search(
      strong_blank.data(), 1, 2, final_options);
  CHECK(blank_result.tokens[0].empty());

  asr_decoder::DecoderConfig invalid_config;
  invalid_config.context_token_ids = {{-1}};
  asr_decoder::CTCDecoder invalid_decoder(invalid_config);
  CHECK(!invalid_decoder.valid());

  asr_decoder::CTCDecoder checked_decoder({});
  const std::vector<float> nan_values{0.0f,
                                      std::numeric_limits<float>::quiet_NaN()};
  CHECK(checked_decoder.prefix_beam_search(nan_values.data(), 1, 2)
            .tokens.empty());
  CHECK(!checked_decoder.valid());
  checked_decoder.reset();
  CHECK(checked_decoder.valid());

  asr_decoder::DecodeOptions first_options;
  first_options.beam_size = 2;
  checked_decoder.prefix_beam_search(values.data(), 1, 3, first_options);
  asr_decoder::DecodeOptions changed_options;
  changed_options.beam_size = 3;
  CHECK(checked_decoder
            .prefix_beam_search(values.data() + 3, 1, 3, changed_options)
            .tokens.empty());
  CHECK(!checked_decoder.valid());
  checked_decoder.reset();
  CHECK(checked_decoder.valid());

  // Match Python: hypotheses whose acoustic score is -infinity are not part
  // of the N-best result, even when the requested beam has spare capacity.
  const float negative_infinity = -std::numeric_limits<float>::infinity();
  const std::vector<float> impossible{negative_infinity, 0.0f,
                                      negative_infinity};
  asr_decoder::DecodeOptions impossible_options;
  impossible_options.beam_size = 3;
  impossible_options.token_beam_size = 3;
  impossible_options.finalize = true;
  asr_decoder::CTCDecoder impossible_decoder({});
  const auto possible_only = impossible_decoder.prefix_beam_search(
      impossible.data(), 1, 3, impossible_options);
  CHECK(possible_only.tokens.size() == 1);
  CHECK((possible_only.tokens[0] == std::vector<int>{1}));

  // Omitting beam options on a later chunk reuses the stream's locked search
  // configuration instead of falling back to the quality preset.
  asr_decoder::CTCDecoder reused_options_decoder({});
  asr_decoder::DecodeOptions custom_options;
  custom_options.beam_size = 2;
  custom_options.token_beam_size = 2;
  reused_options_decoder.prefix_beam_search(values.data(), 1, 3,
                                            custom_options);
  const auto reused_options = reused_options_decoder.prefix_beam_search(
      values.data() + 3, 1, 3, final_options);
  CHECK(reused_options_decoder.valid());
  CHECK(!reused_options.tokens.empty());

  // Match Python's word-boundary guard: an acoustically supported word start
  // suppresses a context-only in-word continuation.
  const std::vector<float> boundary_values{-8.0f, -0.1f, -8.0f, -8.0f, -8.0f,
                                           -8.0f, -8.0f, -2.1f, -0.1f, -1.1f};
  asr_decoder::DecoderConfig boundary_config;
  boundary_config.context_token_ids = {{1, 2}};
  boundary_config.word_boundary_token_ids = {3};
  asr_decoder::ContextPolicy boundary_policy;
  boundary_policy.completion_bonus = 1.0;
  boundary_policy.context_token_prune_threshold = 3.0;
  boundary_policy.gating.enabled = false;
  boundary_config.context_policy = boundary_policy;
  asr_decoder::CTCDecoder boundary_decoder(boundary_config);
  asr_decoder::DecodeOptions boundary_options;
  boundary_options.token_beam_size = 2;
  boundary_options.finalize = true;
  boundary_options.return_gating_diagnostics = true;
  const auto boundary_result = boundary_decoder.prefix_beam_search(
      boundary_values.data(), 2, 5, boundary_options);
  CHECK((boundary_result.tokens[0] == std::vector<int>{1, 3}));
  CHECK(boundary_result.gating.has_value());
  CHECK(boundary_result.gating->boundary_suppressed_context_candidates > 0);

  // Match Python's all-or-none root policy. A large dictionary must not inject
  // an arbitrary first subset of phrase starts.
  asr_decoder::DecoderConfig dictionary_config;
  dictionary_config.context_policy = asr_decoder::ContextPolicy{};
  for (int token = 1; token <= 1000; ++token) {
    dictionary_config.context_token_ids.push_back({token});
  }
  std::vector<float> dictionary_values(1001, 0.0f);
  asr_decoder::CTCDecoder dictionary_decoder(dictionary_config);
  asr_decoder::DecodeOptions dictionary_options;
  dictionary_options.beam_size = 2;
  dictionary_options.token_beam_size = 2;
  dictionary_options.finalize = true;
  dictionary_options.return_gating_diagnostics = true;
  const auto dictionary_result = dictionary_decoder.prefix_beam_search(
      dictionary_values.data(), 1, dictionary_values.size(),
      dictionary_options);
  CHECK(dictionary_result.gating.has_value());
  CHECK(dictionary_result.gating->evaluated_context_candidates == 0);
  CHECK(dictionary_result.gating->gathered_context_candidates == 0);
  CHECK(dictionary_result.gating->context_candidate_limit_hits == 1);

  // Adaptive gating always examines up to the top eight acoustic tokens,
  // independently of the search token beam.
  const std::vector<float> gating_values{-5.0f, -0.1f, -0.2f,
                                         -0.3f, -0.4f, -2.0f};
  asr_decoder::DecoderConfig gating_config;
  gating_config.context_token_ids = {{5, 5}};
  asr_decoder::CTCDecoder narrow_gating(gating_config);
  asr_decoder::CTCDecoder wide_gating(gating_config);
  asr_decoder::DecodeOptions narrow_gating_options;
  narrow_gating_options.token_beam_size = 2;
  narrow_gating_options.return_gating_diagnostics = true;
  asr_decoder::DecodeOptions wide_gating_options = narrow_gating_options;
  wide_gating_options.token_beam_size = 6;
  const auto narrow_gating_result = narrow_gating.prefix_beam_search(
      gating_values.data(), 1, 6, narrow_gating_options);
  const auto wide_gating_result = wide_gating.prefix_beam_search(
      gating_values.data(), 1, 6, wide_gating_options);
  CHECK(narrow_gating_result.gating->mean_gating_acoustic_scale ==
        wide_gating_result.gating->mean_gating_acoustic_scale);

  // Candidate competition tightens the injection gate as in Python. This is
  // the protection that prevents a wide hotword set from over-triggering.
  asr_decoder::DecoderConfig pressure_config;
  pressure_config.context_token_ids = {{1}, {2}, {3}, {4}};
  pressure_config.context_policy = asr_decoder::ContextPolicy{};
  asr_decoder::DecoderConfig no_pressure_config = pressure_config;
  no_pressure_config.context_policy->gating.max_active_context_penalty = 0.0;
  const std::vector<float> pressure_values{-2.0f, -0.1f, -0.2f, -0.3f, -0.4f};
  asr_decoder::DecodeOptions pressure_options;
  pressure_options.token_beam_size = 2;
  pressure_options.return_gating_diagnostics = true;
  asr_decoder::CTCDecoder pressure_decoder(pressure_config);
  asr_decoder::CTCDecoder no_pressure_decoder(no_pressure_config);
  const auto pressure_result = pressure_decoder.prefix_beam_search(
      pressure_values.data(), 1, 5, pressure_options);
  const auto no_pressure_result = no_pressure_decoder.prefix_beam_search(
      pressure_values.data(), 1, 5, pressure_options);
  CHECK(pressure_result.gating->mean_gating_factor.has_value());
  CHECK(no_pressure_result.gating->mean_gating_factor.has_value());
  CHECK(*pressure_result.gating->mean_gating_factor <
        *no_pressure_result.gating->mean_gating_factor);

  // A zero completion bonus validates hotword IDs but disables all contextual
  // candidate and gating work, matching the Python no-op behavior.
  asr_decoder::DecoderConfig zero_bonus_config;
  zero_bonus_config.context_token_ids = {{4}};
  zero_bonus_config.context_policy = asr_decoder::ContextPolicy{};
  zero_bonus_config.context_policy->completion_bonus = 0.0;
  asr_decoder::CTCDecoder zero_bonus_decoder(zero_bonus_config);
  asr_decoder::DecodeOptions zero_bonus_options;
  zero_bonus_options.finalize = true;
  zero_bonus_options.return_gating_diagnostics = true;
  const auto zero_bonus_result = zero_bonus_decoder.prefix_beam_search(
      gating_values.data(), 1, 6, zero_bonus_options);
  CHECK(zero_bonus_result.gating.has_value());
  CHECK(!zero_bonus_result.gating->contextual_biasing);
  CHECK(!zero_bonus_result.gating->adaptive_context_gating);
  CHECK(zero_bonus_result.gating->evaluated_context_candidates == 0);

  asr_decoder::DecoderConfig exclusive_policy_config;
  exclusive_policy_config.context_policy = asr_decoder::ContextPolicy{};
  exclusive_policy_config.hotword_strength =
      asr_decoder::HotwordStrength::kAggressive;
  asr_decoder::CTCDecoder exclusive_policy_decoder(exclusive_policy_config);
  CHECK(!exclusive_policy_decoder.valid());

  // Greedy search uses frame argmax, standard CTC collapse, half-open spans,
  // and the maximum frame probability for each collapsed token.
  const std::vector<float> greedy_values{
      -5.0f, 0.0f,  -5.0f, -5.0f, -0.1f, -5.0f, 0.0f,  -5.0f, -5.0f,
      -5.0f, -0.2f, -5.0f, -5.0f, -5.0f, -0.3f, -5.0f, -5.0f, -0.1f};
  asr_decoder::DecoderConfig greedy_config;
  greedy_config.frame_shift_ms = 20.0;
  asr_decoder::CTCDecoder greedy_decoder(greedy_config);
  asr_decoder::GreedyOptions greedy_options;
  greedy_options.finalize = true;
  greedy_options.return_token_probabilities = true;
  const auto greedy_result =
      greedy_decoder.greedy_search(greedy_values.data(), 6, 3, greedy_options);
  CHECK((greedy_result.tokens[0] == std::vector<int>{1, 1, 2}));
  CHECK((greedy_result.timestamps[0] ==
         std::vector<asr_decoder::Timestamp>{
             {0, 2, 0.0, 40.0}, {3, 4, 60.0, 80.0}, {4, 6, 80.0, 120.0}}));
  CHECK(greedy_result.probabilities[0].size() == 3);
  CHECK(std::abs(greedy_result.probabilities[0][0] - 1.0) < 1e-12);
  CHECK(std::abs(greedy_result.probabilities[0][2] - std::exp(-0.1)) < 1e-6);

  // Greedy output is invariant to chunking, including timestamps and token
  // probabilities.
  asr_decoder::CTCDecoder greedy_stream(greedy_config);
  asr_decoder::GreedyOptions partial_greedy_options;
  partial_greedy_options.return_token_probabilities = true;
  greedy_stream.greedy_search(greedy_values.data(), 2, 3,
                              partial_greedy_options);
  const auto streamed_greedy = greedy_stream.greedy_search(
      greedy_values.data() + 6, 4, 3, greedy_options);
  CHECK(streamed_greedy.tokens == greedy_result.tokens);
  CHECK(streamed_greedy.timestamps == greedy_result.timestamps);
  CHECK(streamed_greedy.probabilities == greedy_result.probabilities);

  // Streams copy immutable configuration but never mutable decoding state.
  asr_decoder::CTCDecoder stream_template(greedy_config);
  auto first_stream = stream_template.create_stream();
  auto second_stream = stream_template.create_stream();
  const auto first_stream_result =
      first_stream.greedy_search(greedy_values.data(), 1, 3);
  const auto second_stream_result = second_stream.greedy_search(nullptr, 0, 3);
  CHECK((first_stream_result.tokens[0] == std::vector<int>{1}));
  CHECK(second_stream_result.tokens[0].empty());

  // A stream cannot switch search algorithms until reset or finalize.
  asr_decoder::CTCDecoder greedy_then_prefix({});
  greedy_then_prefix.greedy_search(greedy_values.data(), 1, 3);
  CHECK(greedy_then_prefix.prefix_beam_search(greedy_values.data(), 1, 3)
            .tokens.empty());
  CHECK(!greedy_then_prefix.valid());
  greedy_then_prefix.reset();
  CHECK(!greedy_then_prefix
             .prefix_beam_search(greedy_values.data(), 1, 3, final_options)
             .tokens.empty());

  asr_decoder::CTCDecoder prefix_then_greedy({});
  prefix_then_greedy.prefix_beam_search(greedy_values.data(), 1, 3);
  CHECK(prefix_then_greedy.greedy_search(greedy_values.data(), 1, 3)
            .tokens.empty());
  CHECK(!prefix_then_greedy.valid());
  prefix_then_greedy.prefix_beam_search(greedy_values.data(), 1, 3,
                                        final_options);
  CHECK(prefix_then_greedy.valid());
  CHECK(!prefix_then_greedy
             .greedy_search(greedy_values.data(), 1, 3, greedy_options)
             .tokens.empty());

  // Invalid greedy input does not lock or mutate the stream.
  asr_decoder::CTCDecoder invalid_greedy({});
  const std::vector<float> all_impossible(3, negative_infinity);
  CHECK(
      invalid_greedy.greedy_search(all_impossible.data(), 1, 3).tokens.empty());
  CHECK(!invalid_greedy.valid());
  CHECK(!invalid_greedy
             .prefix_beam_search(greedy_values.data(), 1, 3, final_options)
             .tokens.empty());
  return 0;
}
