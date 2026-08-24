#include "asr_decoder/ctc_decoder.h"

#include <cmath>
#include <cstdio>
#include <limits>
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
  return 0;
}
