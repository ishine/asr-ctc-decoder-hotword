#pragma once

#include <cstddef>
#include <memory>
#include <vector>

namespace asr_decoder {

enum class HotwordStrength {
  kConservative,
  kBalanced,
  kAggressive,
  kLowFalseActivation,
};

enum class DecodingQuality {
  kLowLatency,
  kBalanced,
  kHighAccuracy,
};

struct DecoderConfig {
  std::vector<std::vector<int>> context_token_ids;
  HotwordStrength hotword_strength = HotwordStrength::kBalanced;
  DecodingQuality decoding_quality = DecodingQuality::kBalanced;
  int blank_id = 0;
  double frame_shift_ms = 0.0;
};

struct DecodeOptions {
  size_t beam_size = 0;
  size_t token_beam_size = 0;
  bool finalize = false;
  bool return_token_probabilities = false;
};

struct Timestamp {
  size_t start_frame = 0;
  size_t end_frame = 0;
  double start_ms = -1.0;
  double end_ms = -1.0;

  bool operator==(const Timestamp& other) const {
    return start_frame == other.start_frame && end_frame == other.end_frame &&
           start_ms == other.start_ms && end_ms == other.end_ms;
  }
};

struct HypothesisScore {
  double acoustic = 0.0;
  double contextual_bias = 0.0;
  double total = 0.0;
};

struct DecodeResult {
  std::vector<std::vector<int>> tokens;
  std::vector<std::vector<Timestamp>> timestamps;
  std::vector<std::vector<double>> probabilities;
  std::vector<HypothesisScore> scores;
};

class CTCDecoder {
 public:
  explicit CTCDecoder(DecoderConfig config);
  ~CTCDecoder();

  CTCDecoder(CTCDecoder&&) noexcept;
  CTCDecoder& operator=(CTCDecoder&&) noexcept;
  CTCDecoder(const CTCDecoder&) = delete;
  CTCDecoder& operator=(const CTCDecoder&) = delete;

  bool valid() const;
  const char* error() const;
  void reset();

  DecodeResult prefix_beam_search(const float* log_probabilities, size_t frames,
                                  size_t vocabulary_size,
                                  DecodeOptions options = {});

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace asr_decoder
