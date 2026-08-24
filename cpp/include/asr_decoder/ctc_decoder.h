#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
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

struct AdaptiveGatingConfig {
  bool enabled = true;
  double reference_top_spread = 4.0;
  double min_acoustic_scale = 0.5;
  double max_acoustic_scale = 2.0;
  double min_entropy_confidence_factor = 0.65;
  double max_active_context_penalty = 0.35;
};

struct ContextPolicy {
  double completion_bonus = 3.5;
  double context_token_prune_threshold = 1.5;
  size_t max_injected_candidates = 8;
  size_t max_completed_contexts_per_token = 8;
  AdaptiveGatingConfig gating;

  static ContextPolicy conservative();
  static ContextPolicy balanced();
  static ContextPolicy aggressive();
  static ContextPolicy low_false_activation();
};

struct SentencePieceConfig {
  std::string model_path;
  std::unordered_map<std::string, int> symbol_table;
  bool add_word_boundary = true;
};

struct SentencePieceTokenizationResult {
  std::vector<std::vector<int>> context_token_ids;
  std::vector<int> word_boundary_token_ids;
  std::string error;

  explicit operator bool() const { return error.empty(); }
};

SentencePieceTokenizationResult tokenize_sentencepiece_contexts(
    const std::vector<std::string>& contexts,
    const SentencePieceConfig& config);

struct DecoderConfig {
  std::vector<std::vector<int>> context_token_ids;
  HotwordStrength hotword_strength = HotwordStrength::kBalanced;
  DecodingQuality decoding_quality = DecodingQuality::kBalanced;
  int blank_id = 0;
  double frame_shift_ms = 0.0;
  std::optional<ContextPolicy> context_policy;
  std::vector<int> word_boundary_token_ids;
  std::vector<std::string> contexts;
  SentencePieceConfig sentencepiece;
};

struct DecodeOptions {
  size_t beam_size = 0;
  size_t token_beam_size = 0;
  bool finalize = false;
  bool return_token_probabilities = false;
  bool return_gating_diagnostics = false;
};

struct GreedyOptions {
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

struct GatingDiagnostics {
  bool contextual_biasing = false;
  bool adaptive_context_gating = false;
  size_t gating_frames = 0;
  std::optional<double> mean_gating_acoustic_scale;
  std::optional<double> mean_gating_factor;
  size_t nonblank_gating_frames = 0;
  size_t evaluated_context_candidates = 0;
  size_t gathered_context_candidates = 0;
  size_t max_context_candidates_per_frame = 0;
  size_t context_candidate_limit_hits = 0;
  bool word_boundary_aware_context = false;
  size_t boundary_suppressed_context_candidates = 0;
  size_t beam_size = 0;
  size_t token_beam_size = 0;
};

struct DecodeResult {
  std::vector<std::vector<int>> tokens;
  std::vector<std::vector<Timestamp>> timestamps;
  std::vector<std::vector<double>> probabilities;
  std::vector<HypothesisScore> scores;
  std::optional<GatingDiagnostics> gating;
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
  CTCDecoder create_stream() const;

  DecodeResult greedy_search(const float* log_probabilities, size_t frames,
                             size_t vocabulary_size,
                             GreedyOptions options = {});

  DecodeResult prefix_beam_search(const float* log_probabilities, size_t frames,
                                  size_t vocabulary_size,
                                  DecodeOptions options = {});

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace asr_decoder
