#include "asr_decoder/ctc_decoder.h"

#include <utility>

#include "decoder_state.h"
#include "prefix_beam_search.h"

namespace asr_decoder {

struct CTCDecoder::Impl : internal::DecoderState {
  using DecoderState::DecoderState;
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
  return impl_ ? internal::run_prefix_beam_search(*impl_, values, frames,
                                                  vocabulary_size, options)
               : DecodeResult{};
}

}  // namespace asr_decoder
