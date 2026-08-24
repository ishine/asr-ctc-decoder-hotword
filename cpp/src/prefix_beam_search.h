#pragma once

#include "decoder_state.h"

namespace asr_decoder::internal {

DecodeResult run_prefix_beam_search(DecoderState& state, const float* values,
                                    size_t frames, size_t vocabulary_size,
                                    DecodeOptions options);

}  // namespace asr_decoder::internal
