# asr-decoder (C++17)

无第三方依赖的 C++17 CTC Prefix Beam Search 与热词偏置库。输入为连续、按行存储的
`(frames, vocabulary_size)` `float` log probabilities。

```cpp
#include <asr_decoder/ctc_decoder.h>

asr_decoder::DecoderConfig config;
config.blank_id = 0;
config.frame_shift_ms = 40.0;
config.context_token_ids = {{23, 41}, {81, 19, 7}};
asr_decoder::CTCDecoder decoder(std::move(config));

asr_decoder::DecodeOptions options;
options.finalize = true;
auto result = decoder.prefix_beam_search(values, frames, vocabulary, options);
```

使用 CMake：

```cmake
add_subdirectory(asr-decoder)
target_link_libraries(your-target PRIVATE asr_decoder::asr_decoder)
```

同一个 `CTCDecoder` 可连续接收多个音频块。流内的词表大小、`beam_size` 和
`token_beam_size` 必须保持不变；最后一块设置 `finalize = true`，或显式调用
`reset()` 开始下一条音频。模型 tokenizer 以及“热词文本到 token ID”的转换由调用方负责。
接口不抛异常；构造或搜索失败时 `valid()` 返回 `false`，具体原因由 `error()` 返回。

实现按职责拆分：`context_graph` 管理热词状态图，`context_policy` 管理预设和自适应门控，
`prefix_score` 管理 CTC 前缀分数与对齐，`prefix_beam_search` 执行搜索，`decoder_state`
保存流式状态，`ctc_decoder` 只保留公开 API 适配。
