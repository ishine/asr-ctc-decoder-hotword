# asr-decoder (C++17)

无第三方依赖的 C++17 CTC Prefix Beam Search 与热词偏置库。输入为连续、按行存储的
`(frames, vocabulary_size)` `float` log probabilities。

```cpp
#include <asr_decoder/ctc_decoder.h>

asr_decoder::DecoderConfig config;
config.blank_id = 0;
config.frame_shift_ms = 40.0;
config.context_token_ids = {{23, 41}, {81, 19, 7}};
config.hotword_strength = asr_decoder::HotwordStrength::kBalanced;
asr_decoder::CTCDecoder decoder(std::move(config));

asr_decoder::DecodeOptions options;
options.finalize = true;
auto result = decoder.prefix_beam_search(values, frames, vocabulary, options);
```

只需要标准 CTC 贪心解码时：

```cpp
asr_decoder::GreedyOptions options;
options.finalize = true;
auto result = decoder.greedy_search(values, frames, vocabulary, options);
```

需要与 Python 的高级热词策略逐项对齐时，可以直接配置策略和词边界：

```cpp
asr_decoder::DecoderConfig config;
config.context_token_ids = {{23, 41}};
config.context_policy = asr_decoder::ContextPolicy::balanced();
config.context_policy->gating.enabled = true;
config.word_boundary_token_ids = {17, 29};
```

设置 `context_policy` 时不要再选择非默认的 `hotword_strength`。
`word_boundary_token_ids` 应使用模型 tokenizer 中表示词首的 token ID。

使用 CMake：

```cmake
add_subdirectory(asr-decoder)
target_link_libraries(your-target PRIVATE asr_decoder::asr_decoder)
```

同一个 `CTCDecoder` 可连续接收多个音频块。流内的词表大小、`beam_size` 和
`token_beam_size` 必须保持不变；最后一块设置 `finalize = true`，或显式调用
`reset()` 开始下一条音频。后续块将 `beam_size` 或 `token_beam_size` 保持为 0 时，
会复用首块已经锁定的值。模型 tokenizer 以及“热词文本到 token ID”的转换由调用方负责。
接口不抛异常；构造或搜索失败时 `valid()` 返回 `false`，具体原因由 `error()` 返回。
同一条流中不能混用贪心搜索和 Prefix Beam Search；`finalize` 或 `reset()` 后可以切换。
需要从同一份配置创建互不影响的解码状态时，使用 `decoder.create_stream()`。

实现按职责拆分：`context_graph` 管理热词状态图，`context_policy` 管理预设和自适应门控，
`prefix_score` 管理 CTC 前缀分数与对齐，`prefix_beam_search` 执行搜索，`decoder_state`
保存流式状态，`ctc_decoder` 只保留公开 API 适配。
