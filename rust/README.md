# asr-decoder (Rust)

Python 实现的纯 Rust 对应版本，支持流式贪心搜索、Prefix Beam Search、N-best、token
概率与时间戳，以及基于 token ID 的热词偏置。crate 不依赖具体推理框架，也没有第三方依赖。

输入为连续的、按行存储的 `(frames, vocabulary_size)` `f32` log probabilities：

```rust
use asr_decoder::{
    CTCDecoder, DecoderConfig, LogProbabilities, PrefixBeamSearchOptions,
};

let values = vec![
    -5.0, -0.1, -4.0,
    -0.1, -5.0, -4.0,
    -5.0, -4.0, -0.1,
];
let probabilities = LogProbabilities::new(&values, 3, 3)?;

let mut decoder = CTCDecoder::new(DecoderConfig {
    blank_id: 0,
    frame_shift_ms: Some(40.0),
    ..DecoderConfig::default()
})?;
let result = decoder.prefix_beam_search(
    probabilities,
    PrefixBeamSearchOptions {
        finalize: true,
        ..PrefixBeamSearchOptions::default()
    },
)?;
assert_eq!(result.tokens[0], vec![1, 2]);
# Ok::<(), asr_decoder::DecoderError>(())
```

流式输入时，对同一个 decoder 连续调用搜索方法。中间块保持 `finalize: false`，最后一块设置
`finalize: true`。也可以从模板调用 `create_stream()` 创建共享配置、状态独立的流。

热词直接使用模型 tokenizer 的 token ID：

```rust
use asr_decoder::{CTCDecoder, DecoderConfig, HotwordStrength};

let decoder = CTCDecoder::new(DecoderConfig {
    context_token_ids: vec![vec![23, 41], vec![81, 19, 7]],
    hotword_strength: Some(HotwordStrength::Balanced),
    ..DecoderConfig::default()
})?;
# Ok::<(), asr_decoder::DecoderError>(())
```

`ContextPolicy` 和 `hotword_strength` 二选一；Rust 端不负责文本分词，避免与具体 tokenizer
绑定。
