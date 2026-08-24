//! A model-agnostic streaming CTC decoder.
//!
//! Inputs are log probabilities in row-major `(frames, vocabulary_size)`
//! layout. The crate implements greedy search, prefix beam search, N-best
//! results, token timestamps, and contextual hotword biasing.

mod context_graph;
mod context_policy;
mod decoder;
mod error;
mod greedy_search;
mod input;
mod prefix_beam_search;
mod prefix_score;
mod result;

pub use context_policy::{AdaptiveGatingConfig, ContextPolicy, DecodingQuality, HotwordStrength};
pub use decoder::{CTCDecoder, DecoderConfig, PrefixBeamSearchOptions};
pub use error::DecoderError;
pub use input::LogProbabilities;
pub use result::{DecodeResult, GatingDiagnostics, HypothesisScore, Timestamp};
